"""
train_qlora.py
--------------
QLoRA fine-tuning of Llama-3.2-3B-Instruct on the b-mc2/sql-create-context
dataset using Unsloth for 2-4× faster training and 60 % lower VRAM usage.

Run on Google Colab (T4 GPU):
    !pip install -q unsloth trl peft bitsandbytes mlflow evaluate
    !python train_qlora.py

Run locally (requires CUDA):
    pip install -r requirements.txt
    python train_qlora.py
"""

from __future__ import annotations

import os
import time
import logging
from pathlib import Path

import torch
from datasets import DatasetDict
from transformers import TrainingArguments
from trl import SFTTrainer, SFTConfig
from peft import PeftModel
from unsloth import FastLanguageModel

# ── Project-local imports ──────────────────────────────────────────────────────
from dataset_formatter import load_and_format
from mlflow_tracker import MLflowTracker

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# 1.  CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────

CFG = {
    # ── Model ────────────────────────────────────────────────────────────────
    "base_model":        "unsloth/Llama-3.2-3B-Instruct",
    "output_dir":        "./adapter_weights",
    "model_version":     "v1.0.0",

    # ── QLoRA ────────────────────────────────────────────────────────────────
    "lora_r":            16,
    "lora_alpha":        16,
    "lora_dropout":      0.05,
    "target_modules":    [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    "bias":              "none",

    # ── Quantization ─────────────────────────────────────────────────────────
    "load_in_4bit":      True,
    "bnb_4bit_compute_dtype": "bfloat16",  # "float16" on older Colab GPUs
    "max_seq_length":    1024,

    # ── Dataset ──────────────────────────────────────────────────────────────
    "dataset_name":      "b-mc2/sql-create-context",
    "max_samples":       None,       # set e.g. 5000 for quick smoke-testing
    "train_split":       0.95,

    # ── Training hyper-parameters ─────────────────────────────────────────────
    "num_train_epochs":  3,
    "per_device_train_batch_size": 4,
    "gradient_accumulation_steps": 4,    # effective batch = 16
    "warmup_ratio":      0.05,
    "learning_rate":     2e-4,
    "weight_decay":      0.01,
    "lr_scheduler_type": "cosine",
    "fp16":              not torch.cuda.is_bf16_supported(),
    "bf16":              torch.cuda.is_bf16_supported(),
    "logging_steps":     10,
    "eval_steps":        100,
    "save_steps":        200,
    "save_total_limit":  2,
    "seed":              42,

    # ── MLflow ────────────────────────────────────────────────────────────────
    "mlflow_experiment": "sql-qlora-llama-3.2-3b",
    "mlflow_run_name":   f"qlora-r16-{time.strftime('%Y%m%d-%H%M%S')}",
}


# ──────────────────────────────────────────────────────────────────────────────
# 2.  MODEL + TOKENIZER LOADING
# ──────────────────────────────────────────────────────────────────────────────

def load_model_and_tokenizer(cfg: dict):
    """
    Load the base model with 4-bit NF4 quantization via Unsloth and attach
    LoRA adapters according to the QLoRA configuration.

    Returns:
        (model, tokenizer) — model is ready for SFTTrainer.
    """
    logger.info("Loading base model: %s", cfg["base_model"])

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg["base_model"],
        max_seq_length=cfg["max_seq_length"],
        dtype=None,                 # auto-detect
        load_in_4bit=cfg["load_in_4bit"],
    )

    logger.info("Attaching LoRA adapters (r=%d, alpha=%d)…", cfg["lora_r"], cfg["lora_alpha"])

    model = FastLanguageModel.get_peft_model(
        model,
        r=cfg["lora_r"],
        lora_alpha=cfg["lora_alpha"],
        lora_dropout=cfg["lora_dropout"],
        target_modules=cfg["target_modules"],
        bias=cfg["bias"],
        use_gradient_checkpointing="unsloth",   # Unsloth's memory-efficient variant
        random_state=cfg["seed"],
        use_rslora=False,
    )

    # Ensure EOS token is set (required for generative training)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    logger.info(
        "Trainable params: %s (%.2f %% of %s total)",
        f"{trainable:,}", 100 * trainable / total, f"{total:,}",
    )

    return model, tokenizer


# ──────────────────────────────────────────────────────────────────────────────
# 3.  TRAINER SETUP
# ──────────────────────────────────────────────────────────────────────────────

def build_trainer(
    model,
    tokenizer,
    dataset: DatasetDict,
    cfg: dict,
    mlflow_callback=None,
) -> SFTTrainer:
    """
    Construct a TRL SFTTrainer with the configured training arguments.

    Args:
        model:            LoRA-patched language model.
        tokenizer:        Matching tokenizer.
        dataset:          DatasetDict with 'train' and 'test' splits.
        cfg:              Global configuration dict.
        mlflow_callback:  Optional HuggingFace callback for MLflow logging.

    Returns:
        Configured SFTTrainer instance.
    """
    sft_config = SFTConfig(
        output_dir=cfg["output_dir"],
        num_train_epochs=cfg["num_train_epochs"],
        per_device_train_batch_size=cfg["per_device_train_batch_size"],
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        warmup_steps=10,
        learning_rate=cfg["learning_rate"],
        weight_decay=cfg["weight_decay"],
        lr_scheduler_type=cfg["lr_scheduler_type"],
        fp16=cfg["fp16"],
        bf16=cfg["bf16"],
        logging_steps=cfg["logging_steps"],
        eval_steps=cfg["eval_steps"],
        evaluation_strategy="steps",
        save_steps=cfg["save_steps"],
        save_total_limit=cfg["save_total_limit"],
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to="none",           # we handle MLflow ourselves
        seed=cfg["seed"],
        dataset_text_field="text",  # column produced by dataset_formatter
        max_seq_length=cfg["max_seq_length"],
        packing=False,              # disable sample packing for clearer eval
    )

    callbacks = [mlflow_callback] if mlflow_callback else []

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        args=sft_config,
        callbacks=callbacks,
    )
    return trainer


# ──────────────────────────────────────────────────────────────────────────────
# 4.  EVALUATION HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_exact_match(model, tokenizer, test_dataset, n_samples: int = 50, cfg: dict = None) -> dict:
    """
    Run greedy decoding on a sample of the test set and compute exact-match
    accuracy between generated SQL and the ground-truth SQL.

    Args:
        model:        Fine-tuned model (in inference mode).
        tokenizer:    Matching tokenizer.
        test_dataset: HuggingFace Dataset with a 'text' column.
        n_samples:    Number of samples to evaluate (keep small for speed).
        cfg:          Configuration dict for generation parameters.

    Returns:
        dict with keys 'exact_match_accuracy' and 'n_evaluated'.
    """
    if cfg is None:
        cfg = {}

    FastLanguageModel.for_inference(model)     # enable Unsloth's 2× faster inference

    correct = 0
    subset = test_dataset.select(range(min(n_samples, len(test_dataset))))

    for example in subset:
        full_text: str = example["text"]

        # Split prompt / reference at the assistant turn delimiter
        assistant_tag = "<|start_header_id|>assistant<|end_header_id|>"
        if assistant_tag not in full_text:
            continue

        prompt_part, reference_sql = full_text.split(assistant_tag, 1)
        reference_sql = reference_sql.replace("<|eot_id|>", "").strip()
        prompt_part  = prompt_part + assistant_tag + "\n"

        inputs = tokenizer(prompt_part, return_tensors="pt").to(model.device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=cfg.get("max_new_tokens", 256),
                temperature=cfg.get("temperature", 0.0),
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        generated = tokenizer.decode(
            output_ids[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        ).strip()

        # Normalise whitespace before comparing
        if generated.lower().split() == reference_sql.lower().split():
            correct += 1

    accuracy = correct / len(subset) if subset else 0.0
    logger.info(
        "Exact-match accuracy: %.2f %% (%d / %d)",
        accuracy * 100, correct, len(subset),
    )
    return {"exact_match_accuracy": accuracy, "n_evaluated": len(subset)}


# ──────────────────────────────────────────────────────────────────────────────
# 5.  MAIN TRAINING PIPELINE
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    # ── 5a. Initialise MLflow ─────────────────────────────────────────────────
    tracker = MLflowTracker(experiment_name=CFG["mlflow_experiment"])
    tracker.start_run(
        run_name=CFG["mlflow_run_name"],
        tags={"model": CFG["base_model"], "dataset": CFG["dataset_name"]},
    )
    tracker.log_params({k: v for k, v in CFG.items() if not isinstance(v, list)})
    tracker.log_params({"target_modules": ",".join(CFG["target_modules"])})

    try:
        # ── 5b. Dataset ───────────────────────────────────────────────────────
        dataset = load_and_format(
            dataset_name=CFG["dataset_name"],
            train_split=CFG["train_split"],
            max_samples=CFG["max_samples"],
            seed=CFG["seed"],
        )
        tracker.log_params({
            "train_samples": len(dataset["train"]),
            "eval_samples":  len(dataset["test"]),
        })

        # ── 5c. Model ─────────────────────────────────────────────────────────
        model, tokenizer = load_model_and_tokenizer(CFG)

        # ── 5d. Trainer ───────────────────────────────────────────────────────
        mlflow_cb = tracker.build_trainer_callback()
        trainer   = build_trainer(model, tokenizer, dataset, CFG, mlflow_cb)

        # ── 5e. Train ─────────────────────────────────────────────────────────
        logger.info("Starting QLoRA fine-tuning…")
        t0 = time.time()
        train_result = trainer.train()
        elapsed = time.time() - t0

        logger.info("Training complete in %.1f s.", elapsed)
        tracker.log_metrics({
            "training_time_s":  elapsed,
            "train_loss":       train_result.training_loss,
        })

        # ── 5f. Save adapter weights ──────────────────────────────────────────
        output_path = Path(CFG["output_dir"])
        output_path.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(output_path))
        tokenizer.save_pretrained(str(output_path))
        logger.info("Adapter weights saved to: %s", output_path.resolve())
        tracker.log_artifact(str(output_path), artifact_path="adapter_weights")

        # ── 5g. Evaluation ────────────────────────────────────────────────────
        logger.info("Running exact-match evaluation on %d test samples…", 50)
        eval_metrics = evaluate_exact_match(
            model, tokenizer, dataset["test"],
            n_samples=50, cfg={"max_new_tokens": 256, "temperature": 0.0},
        )
        tracker.log_metrics(eval_metrics)

        logger.info("Pipeline complete. Metrics: %s", eval_metrics)
        tracker.end_run(status="FINISHED")

    except Exception:
        logger.exception("Training pipeline failed.")
        tracker.end_run(status="FAILED")
        raise


if __name__ == "__main__":
    main()
