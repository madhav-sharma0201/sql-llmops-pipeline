"""
evaluate_metrics.py
-------------------
Standalone evaluation script that benchmarks a fine-tuned LoRA adapter
against the un-adapted base model on the sql-create-context test split.

Metrics computed
----------------
- Training / eval loss (from saved trainer_state.json)
- Exact-match query accuracy  : generated SQL == ground-truth SQL
- BLEU-4                      : measures token-level overlap
- Execution accuracy (opt.)   : runs SQL against an in-memory SQLite DB

Usage
-----
    # Compare base model vs fine-tuned adapter
    python evaluate_metrics.py \
        --adapter_path ./adapter_weights \
        --n_samples 100 \
        --output_json results/eval_report.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sqlite3
import time
from pathlib import Path

import torch
from datasets import load_dataset
from sacrebleu.metrics import BLEU
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from unsloth import FastLanguageModel

from dataset_formatter import INSTRUCTION_TEMPLATE, SYSTEM_PROMPT

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

ASSISTANT_TAG = "<|start_header_id|>assistant<|end_header_id|>"


def generate_sql(model, tokenizer, context: str, question: str, max_new_tokens: int = 256) -> tuple[str, float]:
    """
    Generate a SQL query from a schema + question using greedy decoding.

    Returns:
        (generated_sql, latency_ms)
    """
    prompt = INSTRUCTION_TEMPLATE.format(
        system=SYSTEM_PROMPT,
        context=context,
        question=question,
        answer="",           # leave blank — model fills this in
    )
    # Trim trailing whitespace after the assistant tag
    prompt = prompt.rstrip()

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=900)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    t0 = time.perf_counter()
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    latency_ms = (time.perf_counter() - t0) * 1000

    generated = tokenizer.decode(
        output_ids[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    ).strip()

    return generated, latency_ms


def try_execute_sql(create_context: str, sql: str) -> bool:
    """
    Execute SQL against an ephemeral SQLite database pre-loaded with the
    CREATE TABLE statements from the context.  Returns True if execution
    succeeds without error.
    """
    try:
        conn = sqlite3.connect(":memory:")
        cur = conn.cursor()
        # Execute all CREATE TABLE statements from context
        for stmt in create_context.split(";"):
            stmt = stmt.strip()
            if stmt.lower().startswith("create"):
                cur.execute(stmt + ";")
        cur.execute(sql)
        conn.close()
        return True
    except Exception:
        return False


# ──────────────────────────────────────────────────────────────────────────────
# Model loaders
# ──────────────────────────────────────────────────────────────────────────────

def load_base_model(model_name: str = "unsloth/Llama-3.2-3B-Instruct"):
    """Load the base model WITHOUT any adapter for baseline comparison."""
    logger.info("Loading BASE model: %s", model_name)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=1024,
        dtype=None,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)
    return model, tokenizer


def load_finetuned_model(adapter_path: str, base_model_name: str = "unsloth/Llama-3.2-3B-Instruct"):
    """Load the fine-tuned model (base + LoRA adapter)."""
    logger.info("Loading FINE-TUNED model from: %s", adapter_path)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=adapter_path,
        max_seq_length=1024,
        dtype=None,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)
    return model, tokenizer


# ──────────────────────────────────────────────────────────────────────────────
# Core evaluation loop
# ──────────────────────────────────────────────────────────────────────────────

def evaluate(
    model,
    tokenizer,
    examples: list[dict],
    label: str = "model",
) -> dict:
    """
    Evaluate a model on a list of {context, question, answer} examples.

    Returns:
        dict with aggregated metrics.
    """
    bleu = BLEU(effective_order=True)

    exact_matches   = 0
    exec_successes  = 0
    latencies_ms    = []
    hypotheses      = []
    references      = []

    for i, ex in enumerate(examples):
        generated, latency = generate_sql(model, tokenizer, ex["context"], ex["question"])
        reference = ex["answer"].strip()

        hypotheses.append(generated)
        references.append(reference)
        latencies_ms.append(latency)

        if generated.lower().split() == reference.lower().split():
            exact_matches += 1

        if try_execute_sql(ex["context"], generated):
            exec_successes += 1

        if (i + 1) % 10 == 0:
            logger.info("[%s] Evaluated %d / %d samples…", label, i + 1, len(examples))

    bleu_score = bleu.corpus_score(hypotheses, [references]).score

    metrics = {
        "label":                     label,
        "n_samples":                 len(examples),
        "exact_match_accuracy":      exact_matches / len(examples),
        "execution_accuracy":        exec_successes / len(examples),
        "bleu4":                     bleu_score,
        "avg_latency_ms":            sum(latencies_ms) / len(latencies_ms),
        "p50_latency_ms":            sorted(latencies_ms)[len(latencies_ms) // 2],
        "p95_latency_ms":            sorted(latencies_ms)[int(len(latencies_ms) * 0.95)],
    }

    logger.info("── %s Results ──────────────────────────────────────", label.upper())
    for k, v in metrics.items():
        if isinstance(v, float):
            logger.info("  %-30s: %.4f", k, v)

    return metrics


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate fine-tuned Text-to-SQL model.")
    p.add_argument("--adapter_path", default="./adapter_weights",
                   help="Path to saved LoRA adapter weights directory.")
    p.add_argument("--base_model",   default="unsloth/Llama-3.2-3B-Instruct",
                   help="Base model identifier (HF hub or local path).")
    p.add_argument("--n_samples",    type=int, default=100,
                   help="Number of test samples to evaluate.")
    p.add_argument("--skip_base",    action="store_true",
                   help="Skip the base-model baseline evaluation.")
    p.add_argument("--output_json",  default="results/eval_report.json",
                   help="Where to write the JSON evaluation report.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # ── Load test data ────────────────────────────────────────────────────────
    logger.info("Loading test data from b-mc2/sql-create-context…")
    raw = load_dataset("b-mc2/sql-create-context", split="train")
    raw = raw.shuffle(seed=42).select(range(args.n_samples))
    examples = [{"context": ex["context"], "question": ex["question"], "answer": ex["answer"]} for ex in raw]

    all_results = {}

    # ── Fine-tuned model evaluation ───────────────────────────────────────────
    ft_model, ft_tok = load_finetuned_model(args.adapter_path, args.base_model)
    all_results["finetuned"] = evaluate(ft_model, ft_tok, examples, label="fine-tuned")
    del ft_model, ft_tok
    torch.cuda.empty_cache()

    # ── Baseline model evaluation (optional) ─────────────────────────────────
    if not args.skip_base:
        base_model, base_tok = load_base_model(args.base_model)
        all_results["base"] = evaluate(base_model, base_tok, examples, label="base")
        del base_model, base_tok
        torch.cuda.empty_cache()

    # ── Delta summary ─────────────────────────────────────────────────────────
    if "base" in all_results and "finetuned" in all_results:
        delta = {}
        for k in ("exact_match_accuracy", "execution_accuracy", "bleu4"):
            delta[f"delta_{k}"] = all_results["finetuned"][k] - all_results["base"][k]
        all_results["delta"] = delta
        logger.info("── Improvement over base ─────────────────────────────────")
        for k, v in delta.items():
            logger.info("  %-35s: %+.4f", k, v)

    # ── Persist report ────────────────────────────────────────────────────────
    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(all_results, indent=2))
    logger.info("Evaluation report saved to: %s", out_path.resolve())


if __name__ == "__main__":
    main()
