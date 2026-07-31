"""
mlflow_tracker.py
-----------------
Lightweight MLflow experiment tracker for the QLoRA fine-tuning run.

Usage (standalone):
    python mlflow_tracker.py

Usage (from train_qlora.py):
    from mlflow_tracker import MLflowTracker
    tracker = MLflowTracker(experiment_name="sql-qlora-v1")
    tracker.start_run(run_name="llama-3.2-3b-r16")
    tracker.log_params({...})
    tracker.log_metrics({...})
    tracker.end_run()
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import mlflow
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class MLflowTracker:
    """
    Thin wrapper around the MLflow Python SDK that handles experiment /
    run lifecycle and provides helper methods for logging training artefacts.
    """

    def __init__(
        self,
        experiment_name: str = "sql-text2sql-qlora",
        tracking_uri: str | None = None,
    ) -> None:
        """
        Args:
            experiment_name: Name of the MLflow experiment bucket.
            tracking_uri:    Remote MLflow server URI, e.g.
                             "http://localhost:5000". Defaults to the
                             local ./mlruns directory.
        """
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri or os.getenv("MLFLOW_TRACKING_URI", "mlruns")
        self._run = None

        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(experiment_name)
        logger.info("MLflow tracking URI: %s | Experiment: %s", self.tracking_uri, experiment_name)

    # ------------------------------------------------------------------
    # Run lifecycle
    # ------------------------------------------------------------------

    def start_run(self, run_name: str | None = None, tags: dict | None = None) -> None:
        """Start a new MLflow run."""
        self._run = mlflow.start_run(run_name=run_name, tags=tags or {})
        logger.info("MLflow run started — run_id: %s", self._run.info.run_id)

    def end_run(self, status: str = "FINISHED") -> None:
        """End the active MLflow run."""
        mlflow.end_run(status=status)
        logger.info("MLflow run ended with status: %s", status)

    # ------------------------------------------------------------------
    # Logging helpers
    # ------------------------------------------------------------------

    def log_params(self, params: dict[str, Any]) -> None:
        """Log a flat dictionary of hyper-parameters."""
        mlflow.log_params(params)
        logger.info("Logged %d parameters.", len(params))

    def log_metric(self, key: str, value: float, step: int | None = None) -> None:
        """Log a single scalar metric."""
        mlflow.log_metric(key, value, step=step)

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        """Log a dictionary of scalar metrics at a given step."""
        mlflow.log_metrics(metrics, step=step)

    def log_artifact(self, local_path: str, artifact_path: str | None = None) -> None:
        """Upload a local file or directory as an MLflow artefact."""
        mlflow.log_artifact(local_path, artifact_path=artifact_path)
        logger.info("Logged artifact: %s", local_path)

    def log_model_summary(self, summary_text: str) -> None:
        """Save a plain-text model architecture summary as an artefact."""
        tmp = Path("model_summary.txt")
        tmp.write_text(summary_text)
        self.log_artifact(str(tmp))
        tmp.unlink(missing_ok=True)

    # ------------------------------------------------------------------
    # Convenience: log training loop metrics from a Trainer callback
    # ------------------------------------------------------------------

    def build_trainer_callback(self):
        """
        Returns a HuggingFace TrainerCallback that pipes training metrics
        directly into this MLflow tracker instance.

        Usage:
            trainer = SFTTrainer(..., callbacks=[tracker.build_trainer_callback()])
        """
        from transformers import TrainerCallback  # lazy import

        tracker_ref = self  # closure reference

        class _MLflowCallback(TrainerCallback):
            def on_log(self, args, state, control, logs=None, **kwargs):
                if logs is None:
                    return
                step = state.global_step
                metrics = {k: v for k, v in logs.items() if isinstance(v, (int, float))}
                tracker_ref.log_metrics(metrics, step=step)

        return _MLflowCallback()


# ---------------------------------------------------------------------------
# CLI smoke-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    tracker = MLflowTracker(experiment_name="smoke-test")
    tracker.start_run(run_name="demo-run", tags={"env": "local"})

    tracker.log_params(
        {
            "model": "unsloth/Llama-3.2-3B-Instruct",
            "lora_r": 16,
            "lora_alpha": 16,
            "batch_size": 4,
            "epochs": 3,
        }
    )

    for step in range(5):
        fake_loss = 2.0 - step * 0.3
        tracker.log_metrics({"train/loss": fake_loss, "train/lr": 2e-4}, step=step)
        time.sleep(0.1)

    tracker.end_run()
    print("✅ MLflow smoke-test complete. Open the UI with: mlflow ui --port 5000")
