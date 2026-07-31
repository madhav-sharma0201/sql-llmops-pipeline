"""
dataset_formatter.py
--------------------
Converts the b-mc2/sql-create-context Hugging Face dataset into an
Alpaca-style instruction format suitable for QLoRA fine-tuning.

Each example becomes:
    [INST] You are an expert SQL assistant. Given the database schema and a
    natural language question, write the correct SQL query.

    ### Database Schema:
    <CREATE TABLE ...>

    ### Question:
    <natural language question>

    ### SQL Query:
    [/INST] <SQL answer>
"""

from __future__ import annotations

from datasets import load_dataset, DatasetDict, Dataset
from typing import Optional
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "You are an expert SQL assistant. Given the database schema and a natural "
    "language question, write the correct SQL query. Output only the SQL — no "
    "explanation, no markdown fences."
)

INSTRUCTION_TEMPLATE = """\
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
{system}<|eot_id|><|start_header_id|>user<|end_header_id|>

### Database Schema:
{context}

### Question:
{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
{answer}<|eot_id|>"""


def format_example(example: dict) -> dict:
    """
    Map a single dataset row into a formatted instruction string.

    Args:
        example: dict with keys 'context', 'question', 'answer'

    Returns:
        dict with a single key 'text' containing the full prompt+response.
    """
    return {
        "text": INSTRUCTION_TEMPLATE.format(
            system=SYSTEM_PROMPT,
            context=example["context"].strip(),
            question=example["question"].strip(),
            answer=example["answer"].strip(),
        )
    }


def load_and_format(
    dataset_name: str = "b-mc2/sql-create-context",
    train_split: float = 0.95,
    max_samples: Optional[int] = None,
    seed: int = 42,
) -> DatasetDict:
    """
    Load the sql-create-context dataset from Hugging Face, format it, and
    return a DatasetDict with 'train' and 'test' splits.

    Args:
        dataset_name:  HuggingFace dataset identifier.
        train_split:   Fraction of data to use for training (rest → test).
        max_samples:   Cap the total number of samples (useful for debugging).
        seed:          Random seed for reproducibility.

    Returns:
        DatasetDict with 'train' and 'test' splits, each containing a 'text' column.
    """
    logger.info("Loading dataset: %s", dataset_name)
    raw: DatasetDict = load_dataset(dataset_name)

    # The dataset ships with a single 'train' split — we create a test set.
    full_dataset: Dataset = raw["train"]

    if max_samples:
        full_dataset = full_dataset.select(range(min(max_samples, len(full_dataset))))
        logger.info("Capped dataset to %d samples.", max_samples)

    logger.info("Total samples before split: %d", len(full_dataset))

    # Shuffle then split
    split = full_dataset.train_test_split(test_size=1 - train_split, seed=seed)

    # Apply formatting
    logger.info("Formatting dataset into instruction-tuning prompts…")
    formatted = DatasetDict(
        {
            "train": split["train"].map(format_example, remove_columns=split["train"].column_names),
            "test": split["test"].map(format_example, remove_columns=split["test"].column_names),
        }
    )

    logger.info(
        "Dataset ready — train: %d samples | test: %d samples",
        len(formatted["train"]),
        len(formatted["test"]),
    )
    return formatted


def preview(dataset: DatasetDict, n: int = 2) -> None:
    """Print a few formatted examples to stdout for quick sanity-checking."""
    print("\n" + "=" * 80)
    print("DATASET PREVIEW")
    print("=" * 80)
    for i, example in enumerate(dataset["train"].select(range(n))):
        print(f"\n--- Example {i + 1} ---")
        print(example["text"])
    print("=" * 80 + "\n")


if __name__ == "__main__":
    ds = load_and_format(max_samples=1000)
    preview(ds, n=2)
