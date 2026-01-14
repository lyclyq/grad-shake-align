# src/data_glue.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from datasets import DatasetDict, load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding


@dataclass
class GlueData:
    train: Any
    validation: Any
    test: Any
    tokenizer: Any
    collator: Any
    num_labels: int


def _normalize_task_name(task_name: str) -> str:
    """
    Accepts either:
      - "glue/rte"
      - "rte"
    Returns:
      - "rte"
    """
    task_name = str(task_name)
    if "/" in task_name:
        return task_name.split("/", 1)[1]
    return task_name


def _sentence_keys(task: str) -> Tuple[str, Optional[str]]:
    return {
        "cola": ("sentence", None),
        "sst2": ("sentence", None),
        "mrpc": ("sentence1", "sentence2"),
        "qqp": ("question1", "question2"),
        "stsb": ("sentence1", "sentence2"),
        "mnli": ("premise", "hypothesis"),
        "qnli": ("question", "sentence"),
        "rte": ("sentence1", "sentence2"),
        "wnli": ("sentence1", "sentence2"),
    }[task]


def load_glue(task_name: str, model_name: str, max_len: int) -> GlueData:
    task = _normalize_task_name(task_name)
    ds: DatasetDict = load_dataset("glue", task)

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    s1_key, s2_key = _sentence_keys(task)

    def tok(ex: Dict[str, List[Any]]) -> Dict[str, Any]:
        if s2_key is None:
            return tokenizer(
                ex[s1_key],
                truncation=True,
                max_length=max_len,
            )
        return tokenizer(
            ex[s1_key],
            ex[s2_key],
            truncation=True,
            max_length=max_len,
        )

    # IMPORTANT:
    # Remove original text columns (sentence1/sentence2/idx etc.) so collator won't see strings.
    # Keep label -> later rename to labels.
    remove_cols = list(ds["train"].column_names)
    # keep label if present, we'll rename it later
    if "label" in remove_cols:
        remove_cols.remove("label")

    ds = ds.map(tok, batched=True, remove_columns=remove_cols)

    # Rename label -> labels (trainer expects batch["labels"])
    if "label" in ds["train"].column_names:
        ds = ds.rename_column("label", "labels")

    # Torch format: only tensor columns.
    keep_cols = [c for c in ds["train"].column_names if c in {"input_ids", "attention_mask", "token_type_ids", "labels"}]
    ds.set_format(type="torch", columns=keep_cols)

    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # num_labels
    # After rename, features still knows label classes via original schema
    # easiest robust: read from original dataset features before map, but we already mapped.
    # So we infer from raw glue schema quickly:
    raw = load_dataset("glue", task)
    num_labels = raw["train"].features["label"].num_classes

    return GlueData(
        train=ds["train"],
        validation=ds["validation"],
        test=ds.get("test", None),
        tokenizer=tokenizer,
        collator=collator,
        num_labels=num_labels,
    )
