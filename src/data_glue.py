from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding


@dataclass
class GlueData:
    train: Any
    validation: Any
    test: Any
    tokenizer: Any
    collator: Any
    num_labels: int


def load_glue(task_name: str, model_name: str, max_len: int) -> GlueData:
    # task_name like 'glue/rte'
    _, task = task_name.split('/', 1)
    ds = load_dataset('glue', task)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    sentence1_key, sentence2_key = {
        'cola': ('sentence', None),
        'sst2': ('sentence', None),
        'mrpc': ('sentence1', 'sentence2'),
        'qqp': ('question1', 'question2'),
        'stsb': ('sentence1', 'sentence2'),
        'mnli': ('premise', 'hypothesis'),
        'qnli': ('question', 'sentence'),
        'rte': ('sentence1', 'sentence2'),
        'wnli': ('sentence1', 'sentence2'),
    }[task]

    def tok(ex):
        if sentence2_key is None:
            return tokenizer(ex[sentence1_key], truncation=True, max_length=max_len)
        return tokenizer(ex[sentence1_key], ex[sentence2_key], truncation=True, max_length=max_len)

    ds = ds.map(tok, batched=True)
    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    num_labels = ds['train'].features['label'].num_classes

    return GlueData(
        train=ds['train'],
        validation=ds['validation'],
        test=ds.get('test', None),
        tokenizer=tokenizer,
        collator=collator,
        num_labels=num_labels,
    )
