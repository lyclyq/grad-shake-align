from __future__ import annotations

from typing import Any, Dict

import torch
from transformers import AutoConfig, AutoModelForSequenceClassification


def build_model(model_name: str, num_labels: int) -> torch.nn.Module:
    cfg = AutoConfig.from_pretrained(model_name, num_labels=num_labels)
    return AutoModelForSequenceClassification.from_pretrained(model_name, config=cfg)
