from __future__ import annotations

from typing import Dict

import numpy as np


def accuracy(preds: np.ndarray, labels: np.ndarray) -> float:
    return float((preds == labels).mean())


def glue_metric(task: str, preds: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    task = task.lower()
    if task in {'sst2', 'qnli', 'rte', 'wnli'}:
        return {'accuracy': accuracy(preds, labels)}
    if task in {'mrpc', 'qqp'}:
        # Quick_and_dirty: report acc only; you can extend to F1.
        return {'accuracy': accuracy(preds, labels)}
    if task in {'mnli'}:
        return {'accuracy': accuracy(preds, labels)}
    # Fallback
    return {'accuracy': accuracy(preds, labels)}
