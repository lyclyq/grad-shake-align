from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import torch
from torch.utils.data import DataLoader

from .artifacts import ensure_run_dir
from .data_glue import load_glue
from .lora_layers import apply_lora
from .loggingx import ExperimentLogger
from .models_hf import build_model
from .trainer import train_eval_loop
from .utils import get_device, set_seed


def run_train(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Single run (one seed). Returns summary dict."""
    device = get_device()
    seed = int(cfg['train']['seed'])
    set_seed(seed)

    # Data
    dataset = cfg['task']['name']
    assert dataset.startswith('glue/'), "Only glue/<task> supported in this scaffold"
    glue_task = dataset.split('/', 1)[1]
    data = load_glue(glue_task, cfg['model']['name'], int(cfg['task']['max_len']))

    train_loader = DataLoader(data.train, batch_size=int(cfg['train']['batch_size']), shuffle=True, collate_fn=data.collator)
    val_loader = DataLoader(data.validation, batch_size=int(cfg['train']['batch_size']), shuffle=False, collate_fn=data.collator)

    # Model
    model = build_model(cfg['model']['name'], num_labels=data.num_labels)
    model = apply_lora(model, cfg)
    model.to(device)

    run_dir, run_name = ensure_run_dir(cfg, extra={'seed': seed})
    logger = ExperimentLogger(run_dir, enable_swanlab=bool(cfg['log']['swanlab']['enabled']), swanlab_project=str(cfg['log']['swanlab']['project']))

    # Persist config
    (Path(run_dir) / 'config_resolved.json').write_text(json.dumps(cfg, indent=2, sort_keys=True), encoding='utf-8')

    summary = train_eval_loop(cfg, model, train_loader, val_loader, logger, glue_task)

    # Save summary
    (Path(run_dir) / 'summary.json').write_text(json.dumps(summary, indent=2, sort_keys=True), encoding='utf-8')
    logger.close()
    return summary
