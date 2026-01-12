from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from torch.utils.data import DataLoader

from .artifacts import dump_json, dump_yaml, ensure_run_dir
from .data_glue import load_glue
from .lora_layers import inject_lora
from .loggingx import ExperimentLogger
from .models_hf import build_model
from .trainer import train_one
from .utils import infer_device, set_seed


def _resolve_seed_list(cfg: Dict[str, Any]) -> List[int]:
    if 'seed_list' in cfg:
        return [int(s) for s in cfg['seed_list']]
    seed = cfg.get('train', {}).get('seed', 42)
    return [int(s) for s in seed] if isinstance(seed, list) else [int(seed)]


def run_train(cfg: Dict[str, Any], run_id: str | None = None, trial_id: int | None = None) -> Dict[str, Any]:
    device = infer_device()
    seeds = _resolve_seed_list(cfg)

    dataset = cfg['task']['name']
    assert dataset.startswith('glue/'), "Only glue/<task> supported in this scaffold"
    data = load_glue(dataset, cfg['model']['name'], int(cfg['task']['max_len']))

    run_dir, run_name, _ = ensure_run_dir(cfg, extra={'seeds': seeds}, run_id=run_id)
    dump_yaml(Path(run_dir) / 'config_resolved.yaml', cfg)

    summaries = []
    for seed in seeds:
        seed_cfg = copy.deepcopy(cfg)
        seed_cfg['train']['seed'] = int(seed)
        set_seed(int(seed))

        train_loader = DataLoader(
            data.train,
            batch_size=int(seed_cfg['train']['batch_size']),
            shuffle=True,
            collate_fn=data.collator,
        )
        val_loader = DataLoader(
            data.validation,
            batch_size=int(seed_cfg['train']['batch_size']),
            shuffle=False,
            collate_fn=data.collator,
        )
        test_loader = None
        if data.test is not None:
            test_loader = DataLoader(
                data.test,
                batch_size=int(seed_cfg['train']['batch_size']),
                shuffle=False,
                collate_fn=data.collator,
            )

        model = build_model(seed_cfg['model']['name'], num_labels=data.num_labels)
        inject_lora(
            model,
            mode=seed_cfg['method']['name'],
            r=int(seed_cfg['method']['lora']['r']),
            R=int(seed_cfg['method']['lora']['R']),
            alpha=float(seed_cfg['method']['lora']['alpha']),
            dropout=float(seed_cfg['method']['lora'].get('dropout', 0.0)),
        )
        model.to(device)

        seed_dir = Path(run_dir) / f'seed_{seed}'
        seed_dir.mkdir(parents=True, exist_ok=True)
        logger = ExperimentLogger.create(seed_dir, run_name, seed_cfg, seed=int(seed), trial_id=trial_id)

        summary = train_one(seed_cfg, model, train_loader, val_loader, test_loader, logger, dataset)
        summary['seed'] = int(seed)
        dump_json(seed_dir / 'summary.json', summary)
        logger.close()
        summaries.append(summary)

    best_vals = [s['best_val_acc'] for s in summaries]
    aggregate = {
        'best_val_acc_mean': float(sum(best_vals) / max(len(best_vals), 1)),
        'best_val_acc_std': float(float(np.std(best_vals)) if len(best_vals) > 1 else 0.0),
    }
    out = {'seeds': seeds, 'per_seed': summaries, 'aggregate': aggregate}
    aggregate_csv = Path(run_dir) / 'aggregate.csv'
    with aggregate_csv.open('w', encoding='utf-8') as f:
        f.write('seed,best_val_acc\n')
        for s in summaries:
            f.write(f"{s['seed']},{s['best_val_acc']}\n")
    dump_json(Path(run_dir) / 'summary.json', out)
    return out
