from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Tuple

import yaml


def _stable_hash(d: Dict[str, Any]) -> str:
    s = json.dumps(d, sort_keys=True, separators=(',', ':'))
    return hashlib.sha1(s.encode('utf-8')).hexdigest()[:10]


def make_run_name(cfg: Dict[str, Any], extra: Dict[str, Any] | None = None) -> str:
    """Deterministic, inspection-friendly run naming.

    Format:
      {stage}__{task}__{model}__{method}__ep{epochs}__bs{batch}__lr{lr}__wu{warmup}__gpu{world}
      __date{YYYYMMDD-HHMMSS}__hash{cfg_hash8}
    """
    stage = cfg.get('stage', 'run')
    task = (cfg.get('task', {}).get('name', 'task') or 'task').replace('/', '_')
    model = (cfg.get('model', {}).get('name', 'model') or 'model').replace('/', '_')
    epochs = cfg.get('train', {}).get('epochs', '?')
    batch = cfg.get('train', {}).get('batch_size', '?')
    method = cfg.get('method', {}).get('name', 'baseline')
    lr = cfg.get('train', {}).get('lr')
    warmup = cfg.get('train', {}).get('warmup_ratio')
    world = cfg.get('compute', {}).get('num_gpus', 1)
    lora = cfg.get('method', {}).get('lora', {})
    r = lora.get('r', 'r')
    R = lora.get('R', 'R')

    key = {
        'lr': lr,
        'warmup_ratio': warmup,
        'weight_decay': cfg.get('train', {}).get('weight_decay'),
        'method': cfg.get('method', {}),
        'ranks': {'r': r, 'R': R},
        'extra': extra or {},
    }
    h = _stable_hash(key)
    ts = datetime.now().strftime('%Y%m%d-%H%M%S')
    return (
        f"{stage}__{task}__{model}__{method}__ep{epochs}__bs{batch}"
        f"__lr{lr}__wu{warmup}__gpu{world}__date{ts}__hash{h}"
    )


def prepare_run_dir(root: str | Path, run_name: str, overwrite: str) -> Tuple[Path, bool]:
    """Create (or reuse) run dir.

    Returns (path, resumed).
    overwrite: 'force' | 'resume' | 'ask'
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    run_dir = root / run_name

    if run_dir.exists():
        if overwrite == 'force':
            for p in run_dir.glob('**/*'):
                if p.is_file():
                    p.unlink()
            return run_dir, False
        if overwrite == 'resume':
            return run_dir, True
        raise RuntimeError(
            f"Run dir exists and overwrite='ask': {run_dir}. "
            "Use --set io.overwrite=force or --set io.overwrite=resume."
        )

    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir, False


def ensure_run_dir(
    cfg: Dict[str, Any],
    extra: Dict[str, Any] | None = None,
    run_id: str | None = None,
    kind: str | None = None,
) -> Tuple[Path, str, bool]:
    root = Path(cfg.get('io', {}).get('root', 'runs'))
    if kind:
        root = root / kind
    run_name = run_id or make_run_name(cfg, extra=extra)
    overwrite = cfg.get('io', {}).get('overwrite', 'ask')
    run_dir, resumed = prepare_run_dir(root, run_name, overwrite)
    return run_dir, run_name, resumed


def dump_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding='utf-8')


def dump_yaml(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(obj, sort_keys=False), encoding='utf-8')
