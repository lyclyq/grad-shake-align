from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Tuple


def _stable_hash(d: Dict[str, Any]) -> str:
    s = json.dumps(d, sort_keys=True, separators=(',', ':'))
    return hashlib.sha1(s.encode('utf-8')).hexdigest()[:10]


def make_run_name(cfg: Dict[str, Any], extra: Dict[str, Any] | None = None) -> str:
    """Deterministic, inspection-friendly run naming.

    Includes: stage, task, model, epochs, method, ranks, and a short hash of key hpo/method params.
    """
    stage = cfg.get('stage', 'run')
    task = (cfg.get('task', {}).get('name', 'task') or 'task').replace('/', '_')
    model = (cfg.get('model', {}).get('name', 'model') or 'model').replace('/', '_')
    epochs = cfg.get('train', {}).get('epochs', '?')
    method = cfg.get('method', {}).get('name', 'baseline')
    lora = cfg.get('method', {}).get('lora', {})
    r = lora.get('r', 'r')
    R = lora.get('R', 'R')

    key = {
        'lr': cfg.get('train', {}).get('lr'),
        'warmup_ratio': cfg.get('train', {}).get('warmup_ratio'),
        'wd': cfg.get('train', {}).get('weight_decay'),
        'method': cfg.get('method', {}),
        'extra': extra or {},
    }
    h = _stable_hash(key)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    return f"{ts}__{stage}__{task}__{model}__ep{epochs}__{method}__r{r}_R{R}__{h}"


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
            # destructive: wipe dir
            for p in run_dir.glob('**/*'):
                if p.is_file():
                    p.unlink()
            return run_dir, False
        if overwrite == 'resume':
            return run_dir, True
        # 'ask' should be handled by caller (interactive)
        return run_dir, True

    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir, False


def dump_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding='utf-8')
