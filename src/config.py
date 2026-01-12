from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


def load_yaml(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    with p.open('r', encoding='utf-8') as f:
        return yaml.safe_load(f) or {}


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge `override` into `base` (mutates and returns base)."""
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            deep_merge(base[k], v)
        else:
            base[k] = v
    return base


def _parse_scalar(val: str) -> Any:
    s = val.strip()
    lo = s.lower()
    if lo in {'true', 'false'}:
        return lo == 'true'
    if lo in {'none', 'null'}:
        return None
    # int
    try:
        if s.startswith('0') and len(s) > 1 and s[1].isdigit():
            # avoid octal-like surprises
            raise ValueError
        return int(s)
    except ValueError:
        pass
    # float
    try:
        return float(s)
    except ValueError:
        return s


def apply_set_overrides(cfg: Dict[str, Any], set_args: Optional[List[str]]) -> Dict[str, Any]:
    """Apply CLI `--set a.b.c=value` overrides."""
    if not set_args:
        return cfg
    for kv in set_args:
        if '=' not in kv:
            continue
        key_path, raw_val = kv.split('=', 1)
        val = _parse_scalar(raw_val)
        keys = key_path.split('.')
        cur = cfg
        for k in keys[:-1]:
            if k not in cur or not isinstance(cur[k], dict):
                cur[k] = {}
            cur = cur[k]
        cur[keys[-1]] = val
    return cfg


def resolve_config(base_path: str | Path, schedule_path: Optional[str | Path] = None, set_args: Optional[List[str]] = None) -> Dict[str, Any]:
    cfg = load_yaml(base_path)
    if schedule_path:
        sch = load_yaml(schedule_path)
        cfg = deep_merge(cfg, sch)
    cfg = apply_set_overrides(cfg, set_args)
    _derive_defaults(cfg)
    validate_config(cfg)
    return cfg


def validate_config(cfg: Dict[str, Any]) -> None:
    # minimal validations (keep it lightweight; research code should not be brittle)
    method = cfg.get('method', {}).get('name')
    if method not in {'baseline', 'ours'}:
        raise ValueError(f"method.name must be 'baseline' or 'ours', got: {method}")

    lora = cfg.get('method', {}).get('lora', {})
    r, R = int(lora.get('r', 0)), int(lora.get('R', 0))
    if r <= 0 or R <= 0 or R <= r:
        raise ValueError(f"LoRA ranks must satisfy 0 < r < R. Got r={r}, R={R}")

    compute = cfg.get('compute', {})
    if int(compute.get('gpus_per_trial', 1)) <= 0:
        raise ValueError("compute.gpus_per_trial must be >= 1")


def _derive_defaults(cfg: Dict[str, Any]) -> None:
    cfg.setdefault('log', {})
    cfg['log'].setdefault('swanlab', {})
    cfg.setdefault('io', {})
    cfg.setdefault('train', {})
    cfg['train'].setdefault('seed', 42)

    stage = cfg.get('stage', '')
    seeds_cfg = cfg.get('seeds', {})
    if isinstance(seeds_cfg, dict) and stage in seeds_cfg:
        cfg['seed_list'] = list(seeds_cfg[stage])
    elif 'seed_list' not in cfg:
        seed = cfg['train'].get('seed', 42)
        cfg['seed_list'] = list(seed) if isinstance(seed, list) else [int(seed)]


def dump_json(obj: Any, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, sort_keys=True)
