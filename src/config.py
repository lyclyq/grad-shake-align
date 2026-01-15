from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


def _deep_update(d: Dict[str, Any], u: Dict[str, Any]) -> Dict[str, Any]:
    for k, v in u.items():
        if isinstance(v, dict) and isinstance(d.get(k), dict):
            _deep_update(d[k], v)
        else:
            d[k] = v
    return d


def _set_by_dotted_key(cfg: Dict[str, Any], dotted: str, value: Any) -> None:
    parts = dotted.split(".")
    cur = cfg
    for p in parts[:-1]:
        if p not in cur or not isinstance(cur[p], dict):
            cur[p] = {}
        cur = cur[p]
    cur[parts[-1]] = value


def _parse_value(s: str) -> Any:
    ss = s.strip()
    # bool
    if ss.lower() in {"true", "false"}:
        return ss.lower() == "true"
    # int
    try:
        if ss.startswith("0") and ss != "0" and not ss.startswith("0."):
            # keep as string for things like 02? (rare)
            pass
        else:
            i = int(ss)
            return i
    except Exception:
        pass
    # float
    try:
        f = float(ss)
        return f
    except Exception:
        pass
    # fallback string
    return ss


def load_yaml(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def resolve_config(
    base_path: str,
    schedule_path: Optional[str] = None,
    set_args: Optional[List[str]] = None,
) -> Dict[str, Any]:
    cfg = load_yaml(base_path)

    # schedule overlay
    if schedule_path:
        sched = load_yaml(schedule_path)
        _deep_update(cfg, sched)

    # CLI overrides: --set a.b.c=value
    if set_args:
        for item in set_args:
            if "=" not in item:
                continue
            k, v = item.split("=", 1)
            _set_by_dotted_key(cfg, k.strip(), _parse_value(v))

    validate_config(cfg)
    return cfg


def validate_config(cfg: Dict[str, Any]) -> None:
    # minimal validations (keep research code not brittle)
    method = cfg.get("method", {}).get("name")
    if method not in {"baseline", "ours"}:
        raise ValueError(f"method.name must be 'baseline' or 'ours', got: {method}")

    lora = cfg.get("method", {}).get("lora", {}) or {}
    r = int(lora.get("r", 0))
    R = int(lora.get("R", 0))

    # ✅ Baseline uses single-rank LoRA: allow R==r (or even missing R)
    if method == "baseline":
        if r <= 0:
            raise ValueError(f"Baseline LoRA rank must satisfy r>0. Got r={r}")
    else:
        # ✅ Ours is dual-rank LoRA
        if r <= 0 or R <= 0 or R <= r:
            raise ValueError(f"LoRA ranks must satisfy 0 < r < R. Got r={r}, R={R}")

    compute = cfg.get("compute", {}) or {}
    if int(compute.get("gpus_per_trial", 1)) <= 0:
        raise ValueError("compute.gpus_per_trial must be >= 1")
