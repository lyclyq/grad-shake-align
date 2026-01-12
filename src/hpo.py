from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from .artifacts import ensure_run_dir
from .config import resolve_config


def _logspace(min_exp: int, max_exp: int, points: int) -> List[float]:
    # exponents are base-10
    return np.logspace(min_exp, max_exp, num=points).tolist()


def generate_lr_warmup_trials(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    hpo = cfg.get('hpo', {})
    lr_cfg = hpo.get('lr_grid', {})
    lrs = _logspace(int(lr_cfg.get('min_exp', -6)), int(lr_cfg.get('max_exp', -3)), int(lr_cfg.get('points', 10)))
    warmups = list(hpo.get('warmup_grid', [0.0, 0.06]))
    trials = []
    for lr in lrs:
        for wu in warmups:
            trials.append({'train': {'lr': float(lr), 'warmup_ratio': float(wu)}})
    return trials


def select_best_from_csv(csv_path: Path, metric_key: str = 'val/max') -> Tuple[int, float, Dict[str, Any]]:
    """Reads `trials.csv` and returns (trial_id, best_score, trial_cfg_json)."""
    import csv

    best = (-1, -1e9, {})
    if not csv_path.exists():
        return best

    with csv_path.open('r', encoding='utf-8') as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                score = float(row.get(metric_key, 'nan'))
                trial_id = int(row['trial_id'])
                cfg_json = json.loads(row.get('trial_cfg_json', '{}'))
            except Exception:
                continue
            if score > best[1]:
                best = (trial_id, score, cfg_json)
    return best


def narrow_lr_grid_around(best_lr: float, full_lrs: List[float], radius: int = 5) -> List[float]:
    # pick nearest index
    idx = int(np.argmin(np.abs(np.array(full_lrs) - best_lr)))
    lo = max(0, idx - radius)
    hi = min(len(full_lrs), idx + radius + 1)
    return full_lrs[lo:hi]


def generate_custom_grid(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Builds a cartesian grid for method.ours.* keys listed in hpo.custom.

    grid_power = 2 means each hyperparam gets 2 candidates, etc.
    """
    hpo = cfg.get('hpo', {})
    custom = hpo.get('custom', {})
    if not custom:
        return [{}]

    keys = list(custom.keys())
    vals = [custom[k] for k in keys]
    trials = []
    for combo in itertools.product(*vals):
        d: Dict[str, Any] = {'method': {'ours': {}}}
        for k, v in zip(keys, combo):
            d['method']['ours'][k] = v
        trials.append(d)
    return trials


def run_hpo(args) -> None:
    """Entry point for the `hpo` command.

    Writes:
      - trials.csv (one row per trial)
      - best_hparams.json
      - state.json for crash-safe resume

    The actual training execution is delegated to `scripts/run.py` (subprocess) in this scaffold.
    You can replace that with direct Python calls if you prefer.
    """
    import subprocess

    base = resolve_config(args.config, schedule_path=args.schedule, set_args=args.set)

    run_dir = ensure_run_dir(base, kind='hpo')
    state_path = run_dir / 'state.json'
    trials_csv = run_dir / 'trials.csv'

    # resume
    done = set()
    if state_path.exists() and base['io'].get('overwrite', 'ask') == 'resume':
        try:
            done = set(json.loads(state_path.read_text())['done'])
        except Exception:
            done = set()

    # Stage 1: baseline lr/warmup grid
    lr_trials = generate_lr_warmup_trials(base)

    # Use a deterministic trial list order
    trial_specs: List[Tuple[str, Dict[str, Any]]] = []
    for i, t in enumerate(lr_trials):
        trial_specs.append((f'baseline_lrwu_{i}', {'method': {'name': 'baseline'}, **t}))

    # Stage 2: ours custom grid around best baseline lr
    # We still create these specs now; the launcher will skip until baseline best exists.
    trial_specs.append(('__STAGE2_MARKER__', {}))

    # CSV header init
    if not trials_csv.exists() or base['io'].get('overwrite') == 'force':
        trials_csv.write_text('trial_id,trial_tag,val/max,seed,trial_cfg_json\n')

    for tid, (tag, override) in enumerate(trial_specs):
        if tag == '__STAGE2_MARKER__':
            break
        if tid in done:
            continue

        cmd = [
            'python', 'scripts/run.py', 'train',
            '--config', args.config,
        ]
        if args.schedule:
            cmd += ['--schedule', args.schedule]
        # apply base --set and per-trial overrides
        sets = (args.set or [])
        trial_sets: List[str] = []
        def _flatten(prefix: str, d: Dict[str, Any]):
            for k, v in d.items():
                p = f'{prefix}.{k}' if prefix else k
                if isinstance(v, dict):
                    _flatten(p, v)
                else:
                    trial_sets.append(f'{p}={v}')
        _flatten('', override)
        for s in sets + trial_sets:
            cmd += ['--set', s]
        cmd += ['--trial-tag', tag]
        print(' '.join(cmd))
        subprocess.run(cmd, check=False)

        done.add(tid)
        state_path.write_text(json.dumps({'done': sorted(done)}, indent=2))

    # Determine baseline best
    best_trial_id, best_score, best_cfg = select_best_from_csv(trials_csv)
    if best_trial_id < 0:
        print('[WARN] No baseline trials recorded; stage2 skipped.')
        return

    # Build stage2 trials
    full_lrs = _logspace(int(base['hpo']['lr_grid']['min_exp']), int(base['hpo']['lr_grid']['max_exp']), int(base['hpo']['lr_grid']['points']))
    best_lr = float(best_cfg.get('train', {}).get('lr', full_lrs[len(full_lrs)//2]))
    lrs2 = narrow_lr_grid_around(best_lr, full_lrs, radius=int(base['hpo'].get('radius', 5)))
    warmups2 = list(base['hpo'].get('warmup_grid', [0.0, 0.06]))

    # custom grid for ours
    custom_trials = generate_custom_grid(base)

    stage2_specs: List[Tuple[str, Dict[str, Any]]] = []
    j = 0
    for lr in lrs2:
        for wu in warmups2:
            for c in custom_trials:
                stage2_specs.append((f'ours_{j}', {'method': {'name': 'ours'}, 'train': {'lr': float(lr), 'warmup_ratio': float(wu)}, **c}))
                j += 1

    # Launch stage2 (continuing trial_id numbering)
    start_id = len(lr_trials)
    for k, (tag, override) in enumerate(stage2_specs):
        tid = start_id + k
        if tid in done:
            continue

        cmd = ['python', 'scripts/run.py', 'train', '--config', args.config]
        if args.schedule:
            cmd += ['--schedule', args.schedule]
        sets = (args.set or [])
        trial_sets: List[str] = []
        def _flatten(prefix: str, d: Dict[str, Any]):
            for kk, vv in d.items():
                p = f'{prefix}.{kk}' if prefix else kk
                if isinstance(vv, dict):
                    _flatten(p, vv)
                else:
                    trial_sets.append(f'{p}={vv}')
        _flatten('', override)
        for s in sets + trial_sets:
            cmd += ['--set', s]
        cmd += ['--trial-tag', tag]
        print(' '.join(cmd))
        subprocess.run(cmd, check=False)

        done.add(tid)
        state_path.write_text(json.dumps({'done': sorted(done)}, indent=2))

    # best of all
    best_trial_id, best_score, best_cfg = select_best_from_csv(trials_csv)
    (run_dir / 'best_hparams.json').write_text(json.dumps({'trial_id': best_trial_id, 'val/max': best_score, 'cfg': best_cfg}, indent=2))
