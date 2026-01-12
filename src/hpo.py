from __future__ import annotations

import csv
import json
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

from .artifacts import dump_json, dump_yaml, ensure_run_dir


def _logspace(min_exp: int, max_exp: int, points: int) -> List[float]:
    return np.logspace(min_exp, max_exp, num=points).tolist()


def _linearspace(min_v: float, max_v: float, points: int) -> List[float]:
    return np.linspace(min_v, max_v, num=points).tolist()


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


def narrow_lr_grid_around(best_lr: float, full_lrs: List[float], radius: int) -> List[float]:
    idx = int(np.argmin(np.abs(np.array(full_lrs) - best_lr)))
    lo = max(0, idx - radius)
    hi = min(len(full_lrs), idx + radius + 1)
    return full_lrs[lo:hi]


def narrow_warmup_grid_around(best_warmup: float, full_warmups: List[float], radius: int) -> List[float]:
    idx = int(np.argmin(np.abs(np.array(full_warmups) - best_warmup)))
    lo = max(0, idx - radius)
    hi = min(len(full_warmups), idx + radius + 1)
    return full_warmups[lo:hi]


def _auto_candidates(default: float, power: int) -> List[float]:
    if power <= 1:
        return [default]
    scales = np.geomspace(0.5, 2.0, num=power)
    return [float(default * s) for s in scales]


def generate_custom_grid(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Build cartesian grid for custom ours params in hpo.custom."""
    hpo = cfg.get('hpo', {})
    custom = hpo.get('custom', {})
    if not custom:
        return [{}]

    power = int(hpo.get('custom_grid_power', 2))
    keys = list(custom.keys())
    values: List[List[Any]] = []
    for k in keys:
        v = custom[k]
        if v == 'auto':
            default = float(cfg['method']['ours'].get(k, 0.0))
            values.append(_auto_candidates(default, power))
        elif isinstance(v, dict):
            if v.get('space') == 'log':
                values.append(_logspace(int(v['min_exp']), int(v['max_exp']), int(v['points'])))
            elif v.get('space') == 'linear':
                values.append(_linearspace(float(v['min']), float(v['max']), int(v['points'])))
            else:
                values.append(list(v.get('candidates', [])))
        else:
            values.append(list(v))

    trials: List[Dict[str, Any]] = []
    for combo in np.array(np.meshgrid(*values)).T.reshape(-1, len(keys)):
        d: Dict[str, Any] = {'method': {'ours': {}}}
        for k, v in zip(keys, combo):
            d['method']['ours'][k] = float(v) if isinstance(v, (np.floating, float)) else v
        trials.append(d)
    return trials


def _flatten_overrides(prefix: str, d: Dict[str, Any], out: List[str]) -> None:
    for k, v in d.items():
        p = f'{prefix}.{k}' if prefix else k
        if isinstance(v, dict):
            _flatten_overrides(p, v, out)
        else:
            out.append(f'{p}={v}')


def _select_best(trials: List[Dict[str, Any]], metric_key: str) -> Tuple[int, float, Dict[str, Any]]:
    best = (-1, -1e9, {})
    for t in trials:
        score = float(t.get(metric_key, -1e9))
        if score > best[1]:
            best = (int(t['trial_id']), score, t.get('trial_cfg_json', {}))
    return best


@dataclass
class TrialSpec:
    trial_id: int
    tag: str
    override: Dict[str, Any]
    stage: str


class TrialScheduler:
    def __init__(self, gpu_ids: List[int], gpus_per_trial: int):
        self.slots = [gpu_ids[i:i + gpus_per_trial] for i in range(0, len(gpu_ids), gpus_per_trial)]
        if not self.slots:
            self.slots = [[]]
        self.running: Dict[int, subprocess.Popen] = {}

    def run(self, cmd: List[str], env: Dict[str, str], slot_idx: int) -> subprocess.Popen:
        proc = subprocess.Popen(cmd, env=env)
        self.running[slot_idx] = proc
        return proc

    def poll_done(self) -> List[int]:
        done = []
        for slot, proc in list(self.running.items()):
            if proc.poll() is not None:
                done.append(slot)
                self.running.pop(slot, None)
        return done


def run_hpo(cfg: Dict[str, Any], base_config_path: str, schedule_path: Optional[str], set_args: List[str]) -> None:
    run_dir, run_name, resumed = ensure_run_dir(cfg, kind='hpo')
    dump_yaml(run_dir / 'config_resolved.yaml', cfg)
    state_path = run_dir / 'state.json'
    trials_csv = run_dir / 'trials.csv'
    trials_root = run_dir / 'trials'
    trials_root.mkdir(parents=True, exist_ok=True)

    done_ids = set()
    if resumed and state_path.exists():
        try:
            done_ids = set(json.loads(state_path.read_text()).get('done_ids', []))
        except Exception:
            done_ids = set()

    if not trials_csv.exists() or cfg['io'].get('overwrite') == 'force':
        with trials_csv.open('w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['trial_id', 'trial_tag', 'stage', 'val/best', 'trial_cfg_json'])
            writer.writeheader()

    lr_trials = generate_lr_warmup_trials(cfg)
    trial_specs: List[TrialSpec] = []
    baseline_r = int(cfg['method']['lora']['r'])
    baseline_R = int(cfg['method']['lora']['R'])
    multipliers = cfg['hpo'].get('hierarchical', {}).get('baseline_rank_multipliers', [1])
    multipliers = [int(m) for m in multipliers]
    trial_id = 0
    for m in multipliers:
        rank_r = baseline_r * m
        rank_R = baseline_R * m
        for t in lr_trials:
            override = {
                'method': {'name': 'baseline', 'lora': {'r': rank_r, 'R': rank_R}},
                **t,
            }
            tag = f'baseline_rmul{m}_lrwu_{trial_id}'
            trial_specs.append(TrialSpec(trial_id=trial_id, tag=tag, override=override, stage='baseline'))
            trial_id += 1

    baseline_trials = len(trial_specs)
    full_lrs = _logspace(int(cfg['hpo']['lr_grid']['min_exp']), int(cfg['hpo']['lr_grid']['max_exp']), int(cfg['hpo']['lr_grid']['points']))

    best_baseline_lr = float(cfg['train']['lr'])
    best_baseline_warmup = float(cfg['train']['warmup_ratio'])
    best_rank_multiplier = max(multipliers) if multipliers else 1
    best_rank_r = baseline_r * best_rank_multiplier
    rows = _read_trials(trials_csv)
    best_id, _, best_cfg = _select_best(
        [
            r for r in rows
            if r['stage'] == 'baseline'
            and int(r.get('trial_cfg_json', {}).get('method', {}).get('lora', {}).get('r', -1)) == best_rank_r
        ],
        metric_key='val/best',
    )
    if best_id >= 0:
        best_baseline_lr = float(best_cfg.get('train', {}).get('lr', best_baseline_lr))
        best_baseline_warmup = float(best_cfg.get('train', {}).get('warmup_ratio', best_baseline_warmup))

    hierarchical = cfg['hpo'].get('hierarchical', {})
    lr_radius = int(hierarchical.get('baseline_lr_neighborhood', 5))
    warmup_radius = int(hierarchical.get('baseline_warmup_neighborhood', 2))
    lrs2 = narrow_lr_grid_around(best_baseline_lr, full_lrs, radius=lr_radius)
    full_warmups = list(cfg['hpo'].get('warmup_grid', [best_baseline_warmup]))
    warmups2 = narrow_warmup_grid_around(best_baseline_warmup, full_warmups, radius=warmup_radius)
    custom_trials = generate_custom_grid(cfg)

    j = 0
    for lr in lrs2:
        for wu in warmups2:
            for c in custom_trials:
                override = {'method': {'name': 'ours'}, 'train': {'lr': float(lr), 'warmup_ratio': float(wu)}, **c}
                trial_specs.append(TrialSpec(trial_id=baseline_trials + j, tag=f'ours_{j}', override=override, stage='ours'))
                j += 1

    num_gpus = cfg.get('compute', {}).get('num_gpus', 1)
    if num_gpus == 'auto':
        num_gpus = 1
    num_gpus = int(num_gpus)
    gpus_per_trial = int(cfg.get('compute', {}).get('gpus_per_trial', 1))
    gpu_ids = list(range(max(num_gpus, 1)))
    scheduler = TrialScheduler(gpu_ids=gpu_ids, gpus_per_trial=gpus_per_trial)

    queue = [t for t in trial_specs if t.trial_id not in done_ids]
    active: Dict[int, TrialSpec] = {}
    slot_cycle = list(range(len(scheduler.slots)))

    while queue or scheduler.running:
        while queue and len(scheduler.running) < len(scheduler.slots):
            spec = queue.pop(0)
            slot_idx = slot_cycle[len(active) % len(scheduler.slots)]
            gpus = scheduler.slots[slot_idx]
            env = dict(os.environ)
            env.update(cfg.get('env', {}))
            if gpus:
                env['CUDA_VISIBLE_DEVICES'] = ','.join(str(g) for g in gpus)

            run_id = f"trial_{spec.trial_id:04d}__{spec.tag}"
            trial_sets: List[str] = []
            _flatten_overrides('', spec.override, trial_sets)
            trial_sets.append(f'io.root={trials_root.as_posix()}')
            cmd = ['python', 'scripts/run.py', 'train', '--config', base_config_path, '--run-id', run_id]
            if schedule_path:
                cmd += ['--schedule', schedule_path]
            for s in set_args + trial_sets:
                cmd += ['--set', s]
            cmd += ['--trial-id', str(spec.trial_id), '--trial-tag', spec.tag]

            scheduler.run(cmd, env=env, slot_idx=slot_idx)
            active[slot_idx] = spec

        done_slots = scheduler.poll_done()
        for slot in done_slots:
            spec = active.pop(slot)
            done_ids.add(spec.trial_id)
            trial_dir = trials_root / f"trial_{spec.trial_id:04d}__{spec.tag}"
            summary_path = trial_dir / 'summary.json'
            best_val = -1.0
            if summary_path.exists():
                summary = json.loads(summary_path.read_text())
                best_val = float(summary.get('aggregate', {}).get('best_val_acc_mean', -1.0))
            with trials_csv.open('a', encoding='utf-8', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=['trial_id', 'trial_tag', 'stage', 'val/best', 'trial_cfg_json'])
                writer.writerow({
                    'trial_id': spec.trial_id,
                    'trial_tag': spec.tag,
                    'stage': spec.stage,
                    'val/best': best_val,
                    'trial_cfg_json': json.dumps(spec.override),
                })
            state_path.write_text(json.dumps({'done_ids': sorted(done_ids)}, indent=2), encoding='utf-8')

        time.sleep(0.2)

    best_baseline = _select_best(_read_trials(trials_csv, stage='baseline'), metric_key='val/best')
    best_ours = _select_best(_read_trials(trials_csv, stage='ours'), metric_key='val/best')
    dump_json(run_dir / 'best_hparams.json', {
        'baseline': {'trial_id': best_baseline[0], 'val/best': best_baseline[1], 'cfg': best_baseline[2]},
        'ours': {'trial_id': best_ours[0], 'val/best': best_ours[1], 'cfg': best_ours[2]},
    })


def _read_trials(trials_csv: Path, stage: Optional[str] = None) -> List[Dict[str, Any]]:
    rows = []
    if not trials_csv.exists():
        return rows
    with trials_csv.open('r', encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                parsed = {
                    'trial_id': int(row['trial_id']),
                    'trial_tag': row['trial_tag'],
                    'stage': row['stage'],
                    'val/best': float(row['val/best']),
                    'trial_cfg_json': json.loads(row['trial_cfg_json']),
                }
            except Exception:
                continue
            if stage is None or parsed['stage'] == stage:
                rows.append(parsed)
    return rows
