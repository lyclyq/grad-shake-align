from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


def _load_metrics(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open('r', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def _group_rows(rows: List[Dict[str, str]]) -> Dict[Tuple[str, str], Dict[int, List[float]]]:
    grouped: Dict[Tuple[str, str], Dict[int, List[float]]] = {}
    for row in rows:
        split = row.get('split', '')
        metric = row.get('metric', '')
        step = int(float(row.get('step', 0)))
        value = float(row.get('value', 0.0))
        key = (split, metric)
        grouped.setdefault(key, {}).setdefault(step, []).append(value)
    return grouped


def _plot_curve(steps: List[int], mean: np.ndarray, band: np.ndarray, out_path: Path, title: str) -> None:
    plt.figure()
    plt.plot(steps, mean, label='mean')
    plt.fill_between(steps, mean - band, mean + band, alpha=0.2, label='±std')
    plt.title(title)
    plt.xlabel('step')
    plt.ylabel('value')
    plt.legend()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close()


def aggregate_runs(runs_dir: Path) -> Dict[str, Dict[str, Dict[int, List[float]]]]:
    run_metrics: Dict[str, Dict[Tuple[str, str], Dict[int, List[float]]]] = {}
    for run in runs_dir.glob('**/metrics.csv'):
        run_metrics[run.parent.as_posix()] = _group_rows(_load_metrics(run))
    return run_metrics


def plot_run(run_dir: Path) -> None:
    plots_dir = run_dir / 'plots'
    rows = []
    for seed_dir in run_dir.glob('seed_*'):
        rows.extend(_load_metrics(seed_dir / 'metrics.csv'))
    if not rows:
        return
    grouped = _group_rows(rows)
    for (split, metric), per_step in grouped.items():
        steps = sorted(per_step.keys())
        values = [per_step[s] for s in steps]
        mean = np.array([np.mean(v) for v in values])
        std = np.array([np.std(v) for v in values])
        title = f'{split}/{metric}'
        out_path = plots_dir / f'{split}_{metric}.png'
        _plot_curve(steps, mean, std, out_path, title)

    _plot_gap(run_dir, plots_dir, rows)
    _plot_convergence(run_dir, plots_dir, rows, threshold=0.9)


def _plot_gap(run_dir: Path, plots_dir: Path, rows: List[Dict[str, str]]) -> None:
    grouped = _group_rows(rows)
    if ('train', 'loss') not in grouped or ('val', 'accuracy') not in grouped:
        return
    train = grouped[('train', 'loss')]
    val = grouped[('val', 'accuracy')]
    steps = sorted(set(train.keys()) & set(val.keys()))
    if not steps:
        return
    gap = []
    for s in steps:
        gap.append(np.mean(val[s]) - np.mean(train[s]))
    plt.figure()
    plt.plot(steps, gap)
    plt.title('val-train gap')
    plt.xlabel('step')
    plt.ylabel('gap')
    plt.savefig(plots_dir / 'gap_val_train.png')
    plt.close()


def _plot_convergence(run_dir: Path, plots_dir: Path, rows: List[Dict[str, str]], threshold: float) -> None:
    grouped = _group_rows(rows)
    if ('val', 'accuracy') not in grouped:
        return
    val = grouped[('val', 'accuracy')]
    steps = sorted(val.keys())
    first_hit = None
    for s in steps:
        if np.mean(val[s]) >= threshold:
            first_hit = s
            break
    if first_hit is None:
        return
    plt.figure()
    plt.axvline(first_hit, linestyle='--', color='red')
    plt.title(f'convergence step @ {threshold}')
    plt.xlabel('step')
    plt.ylabel('val accuracy')
    plt.savefig(plots_dir / 'convergence.png')
    plt.close()


def run_plotting(runs_root: str) -> None:
    root = Path(runs_root)
    if not root.exists():
        return
    for run_dir in root.iterdir():
        if not run_dir.is_dir():
            continue
        plot_run(run_dir)
