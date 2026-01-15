#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
tests/run_rte_gate_auto.py

Full (auto) pipeline on GLUE/RTE with 7 curves in one plot:

A) Baseline LR selection (2 lines)
   - baseline_r32: LoRA rank = 32
   - baseline_R128: LoRA rank = 128
   - epochs = 2, seeds = {12, 24}
   - LR candidates fixed list: 2e-5,1e-4,2e-4,1e-3,2e-3
   - choose best LR by score = 0.5*max + 0.4*final + 0.1*avg_last2(val_acc)

B) OURS HPO Stage-1 (broad)
   - epochs = 1, seeds = {12, 24}
   - schemes (5):
        percentile:  P80_dir, P90_dir, P90_both    (3)
        cos-thr:     cos_dir, cos_both             (2)
   - OURS LR candidates:
        base_lrs = [best_baseline_r32_lr, best_baseline_R128_lr]
        lr_grid  = base_lr * multiplier, multiplier in --ours_lr_mults (default 0.5,1.0)
   - keep topK (default 15) per scheme by score.

C) OURS HPO Stage-2 (competition)
   - epochs = 3, seeds = {12, 24}
   - run those topK configs and pick the best (per scheme) by score.

D) Final runs (7 lines)
   - epochs = 5, seeds = {2, 3, 5}
   - run:
       baseline_r32(best_lr_r32)
       baseline_R128(best_lr_R128)
       ours_best_per_scheme (5 schemes)
   - output:
       tests_root/log/  (all run folders + summary CSVs)
       tests_root/plot/ (curves + bars)

Assumes your main entry supports:
  python scripts/run.py train --config configs/base.yaml --trial_tag XXX --set k=v ...

CSV parsing expects at least: step and val/acc columns.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _now_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _ensure_dirs(root: Path) -> Tuple[Path, Path]:
    log_dir = root / "log"
    plot_dir = root / "plot"
    log_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)
    return log_dir, plot_dir


def _run_cmd(cmd: List[str]) -> int:
    print("\n" + "=" * 120)
    print("CMD:", " ".join(cmd))
    print("=" * 120)
    p = subprocess.run(cmd)
    return p.returncode


def _collect_csvs(root: Path) -> List[Path]:
    if not root.exists():
        return []
    return sorted(root.rglob("*.csv"))


def _read_csv_best_effort(p: Path) -> Optional[pd.DataFrame]:
    try:
        return pd.read_csv(p)
    except Exception:
        return None


def _extract_val_curve(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    df = df.rename(columns={c: str(c).strip() for c in df.columns})

    step_col = None
    for cand in ["step", "global_step", "iter", "iteration"]:
        if cand in df.columns:
            step_col = cand
            break

    val_col = None
    for cand in ["val/acc", "val_acc", "validation/acc", "validation_acc", "eval/acc", "eval_acc"]:
        if cand in df.columns:
            val_col = cand
            break

    if step_col is None or val_col is None:
        return None

    out = df[[step_col, val_col]].copy()
    out = out.rename(columns={step_col: "step", val_col: "val_acc"})
    out = out.dropna(subset=["step", "val_acc"])
    out["step"] = pd.to_numeric(out["step"], errors="coerce")
    out["val_acc"] = pd.to_numeric(out["val_acc"], errors="coerce")
    out = out.dropna(subset=["step", "val_acc"])
    out = out.sort_values("step").drop_duplicates("step", keep="last")
    if out.empty:
        return None
    return out


def _score_from_curve(curve: pd.DataFrame) -> Tuple[float, float, float, float]:
    vals = curve["val_acc"].astype(float).to_list()
    max_v = float(max(vals))
    final_v = float(vals[-1])
    if len(vals) >= 2:
        avg_last2 = float((vals[-1] + vals[-2]) / 2.0)
    else:
        avg_last2 = final_v
    score = 0.5 * max_v + 0.4 * final_v + 0.1 * avg_last2
    return score, max_v, final_v, avg_last2


def _mean_scores(per_seed: Dict[int, Tuple[float, float, float, float]]) -> Tuple[float, float, float, float]:
    arr = list(per_seed.values())
    score = float(sum(a[0] for a in arr) / len(arr))
    max_v = float(sum(a[1] for a in arr) / len(arr))
    final_v = float(sum(a[2] for a in arr) / len(arr))
    avg_last2 = float(sum(a[3] for a in arr) / len(arr))
    return score, max_v, final_v, avg_last2


def _plot_curves(curves: Dict[str, pd.DataFrame], out_png: Path, title: str) -> None:
    plt.figure()
    for name, df in curves.items():
        plt.plot(df["step"], df["val_acc"], label=name)
    plt.xlabel("step")
    plt.ylabel("val_acc")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png)
    plt.close()


def _plot_bars(df: pd.DataFrame, out_png: Path, col: str, title: str) -> None:
    plt.figure()
    plt.bar(df["name"], df[col])
    plt.xticks(rotation=30, ha="right")
    plt.ylabel(col)
    plt.title(title)
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png)
    plt.close()


@dataclass(frozen=True)
class Scheme:
    name: str
    trigger_mode: str               # percentile | cos_threshold
    percentile_p: Optional[int]     # only for percentile
    cos_threshold: Optional[float]  # only for cos_threshold
    correction_mode: str            # direction_only | both


def build_schemes(cos_thr: float) -> List[Scheme]:
    schemes: List[Scheme] = []
    # percentile (按你现在的裁剪：both 只留 P90；dir 留 80/90)
    schemes += [
        Scheme("P80_dir",  "percentile",    80, None, "direction_only"),
        Scheme("P90_dir",  "percentile",    90, None, "direction_only"),
        Scheme("P90_both", "percentile",    90, None, "both"),
    ]
    # cos-threshold（dir + both）
    schemes += [
        Scheme(f"cos{cos_thr:g}_dir",  "cos_threshold", None, cos_thr, "direction_only"),
        Scheme(f"cos{cos_thr:g}_both", "cos_threshold", None, cos_thr, "both"),
    ]
    return schemes


def grid_hypers() -> Dict[str, List[Any]]:
    """
    3-5格点（按敏感度挑点，不均分）。
    你之后要继续精简：优先砍 gamma_hi / lambda_o 的格点数。
    """
    return {
        "method.ours.k":          [4.0, 8.0, 12.0],
        "method.ours.lambda_n":   [0.2, 0.4, 0.6],
        "method.ours.lambda_o":   [0.2, 0.4, 0.6],
        "method.ours.eta":        [0.1, 0.2, 0.3],
        "method.ours.gamma_r":    [0.2, 0.3, 0.4],
        "method.ours.gamma_hi":   [0.2, 0.3, 0.4],
        "method.ours.beta_max":   [0.3, 0.5],
        "method.ours.rho_max":    [0.3, 0.5],
    }


def iter_grid(grid: Dict[str, List[Any]]) -> Iterable[Dict[str, Any]]:
    keys = list(grid.keys())
    for values in product(*[grid[k] for k in keys]):
        yield {k: v for k, v in zip(keys, values)}


def run_one_seed(
    entry: str,
    mode: str,
    config: str,
    trial_tag: str,
    sets: Dict[str, Any],
    seed: int,
) -> Tuple[int, Optional[Path]]:
    io_root = Path(str(sets["io.root"])).resolve()
    before = set(_collect_csvs(io_root))

    cmd = [sys.executable, entry, mode, "--config", config, "--trial_tag", f"{trial_tag}_s{seed}"]
    for k, v in sets.items():
        cmd += ["--set", f"{k}={v}"]
    cmd += ["--set", f"train.seed={seed}"]

    rc = _run_cmd(cmd)
    if rc != 0:
        return rc, None

    after = set(_collect_csvs(io_root))
    new_csvs = sorted(list(after - before))
    if not new_csvs:
        all_csvs = _collect_csvs(io_root)
        if not all_csvs:
            return 0, None
        return 0, max(all_csvs, key=lambda p: p.stat().st_mtime)

    csv_path = max(new_csvs, key=lambda p: p.stat().st_mtime)
    return 0, csv_path


def run_multi_seed_and_score(
    entry: str,
    mode: str,
    config: str,
    trial_tag: str,
    sets: Dict[str, Any],
    seeds: List[int],
) -> Tuple[bool, Dict[int, Path], Dict[int, Tuple[float, float, float, float]]]:
    csv_paths: Dict[int, Path] = {}
    scores: Dict[int, Tuple[float, float, float, float]] = {}

    for sd in seeds:
        rc, csv_path = run_one_seed(entry, mode, config, trial_tag, sets, sd)
        if rc != 0 or csv_path is None:
            return False, csv_paths, scores

        df = _read_csv_best_effort(csv_path)
        if df is None:
            return False, csv_paths, scores

        curve = _extract_val_curve(df)
        if curve is None:
            return False, csv_paths, scores

        csv_paths[sd] = csv_path
        scores[sd] = _score_from_curve(curve)

    return True, csv_paths, scores


def baseline_lr_search(
    *,
    entry: str,
    mode: str,
    config: str,
    io_root: Path,
    task: str,
    warmup_ratio: float,
    batch_size: int,
    epochs: int,
    seeds: List[int],
    lrs: List[float],
    rank: int,
    tag_prefix: str,
) -> Tuple[float, pd.DataFrame]:
    rows: List[Dict[str, Any]] = []
    best_lr = lrs[0]
    best_score = -1e9

    for lr in lrs:
        sets = {
            "method.name": "baseline",
            "method.lora.r": str(rank),
            "task.name": task,
            "train.epochs": str(epochs),
            "train.batch_size": str(batch_size),
            "train.lr": str(lr),
            "train.warmup_ratio": str(warmup_ratio),
            "io.root": str(io_root.as_posix()),
            "log.csv": "true",
            "debug.enabled": "false",
        }
        ok, csv_paths, per_seed = run_multi_seed_and_score(
            entry=entry, mode=mode, config=config,
            trial_tag=f"{tag_prefix}_lr{lr:g}", sets=sets, seeds=seeds
        )
        if not ok:
            rows.append({"name": f"lr{lr:g}", "lr": lr, "status": "fail"})
            continue

        score, max_v, final_v, avg_last2 = _mean_scores(per_seed)
        rows.append({
            "name": f"lr{lr:g}",
            "lr": lr,
            "status": "ok",
            "score": score,
            "max_val": max_v,
            "final_val": final_v,
            "avg_last2": avg_last2,
            "csv_paths": json.dumps({str(k): str(v) for k, v in csv_paths.items()}),
        })

        if score > best_score:
            best_score = score
            best_lr = lr

    # df = pd.DataFrame(rows).sort_values(["status", "score"], ascending=[True, False], na_position="last")
    # return best_lr, df
    df = pd.DataFrame(rows)

    # ---- FIX: ensure columns exist ----
    if "score" not in df.columns:
        df["score"] = np.nan
    if "status" not in df.columns:
        df["status"] = "fail"

    # make status sortable: ok first, then fail
    status_order = {"ok": 0, "fail": 1}
    df["_status_rank"] = df["status"].map(status_order).fillna(9).astype(int)

    # score: higher is better; fail rows will be NaN -> fill with -inf for sorting
    df["_score_sort"] = pd.to_numeric(df["score"], errors="coerce").fillna(-1e18)

    df = df.sort_values(["_status_rank", "_score_sort"], ascending=[True, False]).drop(columns=["_status_rank", "_score_sort"])

    return best_lr, df


def ours_stage1_hpo(
    *,
    entry: str,
    mode: str,
    config: str,
    io_root: Path,
    task: str,
    warmup_ratio: float,
    batch_size: int,
    epochs: int,
    seeds: List[int],
    schemes: List[Scheme],
    base_lrs: List[float],
    lr_mults: List[float],
    topk: int,
    trigger_metric: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    grid = grid_hypers()
    rows: List[Dict[str, Any]] = []

    lr_candidates: List[float] = []
    for blr in base_lrs:
        for m in lr_mults:
            lr_candidates.append(float(blr) * float(m))
    lr_candidates = sorted(list(dict.fromkeys(lr_candidates)))

    for sch in schemes:
        for lr in lr_candidates:
            for hp in iter_grid(grid):
                sets: Dict[str, Any] = {
                    "method.name": "ours",
                    "task.name": task,
                    "train.epochs": str(epochs),
                    "train.batch_size": str(batch_size),
                    "train.lr": str(lr),
                    "train.warmup_ratio": str(warmup_ratio),
                    "io.root": str(io_root.as_posix()),
                    "log.csv": "true",
                    "debug.enabled": "false",
                }

                # trigger
                sets["method.ours.trigger.mode"] = sch.trigger_mode
                sets["method.ours.trigger.metric"] = str(trigger_metric)
                if sch.trigger_mode == "percentile":
                    sets["method.ours.trigger.percentile_p"] = str(int(sch.percentile_p or 90))
                else:
                    sets["method.ours.trigger.cos_threshold"] = str(float(sch.cos_threshold or 0.3))

                # correction (only 2 modes)
                sets["method.ours.correction.mode"] = sch.correction_mode
                # do NOT write direction_rescale; direction_only is pure direction fix now

                for k, v in hp.items():
                    sets[k] = str(v)

                tag = f"s1_{sch.name}_lr{lr:g}_" + "_".join([f"{k.split('.')[-1]}{v}" for k, v in hp.items()])
                ok, csv_paths, per_seed = run_multi_seed_and_score(
                    entry=entry, mode=mode, config=config,
                    trial_tag=tag, sets=sets, seeds=seeds
                )
                if not ok:
                    rows.append({
                        "scheme": sch.name,
                        "stage": "s1",
                        "lr": lr,
                        "status": "fail",
                        "sets_json": json.dumps(sets, ensure_ascii=False),
                    })
                    continue

                score, max_v, final_v, avg_last2 = _mean_scores(per_seed)
                rows.append({
                    "scheme": sch.name,
                    "stage": "s1",
                    "lr": lr,
                    "status": "ok",
                    "score": score,
                    "max_val": max_v,
                    "final_val": final_v,
                    "avg_last2": avg_last2,
                    "csv_paths": json.dumps({str(k): str(v) for k, v in csv_paths.items()}),
                    "sets_json": json.dumps(sets, ensure_ascii=False),
                })

    df = pd.DataFrame(rows)

    out = []
    for sch in schemes:
        sub = df[(df["scheme"] == sch.name) & (df["status"] == "ok")].copy()
        sub = sub.sort_values("score", ascending=False).head(topk)
        out.append(sub)
    top_df = pd.concat(out, axis=0, ignore_index=True) if out else pd.DataFrame()
    return df, top_df


def ours_stage2_compete(
    *,
    entry: str,
    mode: str,
    config: str,
    io_root: Path,
    epochs: int,
    seeds: List[int],
    top_df: pd.DataFrame,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for i, r in top_df.iterrows():
        sets = json.loads(r["sets_json"])
        sets["train.epochs"] = str(epochs)
        sets["io.root"] = str(io_root.as_posix())

        tag = f"s2_{r['scheme']}_{i}"
        ok, csv_paths, per_seed = run_multi_seed_and_score(
            entry=entry, mode=mode, config=config,
            trial_tag=tag, sets=sets, seeds=seeds
        )
        if not ok:
            rows.append({"scheme": r["scheme"], "stage": "s2", "status": "fail", "sets_json": json.dumps(sets)})
            continue

        score, max_v, final_v, avg_last2 = _mean_scores(per_seed)
        rows.append({
            "scheme": r["scheme"],
            "stage": "s2",
            "status": "ok",
            "score": score,
            "max_val": max_v,
            "final_val": final_v,
            "avg_last2": avg_last2,
            "csv_paths": json.dumps({str(k): str(v) for k, v in csv_paths.items()}),
            "sets_json": json.dumps(sets, ensure_ascii=False),
        })

    return pd.DataFrame(rows)


def _mean_curve_from_seed_csvs(csv_paths: Dict[int, Path]) -> Optional[pd.DataFrame]:
    per_seed_curves = []
    for sd, csvp in csv_paths.items():
        df = _read_csv_best_effort(csvp)
        if df is None:
            continue
        curve = _extract_val_curve(df)
        if curve is None:
            continue
        curve = curve.rename(columns={"val_acc": f"val_acc_s{sd}"})
        per_seed_curves.append(curve.set_index("step"))

    if not per_seed_curves:
        return None

    joined = per_seed_curves[0]
    for c in per_seed_curves[1:]:
        joined = joined.join(c, how="outer")
    joined = joined.sort_index()

    val_cols = [c for c in joined.columns if c.startswith("val_acc_s")]
    joined["val_acc_mean"] = joined[val_cols].mean(axis=1, skipna=True)
    mean_curve = joined[["val_acc_mean"]].reset_index().rename(columns={"val_acc_mean": "val_acc"})
    return mean_curve


def baseline_final_runs(
    *,
    entry: str,
    mode: str,
    config: str,
    io_root: Path,
    task: str,
    warmup_ratio: float,
    batch_size: int,
    epochs: int,
    seeds: List[int],
    rank: int,
    lr: float,
    name: str,
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    sets: Dict[str, Any] = {
        "method.name": "baseline",
        "method.lora.r": str(rank),
        "task.name": task,
        "train.epochs": str(epochs),
        "train.batch_size": str(batch_size),
        "train.lr": str(lr),
        "train.warmup_ratio": str(warmup_ratio),
        "io.root": str(io_root.as_posix()),
        "log.csv": "true",
        "debug.enabled": "false",
    }

    ok, csv_paths, per_seed = run_multi_seed_and_score(
        entry=entry, mode=mode, config=config,
        trial_tag=f"final_{name}_lr{lr:g}", sets=sets, seeds=seeds
    )
    if not ok:
        return pd.DataFrame([{"name": name, "status": "fail"}]), None

    score, max_v, final_v, avg_last2 = _mean_scores(per_seed)
    row = pd.DataFrame([{
        "name": name,
        "status": "ok",
        "score": score,
        "max_val": max_v,
        "final_val": final_v,
        "avg_last2": avg_last2,
        "csv_paths": json.dumps({str(k): str(v) for k, v in csv_paths.items()}),
        "sets_json": json.dumps(sets, ensure_ascii=False),
    }])

    curve = _mean_curve_from_seed_csvs(csv_paths)
    return row, curve


def ours_final_runs(
    *,
    entry: str,
    mode: str,
    config: str,
    io_root: Path,
    task: str,
    warmup_ratio: float,
    batch_size: int,
    epochs: int,
    seeds: List[int],
    best_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    curves: Dict[str, pd.DataFrame] = {}
    rows: List[Dict[str, Any]] = []

    for sch_name in sorted(best_df["scheme"].unique().tolist()):
        sub = best_df[(best_df["scheme"] == sch_name) & (best_df["status"] == "ok")].copy()
        if sub.empty:
            continue
        sub = sub.sort_values("score", ascending=False).head(1)
        sets = json.loads(sub.iloc[0]["sets_json"])

        sets["train.epochs"] = str(epochs)
        sets["task.name"] = task
        sets["train.batch_size"] = str(batch_size)
        sets["train.warmup_ratio"] = str(warmup_ratio)
        sets["io.root"] = str(io_root.as_posix())
        sets["log.csv"] = "true"
        sets["debug.enabled"] = "false"

        tag = f"final_{sch_name}"
        ok, csv_paths, per_seed = run_multi_seed_and_score(
            entry=entry, mode=mode, config=config,
            trial_tag=tag, sets=sets, seeds=seeds
        )
        if not ok:
            rows.append({"name": sch_name, "status": "fail"})
            continue

        curve = _mean_curve_from_seed_csvs(csv_paths)
        if curve is None:
            rows.append({"name": sch_name, "status": "fail_no_curve"})
            continue

        curves[sch_name] = curve
        score, max_v, final_v, avg_last2 = _mean_scores(per_seed)
        rows.append({
            "name": sch_name,
            "status": "ok",
            "score": score,
            "max_val": max_v,
            "final_val": final_v,
            "avg_last2": avg_last2,
            "csv_paths": json.dumps({str(k): str(v) for k, v in csv_paths.items()}),
            "sets_json": json.dumps(sets, ensure_ascii=False),
        })

    return pd.DataFrame(rows), curves


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/base.yaml")
    ap.add_argument("--entry", type=str, default="scripts/run.py")
    ap.add_argument("--mode", type=str, default="train")

    ap.add_argument("--root", type=str, default="tests_v2")
    ap.add_argument("--task", type=str, default="glue/rte")

    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--warmup_ratio", type=float, default=0.1)

    # Baseline LR search
    ap.add_argument("--baseline_epochs", type=int, default=2)
    ap.add_argument("--baseline_seeds", type=str, default="12,24")
    ap.add_argument("--baseline_lrs", type=str, default="2e-5,1e-4,2e-4,1e-3,2e-3")
    ap.add_argument("--rank_small", type=int, default=32)
    ap.add_argument("--rank_large", type=int, default=128)

    # Ours HPO
    ap.add_argument("--s1_epochs", type=int, default=1)
    ap.add_argument("--s2_epochs", type=int, default=3)
    ap.add_argument("--final_epochs", type=int, default=5)

    ap.add_argument("--hpo_seeds", type=str, default="12,24")
    ap.add_argument("--final_seeds", type=str, default="2,3,5")

    ap.add_argument("--cos_threshold", type=float, default=0.30)
    ap.add_argument("--trigger_metric", type=str, default="C_R")  # C_r | C_R | A_b | misalign

    # ours LR = baseline_best_lr * mult
    ap.add_argument("--ours_lr_mults", type=str, default="0.5,1.0")
    ap.add_argument("--topk", type=int, default=15)

    args = ap.parse_args()

    tests_root = Path(args.root).resolve()
    log_dir, plot_dir = _ensure_dirs(tests_root)

    sweep_tag = _now_tag()
    io_root = (log_dir / f"runs_{sweep_tag}").resolve()
    io_root.mkdir(parents=True, exist_ok=True)

    baseline_seeds = [int(x) for x in args.baseline_seeds.split(",") if x.strip()]
    hpo_seeds = [int(x) for x in args.hpo_seeds.split(",") if x.strip()]
    final_seeds = [int(x) for x in args.final_seeds.split(",") if x.strip()]

    baseline_lrs = [float(x) for x in args.baseline_lrs.split(",") if x.strip()]
    ours_lr_mults = [float(x) for x in args.ours_lr_mults.split(",") if x.strip()]

    # -------------------------
    # Stage A: Baseline LR search
    # -------------------------
    print("\n[Stage A] Baseline LR search (ep=2, seeds=12/24)")
    best_lr_r, df_r = baseline_lr_search(
        entry=args.entry, mode=args.mode, config=args.config,
        io_root=io_root, task=args.task, warmup_ratio=args.warmup_ratio, batch_size=args.batch_size,
        epochs=args.baseline_epochs, seeds=baseline_seeds, lrs=baseline_lrs,
        rank=args.rank_small, tag_prefix=f"baseline_r{args.rank_small}",
    )
    best_lr_R, df_R = baseline_lr_search(
        entry=args.entry, mode=args.mode, config=args.config,
        io_root=io_root, task=args.task, warmup_ratio=args.warmup_ratio, batch_size=args.batch_size,
        epochs=args.baseline_epochs, seeds=baseline_seeds, lrs=baseline_lrs,
        rank=args.rank_large, tag_prefix=f"baseline_R{args.rank_large}",
    )

    baseline_summary = pd.concat(
        [df_r.assign(line=f"r{args.rank_small}"), df_R.assign(line=f"R{args.rank_large}")],
        ignore_index=True,
    )
    baseline_csv = log_dir / f"baseline_lr_{sweep_tag}.csv"
    baseline_summary.to_csv(baseline_csv, index=False)
    print(f"[OK] baseline summary -> {baseline_csv}")
    print(f"[OK] best baseline lr: r{args.rank_small}={best_lr_r:g}, R{args.rank_large}={best_lr_R:g}")

    # -------------------------
    # Stage B: OURS HPO stage-1
    # -------------------------
    print("\n[Stage B] OURS HPO stage-1 (ep=1 broad, seeds=12/24)")
    schemes = build_schemes(args.cos_threshold)
    base_lrs = [best_lr_r, best_lr_R]

    all_s1, top_s1 = ours_stage1_hpo(
        entry=args.entry, mode=args.mode, config=args.config,
        io_root=io_root, task=args.task, warmup_ratio=args.warmup_ratio, batch_size=args.batch_size,
        epochs=args.s1_epochs, seeds=hpo_seeds, schemes=schemes,
        base_lrs=base_lrs, lr_mults=ours_lr_mults, topk=args.topk,
        trigger_metric=args.trigger_metric,
    )

    s1_all_csv = log_dir / f"ours_s1_all_{sweep_tag}.csv"
    s1_top_csv = log_dir / f"ours_s1_top{args.topk}_{sweep_tag}.csv"
    all_s1.to_csv(s1_all_csv, index=False)
    top_s1.to_csv(s1_top_csv, index=False)
    print(f"[OK] ours s1 all -> {s1_all_csv}")
    print(f"[OK] ours s1 top -> {s1_top_csv}")

    # -------------------------
    # Stage C: OURS HPO stage-2
    # -------------------------
    print("\n[Stage C] OURS HPO stage-2 (ep=3 topK compete, seeds=12/24)")
    s2_df = ours_stage2_compete(
        entry=args.entry, mode=args.mode, config=args.config,
        io_root=io_root, epochs=args.s2_epochs, seeds=hpo_seeds, top_df=top_s1,
    )

    s2_csv = log_dir / f"ours_s2_{sweep_tag}.csv"
    s2_df.to_csv(s2_csv, index=False)
    print(f"[OK] ours s2 -> {s2_csv}")

    best_rows = []
    for sch in schemes:
        sub = s2_df[(s2_df["scheme"] == sch.name) & (s2_df["status"] == "ok")].copy()
        if sub.empty:
            continue
        best_rows.append(sub.sort_values("score", ascending=False).head(1))
    best_df = pd.concat(best_rows, ignore_index=True) if best_rows else pd.DataFrame()
    best_csv = log_dir / f"ours_best_{sweep_tag}.csv"
    best_df.to_csv(best_csv, index=False)
    print(f"[OK] ours best per scheme -> {best_csv}")

    # -------------------------
    # Stage D: Final runs (ep=5)
    # -------------------------
    print("\n[Stage D] Final runs (ep=5, seeds=2/3/5) — 7 curves total")
    curves: Dict[str, pd.DataFrame] = {}
    final_rows: List[pd.DataFrame] = []

    # D1) baseline final curves
    br_row, br_curve = baseline_final_runs(
        entry=args.entry, mode=args.mode, config=args.config,
        io_root=io_root, task=args.task, warmup_ratio=args.warmup_ratio, batch_size=args.batch_size,
        epochs=args.final_epochs, seeds=final_seeds,
        rank=args.rank_small, lr=best_lr_r, name=f"baseline_r{args.rank_small}",
    )
    final_rows.append(br_row)
    if br_curve is not None:
        curves[f"baseline_r{args.rank_small}"] = br_curve

    bR_row, bR_curve = baseline_final_runs(
        entry=args.entry, mode=args.mode, config=args.config,
        io_root=io_root, task=args.task, warmup_ratio=args.warmup_ratio, batch_size=args.batch_size,
        epochs=args.final_epochs, seeds=final_seeds,
        rank=args.rank_large, lr=best_lr_R, name=f"baseline_R{args.rank_large}",
    )
    final_rows.append(bR_row)
    if bR_curve is not None:
        curves[f"baseline_R{args.rank_large}"] = bR_curve

    # D2) ours final curves (5 schemes)
    ours_final_df, ours_curves = ours_final_runs(
        entry=args.entry, mode=args.mode, config=args.config,
        io_root=io_root, task=args.task, warmup_ratio=args.warmup_ratio, batch_size=args.batch_size,
        epochs=args.final_epochs, seeds=final_seeds, best_df=best_df,
    )
    final_rows.append(ours_final_df)
    curves.update(ours_curves)

    final_df = pd.concat(final_rows, ignore_index=True)
    final_csv = log_dir / f"final_{sweep_tag}.csv"
    final_df.to_csv(final_csv, index=False)
    print(f"[OK] final summary -> {final_csv}")

    # -------------------------
    # Plot all curves together
    # -------------------------
    if curves:
        curve_png = plot_dir / f"val_curves_7_{sweep_tag}.png"
        _plot_curves(curves, curve_png, title="RTE val_acc vs step (7 lines, mean over seeds 2/3/5)")
        print(f"[OK] curves plot -> {curve_png}")

        ok_df = final_df[final_df["status"] == "ok"].copy()
        if not ok_df.empty and "name" in ok_df.columns:
            best_png = plot_dir / f"best_val_7_{sweep_tag}.png"
            final_png = plot_dir / f"final_val_7_{sweep_tag}.png"
            _plot_bars(ok_df, best_png, "max_val", "RTE best val_acc (7 lines)")
            _plot_bars(ok_df, final_png, "final_val", "RTE final val_acc (7 lines)")
            print(f"[OK] best bar -> {best_png}")
            print(f"[OK] final bar -> {final_png}")
    else:
        print("[WARN] no curves produced — check CSV columns (step, val/acc) and logging.")

    print("\n[DONE] All outputs are under:")
    print(f"  logs : {log_dir}")
    print(f"  plots: {plot_dir}")
    print(f"  runs : {io_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
