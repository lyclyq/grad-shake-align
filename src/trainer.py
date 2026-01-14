# src/trainer.py
from __future__ import annotations

from typing import Any, Dict, List, Tuple
from itertools import combinations

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

from .loggingx import RunLogger
from .shake_align import ShakeAlignController, BlockStats
from .lora_layers import debug_check_dualrank_init


def _named_dualrank_lora_modules(model) -> Dict[str, torch.nn.Module]:
    out: Dict[str, torch.nn.Module] = {}
    for name, m in model.named_modules():
        if hasattr(m, "lora_A_r") and hasattr(m, "lora_A_hi"):
            out[name] = m
    return out


def _flatten_branch_grads(mod: torch.nn.Module) -> Tuple[torch.Tensor, torch.Tensor]:
    device = next(mod.parameters()).device

    def g_or_zeros(p: torch.nn.Parameter) -> torch.Tensor:
        if p.grad is None:
            return torch.zeros_like(p, device=device).flatten()
        return p.grad.detach().flatten()

    g_r = torch.cat([g_or_zeros(mod.lora_A_r), g_or_zeros(mod.lora_B_r)], dim=0)
    g_hi = torch.cat([g_or_zeros(mod.lora_A_hi), g_or_zeros(mod.lora_B_hi)], dim=0)
    return g_r, g_hi


@torch.no_grad()
def evaluate(model, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(**batch)
        preds = out.logits.argmax(dim=-1)
        labels = batch["labels"]
        correct += (preds == labels).sum().item()
        total += labels.numel()
    model.train()
    return correct / max(total, 1)


def _dbg_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    d = cfg.get("debug", {}) or {}
    return {
        "enabled": bool(d.get("enabled", False)),
        "print_every_steps": int(d.get("print_every_steps", 50)),
        "max_blocks_to_print": int(d.get("max_blocks_to_print", 3)),
        "dump_init": bool(d.get("dump_init", True)),
        "dump_votes": bool(d.get("dump_votes", True)),
        "dump_gates": bool(d.get("dump_gates", True)),
        "dump_grad_norms": bool(d.get("dump_grad_norms", True)),
        "dump_history": bool(d.get("dump_history", False)),
        "assert_hi_zero_init": bool(d.get("assert_hi_zero_init", True)),
    }


def _vote_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    ours = cfg.get("method", {}).get("ours", {}) or {}
    voting = ours.get("voting", {}) or {}
    return {
        # how many samples per vote window
        "samples_per_vote": int(voting.get("samples_per_vote", 8)),  # 16/8/4
        # combine votes (using existing windows, no extra backward)
        "combine_enabled": bool(voting.get("combine_votes", False)),
        "combine_size": int(voting.get("combine_size", 2)),  # combine 2 windows -> 1 combined vote
        "combine_max": int(voting.get("combine_max", 0)),  # 0 means no cap; otherwise cap #combined votes
        # if true, also include the original single-window votes (recommended)
        "keep_single_votes": bool(voting.get("keep_single_votes", True)),
        # If batch not divisible, allow last smaller vote
        "allow_tail": bool(voting.get("allow_tail", True)),
    }


def _split_indices(n: int, chunk: int, allow_tail: bool) -> List[Tuple[int, int]]:
    out = []
    s = 0
    while s < n:
        e = min(s + chunk, n)
        if (e - s) < chunk and (not allow_tail):
            break
        out.append((s, e))
        s = e
    return out


def train_one(
    cfg: dict,
    model,
    train_loader: DataLoader,
    val_loader: DataLoader,
    logger: RunLogger,
) -> Dict[str, float]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    epochs = int(cfg["train"]["epochs"])
    lr = float(cfg["train"]["lr"])
    warmup_ratio = float(cfg["train"]["warmup_ratio"])
    weight_decay = float(cfg["train"]["weight_decay"])
    max_grad_norm = float(cfg["train"].get("max_grad_norm", 1.0))

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    total_steps = epochs * len(train_loader)
    warmup_steps = int(total_steps * warmup_ratio)
    sched = get_linear_schedule_with_warmup(opt, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    lora_modules = _named_dualrank_lora_modules(model)

    dbg = _dbg_cfg(cfg)
    vote_cfg = _vote_cfg(cfg)

    controller = ShakeAlignController(cfg) if cfg["method"]["name"] == "ours" else None
    if controller is not None:
        controller.set_lora_modules(lora_modules)

    # ---- Debug init check (once) ----
    if controller is not None and dbg["enabled"] and dbg["dump_init"]:
        debug_check_dualrank_init(
            model,
            assert_hi_zero=dbg["assert_hi_zero_init"],
            max_blocks_to_print=dbg["max_blocks_to_print"],
        )

    best_val = -1.0
    best_epoch = -1
    val_history: List[float] = []

    global_step = 0
    eval_strategy = cfg["train"]["eval"]["strategy"]
    dense_eval_per_epoch = int(cfg["train"]["eval"].get("dense_early_per_epoch", 8))
    dense_early_epochs = int(cfg["train"]["eval"].get("dense_early_epochs", 2))

    for ep in range(1, epochs + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"epoch {ep}/{epochs}")

        for batch in pbar:
            global_step += 1
            batch = {k: v.to(device) for k, v in batch.items()}

            if controller is None:
                out = model(**batch)
                loss = out.loss
                loss.backward()
                logger.log(global_step, {"train/loss": float(loss.item())})

            else:
                # -------------------------
                # Voting windows
                # -------------------------
                bs = int(batch["input_ids"].shape[0])
                spv = int(vote_cfg["samples_per_vote"])
                allow_tail = bool(vote_cfg["allow_tail"])

                windows = _split_indices(bs, spv, allow_tail=allow_tail)
                if len(windows) == 0:
                    windows = [(0, bs)]

                # per-window vote vectors (single-window votes)
                win_votes_r: Dict[str, List[torch.Tensor]] = {n: [] for n in lora_modules.keys()}
                win_votes_hi: Dict[str, List[torch.Tensor]] = {n: [] for n in lora_modules.keys()}

                # accumulate total grads over windows (sum => full batch grad)
                total_grads: Dict[torch.nn.Parameter, torch.Tensor] = {}
                loss = None

                for (s, e) in windows:
                    sub = {k: v[s:e] for k, v in batch.items()}

                    opt.zero_grad(set_to_none=True)
                    out = model(**sub)
                    loss = out.loss
                    loss.backward()

                    # snapshot per-block vote grads for this window
                    for name, mod in lora_modules.items():
                        g_r, g_hi = _flatten_branch_grads(mod)
                        win_votes_r[name].append(g_r)
                        win_votes_hi[name].append(g_hi)

                    # accumulate into total grads
                    with torch.no_grad():
                        for p in model.parameters():
                            if p.grad is None:
                                continue
                            if p not in total_grads:
                                total_grads[p] = p.grad.detach().clone()
                            else:
                                total_grads[p].add_(p.grad.detach())

                # restore accumulated grad (full batch)
                opt.zero_grad(set_to_none=True)
                with torch.no_grad():
                    for p, g in total_grads.items():
                        p.grad = g

                # -------------------------
                # Build vote matrices (single + combined)
                # -------------------------
                votes_r: Dict[str, List[torch.Tensor]] = {n: [] for n in lora_modules.keys()}
                votes_hi: Dict[str, List[torch.Tensor]] = {n: [] for n in lora_modules.keys()}

                # keep single votes?
                if vote_cfg["keep_single_votes"]:
                    for name in lora_modules.keys():
                        votes_r[name].extend(win_votes_r[name])
                        votes_hi[name].extend(win_votes_hi[name])

                # add combined votes?
                if vote_cfg["combine_enabled"]:
                    comb_k = int(vote_cfg["combine_size"])
                    max_comb = int(vote_cfg["combine_max"])
                    idxs = list(range(len(windows)))
                    combs = list(combinations(idxs, comb_k)) if len(idxs) >= comb_k else []
                    if max_comb > 0:
                        combs = combs[:max_comb]

                    # combined vote = sum of selected window votes (per block)
                    for name in lora_modules.keys():
                        for c in combs:
                            vr = torch.stack([win_votes_r[name][i] for i in c], dim=0).sum(dim=0)
                            vhi = torch.stack([win_votes_hi[name][i] for i in c], dim=0).sum(dim=0)
                            votes_r[name].append(vr)
                            votes_hi[name].append(vhi)

                # -------------------------
                # stats per block (+ window EMA in controller)
                # -------------------------
                stats: Dict[str, BlockStats] = {}
                vote_sums: Dict[str, Dict[str, torch.Tensor]] = {}

                for name in lora_modules.keys():
                    if len(votes_r[name]) == 0:
                        continue
                    vr = torch.stack(votes_r[name], dim=0)     # [V_eff, d_r]
                    vhi = torch.stack(votes_hi[name], dim=0)   # [V_eff, d_hi]
                    fresh = controller.compute_stats_from_votes(vr, vhi)
                    smooth = controller.ema_update(name, fresh)
                    stats[name] = smooth
                    vote_sums[name] = {"sum_r": vr.sum(dim=0)}  # Σ_r in r-space

                # ---- periodic vote/stat debug ----
                if dbg["enabled"] and dbg["dump_votes"] and (global_step % dbg["print_every_steps"] == 0):
                    items = []
                    for n, s in stats.items():
                        mis = float(1.0 - s.A_b)
                        srn = float(vote_sums[n]["sum_r"].norm().item()) if n in vote_sums else 0.0
                        items.append((mis, n, s, srn))
                    items.sort(reverse=True, key=lambda x: x[0])

                    print(f"[DBG][Step={global_step}][VoteCfg] bs={bs} spv={spv} windows={len(windows)} "
                          f"single_votes={len(win_votes_r[next(iter(lora_modules.keys()))]) if lora_modules else 0} "
                          f"combine={vote_cfg['combine_enabled']} size={vote_cfg['combine_size']}")
                    print(f"[DBG][Step={global_step}][Vote] top{dbg['max_blocks_to_print']} by misalign")
                    for mis, n, s, srn in items[: dbg["max_blocks_to_print"]]:
                        print(
                            f"  [{n}] A={s.A_b:.4f} mis={mis:.4f} Cr={s.C_r:.4f} CR={s.C_R:.4f} ||sum_r||={srn:.4f}"
                        )

                # apply correction (+ gate trace + optional history trace)
                info = controller.apply_in_place_corrections(
                    lora_modules=lora_modules,
                    stats=stats,
                    vote_sums=vote_sums,
                    debug=bool(dbg["enabled"] and dbg["dump_gates"]),
                    grad_norm_trace=bool(dbg["enabled"] and dbg["dump_grad_norms"]),
                    debug_history=bool(dbg["enabled"] and dbg["dump_history"]),
                )

                # ---- gate debug printing (only triggered blocks) ----
                if dbg["enabled"] and dbg["dump_gates"] and (global_step % dbg["print_every_steps"] == 0):
                    thr = float(info.get("alignment_threshold", 0.0))
                    trig = int(float(info.get("triggered_blocks", 0.0)))
                    print(f"[DBG][Step={global_step}][Gate] tau={thr:.4f} triggered={trig}/{len(lora_modules)}")

                    per = info.get("per_block", {}) or {}
                    shown = 0
                    for n, t in per.items():
                        if shown >= dbg["max_blocks_to_print"]:
                            break
                        shown += 1
                        print(
                            f"  [{n}] A={t['A_b']:.4f} mis={t['misalign']:.4f} "
                            f"Cr={t['C_r']:.4f} CR={t['C_R']:.4f} dC={t['delta_C']:.4f} "
                            f"wN={t['w_noise']:.4f} wO={t['w_over']:.4f} wI={t['w_insuf']:.4f}"
                        )
                        print(
                            f"    alpha_r: {t['alpha_r_raw']:.4f}->{t['alpha_r']:.4f}  "
                            f"alpha_hi: {t['alpha_hi_raw']:.4f}->{t['alpha_hi']:.4f}"
                        )
                        print(
                            f"    beta_r: {t['beta_r_raw']:.4f}->{t['beta_r']:.4f}  "
                            f"beta_hi: {t['beta_hi_raw']:.4f}->{t['beta_hi']:.4f}  "
                            f"rho: {t['rho_raw']:.4f}->{t['rho']:.4f}"
                        )
                        if dbg["dump_grad_norms"]:
                            gn = (info.get("per_block_grad_norm", {}) or {}).get(n, None)
                            if gn is not None:
                                print(
                                    f"    ||g_r|| {gn['g_r_before']:.4f}->{gn['g_r_after']:.4f}  "
                                    f"||g_hi|| {gn['g_hi_before']:.4f}->{gn['g_hi_after']:.4f}"
                                )
                        if dbg["dump_history"]:
                            hd = (info.get("per_block_history", {}) or {}).get(n, None)
                            if hd is not None and hd.get("enabled", False):
                                print(f"    [HIST] mode={hd.get('mode')} n={hd.get('n')} window={hd.get('window_steps')}")
                                # 只打权重，窗口内容太长的话你自己看 info dict 更稳
                                print(f"    [HIST] weights={hd.get('weights')}")

                loss_val = float(loss.item()) if loss is not None else 0.0
                logger.log(
                    global_step,
                    {
                        "train/loss": loss_val,
                        "train/triggered_blocks": info.get("triggered_blocks", 0.0),
                        "train/align_thr": info.get("alignment_threshold", 0.0),
                    },
                )

            # step
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)

            pbar.set_postfix({"loss": f"{float(loss.item()) if loss is not None else 0.0:.4f}"})

            # dense early eval
            if eval_strategy == "dense_early" and ep <= dense_early_epochs:
                every = max(1, len(train_loader) // dense_eval_per_epoch)
                if (global_step % every) == 0:
                    val_acc = evaluate(model, val_loader, device)
                    val_history.append(val_acc)
                    logger.log(global_step, {"val/acc": val_acc})
                    if val_acc > best_val:
                        best_val = val_acc
                        best_epoch = ep

        # per-epoch eval
        if eval_strategy == "per_epoch":
            val_acc = evaluate(model, val_loader, device)
            val_history.append(val_acc)
            logger.log(global_step, {"val/acc": val_acc, "epoch": ep})
            if val_acc > best_val:
                best_val = val_acc
                best_epoch = ep

    # summarize val stats
    if len(val_history) == 0:
        val_max = best_val
        val_final = best_val
        val_avg = best_val
    else:
        val_max = float(max(val_history))
        val_final = float(val_history[-1])
        val_avg = float(sum(val_history) / len(val_history))

    return {
        "best_val_acc": float(best_val),
        "best_epoch": float(best_epoch),
        "val_max": val_max,
        "val_final": val_final,
        "val_avg": val_avg,
    }
