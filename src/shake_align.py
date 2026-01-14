# src/shake_align.py
from __future__ import annotations

from dataclasses import dataclass
from collections import deque
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np
import torch


@dataclass
class BlockStats:
    C_r: float
    C_R: float
    A_b: float


def _to_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return 0.0


class ShakeAlignController:
    """
    Implements:
      - vote-based C2 stats (C_r, C_R)
      - cross-resolution alignment A_b
      - gating weights
      - in-place correction on the accumulated .grad

    Config supports:
      - method.ours.history: truncated window EMA (recommended)
      - legacy method.ours.ema_H: fallback exponential EMA if history.enabled is false
    """

    def __init__(self, cfg: Dict[str, Any], lora_modules: Optional[Dict[str, torch.nn.Module]] = None):
        self.cfg = cfg
        self.eps = 1e-8

        ours = cfg.get("method", {}).get("ours", {}) or {}
        self.V = int(ours.get("votes", 8))
        self.H_legacy = int(ours.get("ema_H", 4))

        # ---- Truncated window history config ----
        hist = ours.get("history", {}) or {}
        self.hist_enabled = bool(hist.get("enabled", False))
        self.hist_window = int(hist.get("window_steps", self.H_legacy))
        self.hist_weighting = str(hist.get("weighting", "exp"))  # exp | uniform | linear
        self.hist_beta = float(hist.get("exp_beta", 0.7))

        # store per-block recent fresh stats (truncated)
        self._hist: Dict[str, Deque[BlockStats]] = {}

        # legacy single EMA state
        self._ema: Dict[str, BlockStats] = {}

        # lora modules
        self.lora_modules: Dict[str, torch.nn.Module] = lora_modules or {}

        # optional step buffers (if you use capture_vote_grads route)
        self.votes_r: Dict[str, List[torch.Tensor]] = {}
        self.votes_hi: Dict[str, List[torch.Tensor]] = {}

    def set_lora_modules(self, lora_modules: Dict[str, torch.nn.Module]) -> None:
        self.lora_modules = lora_modules

    # -------------------------
    # Stats
    # -------------------------
    def compute_stats_from_votes(
        self,
        votes_r: torch.Tensor,   # [V, d_r]
        votes_hi: torch.Tensor,  # [V, d_hi]
    ) -> BlockStats:
        """Compute C_r, C_R, A_b from vote matrices."""
        V = votes_r.shape[0]
        if V < 2:
            # edge: no pairwise cos
            return BlockStats(C_r=0.0, C_R=0.0, A_b=1.0)

        D_r = votes_r @ votes_r.t()  # [V, V]
        if votes_hi is not None and votes_hi.numel():
            D_hi = votes_hi @ votes_hi.t()
        else:
            D_hi = torch.zeros_like(D_r)

        D_R = D_r + D_hi

        # diagonal (squared norms) should be >=0; clamp for numeric safety
        S_r = torch.diag(D_r).clamp_min(0)
        S_R = torch.diag(D_R).clamp_min(0)

        mask = torch.triu(torch.ones((V, V), device=votes_r.device, dtype=torch.bool), diagonal=1)

        denom_r = (S_r.sqrt().unsqueeze(1) * S_r.sqrt().unsqueeze(0) + self.eps)
        denom_R = (S_R.sqrt().unsqueeze(1) * S_R.sqrt().unsqueeze(0) + self.eps)

        # NOTE: cosine can be negative if gradients are anti-aligned. That is *mathematically valid*.
        C_r = (D_r[mask] / denom_r[mask]).mean().item()
        C_R = (D_R[mask] / denom_R[mask]).mean().item()

        # Alignment A_b:
        # A_b = ||Σ_r|| / ||Σ_R||, with Σ_R = [Σ_r, Σ_hi], so ||Σ_R||^2 = ||Σ_r||^2 + ||Σ_hi||^2
        # where ||Σ_r||^2 = sum_{i,j} <v_i, v_j> = D_r.sum()
        sigma_r_sq = D_r.sum()
        sigma_hi_sq = D_hi.sum() if (votes_hi is not None and votes_hi.numel()) else torch.tensor(0.0, device=votes_r.device)
        sigma_R_sq = sigma_r_sq + sigma_hi_sq
        A_b = (sigma_r_sq.sqrt() / (sigma_R_sq.sqrt() + self.eps)).item()

        return BlockStats(C_r=float(C_r), C_R=float(C_R), A_b=float(A_b))

    # -------------------------
    # Window EMA (truncated) / Legacy EMA
    # -------------------------
    def _weighted_average(self, seq: List[BlockStats]) -> BlockStats:
        """
        Weighted avg over a list of BlockStats, where seq is ordered oldest -> newest.
        """
        n = len(seq)
        if n == 0:
            return BlockStats(0.0, 0.0, 1.0)

        if self.hist_weighting == "uniform":
            w = np.ones(n, dtype=np.float64)
        elif self.hist_weighting == "linear":
            # newer gets bigger weight
            w = np.arange(1, n + 1, dtype=np.float64)
        else:
            # exp decay: newest has weight 1, previous beta, previous beta^2, ...
            beta = float(self.hist_beta)
            beta = max(0.0, min(0.9999, beta))
            # oldest -> newest => weights should increase towards newest
            w = np.array([beta ** (n - 1 - i) for i in range(n)], dtype=np.float64)

        w = w / (w.sum() + 1e-12)

        Cr = float(sum(w[i] * seq[i].C_r for i in range(n)))
        CR = float(sum(w[i] * seq[i].C_R for i in range(n)))
        Ab = float(sum(w[i] * seq[i].A_b for i in range(n)))
        return BlockStats(C_r=Cr, C_R=CR, A_b=Ab)

    def ema_update(self, name: str, fresh: BlockStats) -> BlockStats:
        """
        If history.enabled:
          keep last window_steps fresh stats and compute weighted average (exp/uniform/linear).
        Else:
          use legacy exponential EMA with H_legacy.
        """
        if self.hist_enabled:
            dq = self._hist.get(name)
            if dq is None:
                dq = deque(maxlen=max(1, int(self.hist_window)))
                self._hist[name] = dq
            dq.append(fresh)
            smoothed = self._weighted_average(list(dq))
            return smoothed

        # legacy
        if name not in self._ema:
            self._ema[name] = fresh
            return fresh
        alpha = 2.0 / (float(self.H_legacy) + 1.0)
        prev = self._ema[name]
        smoothed = BlockStats(
            C_r=(1 - alpha) * prev.C_r + alpha * fresh.C_r,
            C_R=(1 - alpha) * prev.C_R + alpha * fresh.C_R,
            A_b=(1 - alpha) * prev.A_b + alpha * fresh.A_b,
        )
        self._ema[name] = smoothed
        return smoothed

    def history_debug_state(self, name: str) -> Dict[str, Any]:
        """
        For debug printing: show window contents and weights.
        """
        if not self.hist_enabled:
            return {"enabled": False, "mode": "legacy"}

        dq = self._hist.get(name, deque())
        seq = list(dq)
        n = len(seq)

        if n == 0:
            return {"enabled": True, "mode": self.hist_weighting, "n": 0}

        # reconstruct weights (same as _weighted_average)
        if self.hist_weighting == "uniform":
            w = np.ones(n, dtype=np.float64)
        elif self.hist_weighting == "linear":
            w = np.arange(1, n + 1, dtype=np.float64)
        else:
            beta = max(0.0, min(0.9999, float(self.hist_beta)))
            w = np.array([beta ** (n - 1 - i) for i in range(n)], dtype=np.float64)
        w = w / (w.sum() + 1e-12)

        return {
            "enabled": True,
            "mode": self.hist_weighting,
            "window_steps": int(self.hist_window),
            "n": int(n),
            "weights": [float(x) for x in w.tolist()],
            "fresh_seq": [{"C_r": s.C_r, "C_R": s.C_R, "A_b": s.A_b} for s in seq],
        }

    # -------------------------
    # Gates / triggers
    # -------------------------
    def gates(self, s: BlockStats) -> Dict[str, float]:
        ours = self.cfg.get("method", {}).get("ours", {}) or {}
        k = float(ours.get("k", 8.0))
        delta_C = s.C_R - s.C_r

        # sigmoid gates
        w_insuf = float(torch.sigmoid(torch.tensor(k * delta_C)))
        w_over = float(torch.sigmoid(torch.tensor(-k * delta_C)))

        # noise proxy: 1 - max(Cr, CR) (can exceed 1 if C is negative; that's fine but you may clamp downstream)
        w_noise = 1.0 - max(s.C_r, s.C_R)

        return {"w_insuf": w_insuf, "w_over": w_over, "w_noise": w_noise, "delta_C": float(delta_C)}

    def triggered_blocks(self, stats: Dict[str, BlockStats]) -> Tuple[float, List[str], Dict[str, float]]:
        ours = self.cfg.get("method", {}).get("ours", {}) or {}
        p = float(ours.get("percentile_p", 90))
        misalign_map = {n: float(1.0 - s.A_b) for n, s in stats.items()}
        mis = np.array(list(misalign_map.values()), dtype=np.float64)
        thr = float(np.percentile(mis, p)) if mis.size else 0.0
        names = [n for n, m in misalign_map.items() if m >= thr]
        return thr, names, misalign_map

    # -------------------------
    # Optional: internal step buffers (unused in your trainer right now)
    # -------------------------
    def reset_step_buffers(self) -> None:
        self.votes_r = {k: [] for k in self.lora_modules.keys()}
        self.votes_hi = {k: [] for k in self.lora_modules.keys()}

    @torch.no_grad()
    def capture_vote_grads(self, vote_idx: int) -> None:
        del vote_idx
        for name, mod in self.lora_modules.items():
            if getattr(mod, "lora_A_r", None) is None:
                continue
            g_ar = mod.lora_A_r.grad if mod.lora_A_r.grad is not None else torch.zeros_like(mod.lora_A_r)
            g_br = mod.lora_B_r.grad if mod.lora_B_r.grad is not None else torch.zeros_like(mod.lora_B_r)
            g_ahi = mod.lora_A_hi.grad if mod.lora_A_hi.grad is not None else torch.zeros_like(mod.lora_A_hi)
            g_bhi = mod.lora_B_hi.grad if mod.lora_B_hi.grad is not None else torch.zeros_like(mod.lora_B_hi)
            self.votes_r[name].append(torch.cat([g_ar.flatten(), g_br.flatten()]).detach().clone())
            self.votes_hi[name].append(torch.cat([g_ahi.flatten(), g_bhi.flatten()]).detach().clone())

    def compute_stats(self) -> Tuple[Dict[str, BlockStats], Dict[str, Dict[str, torch.Tensor]]]:
        stats: Dict[str, BlockStats] = {}
        vote_sums: Dict[str, Dict[str, torch.Tensor]] = {}
        for name in self.votes_r.keys():
            if len(self.votes_r[name]) == 0:
                continue
            votes_r = torch.stack(self.votes_r[name], dim=0)
            votes_hi = torch.stack(self.votes_hi[name], dim=0)
            fresh = self.compute_stats_from_votes(votes_r, votes_hi)
            stats[name] = self.ema_update(name, fresh)
            vote_sums[name] = {"sum_r": votes_r.sum(dim=0), "sum_hi": votes_hi.sum(dim=0)}
        return stats, vote_sums

    # -------------------------
    # Execution: in-place corrections
    # -------------------------
    @torch.no_grad()
    def apply_in_place_corrections(
        self,
        lora_modules: Dict[str, torch.nn.Module],
        stats: Dict[str, BlockStats],
        vote_sums: Dict[str, Dict[str, torch.Tensor]],
        debug: bool = False,
        grad_norm_trace: bool = False,
        debug_history: bool = False,
    ) -> Dict[str, Any]:
        """
        Modifies .grad in-place on triggered modules.

        vote_sums[name] must provide:
            'sum_r' : Σ_r (flattened) in r-space (d_r)
        """
        ours = self.cfg.get("method", {}).get("ours", {}) or {}
        lam_n = float(ours.get("lambda_n", 0.4))
        lam_o = float(ours.get("lambda_o", 0.4))
        alpha_min = float(ours.get("alpha_min", 0.1))
        beta_max = float(ours.get("beta_max", 0.5))
        rho_max = float(ours.get("rho_max", 0.5))
        gamma_r = float(ours.get("gamma_r", 0.3))
        gamma_hi = float(ours.get("gamma_hi", 0.3))
        eta = float(ours.get("eta", 0.2))

        thr, trig, misalign_map = self.triggered_blocks(stats)

        info: Dict[str, Any] = {
            "alignment_threshold": float(thr),
            "triggered_blocks": float(0.0),
        }
        if debug:
            info["per_block"] = {}
        if grad_norm_trace:
            info["per_block_grad_norm"] = {}
        if debug_history:
            info["per_block_history"] = {}

        triggered = 0

        for name in trig:
            mod = lora_modules.get(name, None)
            if mod is None:
                continue

            # must have grads
            if getattr(mod, "lora_A_r", None) is None:
                continue
            if mod.lora_A_r.grad is None or mod.lora_B_r.grad is None or mod.lora_A_hi.grad is None or mod.lora_B_hi.grad is None:
                continue

            triggered += 1

            s = stats[name]
            g = self.gates(s)
            mis = float(misalign_map.get(name, 1.0 - s.A_b))

            # norm snapshot before
            if grad_norm_trace:
                g_r_before = float(torch.cat([mod.lora_A_r.grad.flatten(), mod.lora_B_r.grad.flatten()]).norm().item())
                g_hi_before = float(torch.cat([mod.lora_A_hi.grad.flatten(), mod.lora_B_hi.grad.flatten()]).norm().item())

            # (A) Magnitude gating
            alpha_r_raw = 1.0 - lam_n * g["w_noise"]
            alpha_hi_raw = 1.0 - lam_n * g["w_noise"] - lam_o * g["w_over"]
            alpha_r = float(np.clip(alpha_r_raw, alpha_min, 1.0))
            alpha_hi = float(np.clip(alpha_hi_raw, alpha_min, 1.0))

            mod.lora_A_r.grad.mul_(alpha_r)
            mod.lora_B_r.grad.mul_(alpha_r)
            mod.lora_A_hi.grad.mul_(alpha_hi)
            mod.lora_B_hi.grad.mul_(alpha_hi)

            # flatten for direction ops
            g_r = torch.cat([mod.lora_A_r.grad.flatten(), mod.lora_B_r.grad.flatten()])
            g_hi = torch.cat([mod.lora_A_hi.grad.flatten(), mod.lora_B_hi.grad.flatten()])

            # (B) Cascaded direction correction
            ref = vote_sums[name]["sum_r"]
            beta_r_raw = gamma_r * g["w_insuf"]
            beta_r = float(np.clip(beta_r_raw, 0.0, beta_max))
            if beta_r > 0:
                g_r = (1 - beta_r) * g_r + beta_r * ref

            beta_hi_raw = gamma_hi * g["w_over"]
            beta_hi = float(np.clip(beta_hi_raw, 0.0, beta_max))
            if beta_hi > 0:
                proj = g_r.repeat(int(np.ceil(g_hi.numel() / g_r.numel())))[: g_hi.numel()]
                g_hi = (1 - beta_hi) * g_hi + beta_hi * proj

            # (C) Two-vector joint CAGrad
            rho_raw = eta * g["w_noise"]
            rho = float(np.clip(rho_raw, 0.0, rho_max))
            if rho > 1e-3:
                z1 = torch.cat([g_r, torch.zeros_like(g_hi)])
                z2 = torch.cat([torch.zeros_like(g_r), g_hi])

                g11 = torch.dot(z1, z1)
                g22 = torch.dot(z2, z2)
                g12 = torch.dot(z1, z2)

                lam = (g11 - g12) / (g22 - g12 + self.eps)
                lam = lam.clamp(0, 1)

                z = z1 + lam * z2
                z = z / (z.norm() + self.eps)

                g_r_ca = z[: g_r.numel()]
                g_hi_ca = z[g_r.numel():]
                g_r = (1 - rho) * g_r + rho * g_r_ca
                g_hi = (1 - rho) * g_hi + rho * g_hi_ca

            # write back
            a_r_n = mod.lora_A_r.numel()
            b_r_n = mod.lora_B_r.numel()
            mod.lora_A_r.grad.copy_(g_r[:a_r_n].view_as(mod.lora_A_r))
            mod.lora_B_r.grad.copy_(g_r[a_r_n:a_r_n + b_r_n].view_as(mod.lora_B_r))

            a_hi_n = mod.lora_A_hi.numel()
            b_hi_n = mod.lora_B_hi.numel()
            off = 0
            mod.lora_A_hi.grad.copy_(g_hi[off:off + a_hi_n].view_as(mod.lora_A_hi))
            off += a_hi_n
            mod.lora_B_hi.grad.copy_(g_hi[off:off + b_hi_n].view_as(mod.lora_B_hi))

            # norm snapshot after
            if grad_norm_trace:
                g_r_after = float(torch.cat([mod.lora_A_r.grad.flatten(), mod.lora_B_r.grad.flatten()]).norm().item())
                g_hi_after = float(torch.cat([mod.lora_A_hi.grad.flatten(), mod.lora_B_hi.grad.flatten()]).norm().item())
                info["per_block_grad_norm"][name] = {
                    "g_r_before": g_r_before,
                    "g_r_after": g_r_after,
                    "g_hi_before": g_hi_before,
                    "g_hi_after": g_hi_after,
                }

            if debug:
                info["per_block"][name] = {
                    "A_b": float(s.A_b),
                    "misalign": float(mis),
                    "C_r": float(s.C_r),
                    "C_R": float(s.C_R),
                    "delta_C": float(g["delta_C"]),
                    "w_insuf": float(g["w_insuf"]),
                    "w_over": float(g["w_over"]),
                    "w_noise": float(g["w_noise"]),
                    "alpha_r_raw": float(alpha_r_raw),
                    "alpha_hi_raw": float(alpha_hi_raw),
                    "alpha_r": float(alpha_r),
                    "alpha_hi": float(alpha_hi),
                    "beta_r_raw": float(beta_r_raw),
                    "beta_hi_raw": float(beta_hi_raw),
                    "beta_r": float(beta_r),
                    "beta_hi": float(beta_hi),
                    "rho_raw": float(rho_raw),
                    "rho": float(rho),
                    "tau": float(thr),
                }

            if debug_history:
                info["per_block_history"][name] = self.history_debug_state(name)

        info["triggered_blocks"] = float(triggered)
        return info
