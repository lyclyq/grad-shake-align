from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch


@dataclass
class BlockStats:
    C_r: float
    C_R: float
    A_b: float


class ShakeAlignController:
    """Implements:
    - vote-based C2 stats (C_r, C_R)
    - cross-resolution alignment A_b
    - gating weights
    - in-place correction on the accumulated .grad

    Assumption: you provide per-block, per-vote flattened gradients for:
        r-branch: [A_r, B_r] concatenated
        hi-branch: [A_hi, B_hi] concatenated
    V is small (8), so we keep per-vote buffers but NEVER store the full per-example gradients.
    """

    def __init__(self, cfg: Dict):
        self.cfg = cfg
        self.V = int(cfg['method']['ours']['votes'])
        self.H = int(cfg['method']['ours']['ema_H'])
        self.eps = 1e-8
        self.history: Dict[str, BlockStats] = {}

    def compute_stats_from_votes(
        self,
        votes_r: torch.Tensor,   # [V, d_r]
        votes_hi: torch.Tensor,  # [V, d_hi]
    ) -> BlockStats:
        """Compute C_r, C_R, A_b from vote matrices."""
        V = votes_r.shape[0]
        assert V == self.V

        # Pairwise dot matrices
        D_r = votes_r @ votes_r.t()                       # [V,V]
        D_hi = votes_hi @ votes_hi.t() if votes_hi.numel() else torch.zeros_like(D_r)
        D_R = D_r + D_hi

        S_r = torch.diag(D_r).clamp_min(0)
        S_R = torch.diag(D_R).clamp_min(0)

        # Internal consistency: mean of upper-tri cos
        mask = torch.triu(torch.ones((V, V), device=votes_r.device, dtype=torch.bool), diagonal=1)
        denom_r = (S_r.sqrt().unsqueeze(1) * S_r.sqrt().unsqueeze(0) + self.eps)
        denom_R = (S_R.sqrt().unsqueeze(1) * S_R.sqrt().unsqueeze(0) + self.eps)
        C_r = (D_r[mask] / denom_r[mask]).mean().item()
        C_R = (D_R[mask] / denom_R[mask]).mean().item()

        # Alignment A_b
        # Sigma_r = sum_k v_r^(k)
        # Sigma_R = sum_k (v_r^(k), v_hi^(k))
        # With padding, <Sigma_r, Sigma_R> = ||Sigma_r||^2
        # => A_b = ||Sigma_r|| / ||Sigma_R||
        sigma_r_sq = D_r.sum()                            # ||sum v_r||^2
        sigma_hi_sq = D_hi.sum() if votes_hi.numel() else torch.tensor(0.0, device=votes_r.device)
        sigma_R_sq = sigma_r_sq + sigma_hi_sq
        A_b = (sigma_r_sq.sqrt() / (sigma_R_sq.sqrt() + self.eps)).item()

        return BlockStats(C_r=C_r, C_R=C_R, A_b=A_b)

    def ema_update(self, name: str, fresh: BlockStats) -> BlockStats:
        if name not in self.history:
            self.history[name] = fresh
            return fresh
        alpha = 2 / (self.H + 1)
        prev = self.history[name]
        smoothed = BlockStats(
            C_r=(1 - alpha) * prev.C_r + alpha * fresh.C_r,
            C_R=(1 - alpha) * prev.C_R + alpha * fresh.C_R,
            A_b=(1 - alpha) * prev.A_b + alpha * fresh.A_b,
        )
        self.history[name] = smoothed
        return smoothed

    def gates(self, s: BlockStats) -> Dict[str, float]:
        k = float(self.cfg['method']['ours']['k'])
        delta_C = s.C_R - s.C_r
        w_insuf = float(torch.sigmoid(torch.tensor(k * delta_C)))
        w_over = float(torch.sigmoid(torch.tensor(-k * delta_C)))
        w_noise = 1.0 - max(s.C_r, s.C_R)
        return {'w_insuf': w_insuf, 'w_over': w_over, 'w_noise': w_noise}

    def triggered_blocks(self, stats: Dict[str, BlockStats]) -> Tuple[float, List[str]]:
        p = float(self.cfg['method']['ours']['percentile_p'])
        mis = np.array([1.0 - s.A_b for s in stats.values()], dtype=np.float64)
        thr = float(np.percentile(mis, p))
        names = [n for n, s in stats.items() if (1.0 - s.A_b) >= thr]
        return thr, names

    @torch.no_grad()
    def apply_in_place_corrections(
        self,
        lora_modules: Dict[str, torch.nn.Module],
        stats: Dict[str, BlockStats],
        vote_sums: Dict[str, Dict[str, torch.Tensor]],
    ) -> Dict[str, float]:
        """Modifies .grad in-place on triggered modules.

        vote_sums[name] must provide:
            'sum_r'  : sum_k v_r^(k) flattened (d_r)
            'sum_R_head': head(sum_k v_R^(k)) == sum_r (d_r)
        """
        params = self.cfg['method']['ours']
        lam_n = float(params['lambda_n'])
        lam_o = float(params['lambda_o'])
        alpha_min = float(params['alpha_min'])
        beta_max = float(params['beta_max'])
        rho_max = float(params['rho_max'])
        gamma_r = float(params['gamma_r'])
        gamma_hi = float(params['gamma_hi'])
        eta = float(params['eta'])

        thr, trig = self.triggered_blocks(stats)

        triggered = 0
        for name in trig:
            mod = lora_modules[name]
            # Expect attributes from DualRankLoRALinear
            if getattr(mod, 'lora_A_r', None) is None or mod.lora_A_r.grad is None:
                continue
            triggered += 1

            g = self.gates(stats[name])

            # (A) Magnitude gating
            alpha_r = float(np.clip(1.0 - lam_n * g['w_noise'], alpha_min, 1.0))
            alpha_hi = float(np.clip(1.0 - lam_n * g['w_noise'] - lam_o * g['w_over'], alpha_min, 1.0))

            mod.lora_A_r.grad.mul_(alpha_r)
            mod.lora_B_r.grad.mul_(alpha_r)
            mod.lora_A_hi.grad.mul_(alpha_hi)
            mod.lora_B_hi.grad.mul_(alpha_hi)

            # Flatten grads for direction ops
            g_r = torch.cat([mod.lora_A_r.grad.flatten(), mod.lora_B_r.grad.flatten()])
            g_hi = torch.cat([mod.lora_A_hi.grad.flatten(), mod.lora_B_hi.grad.flatten()])

            # (B) Cascaded direction correction
            # r <- R_head (which is just sum_r, since R_head == r part)
            ref = vote_sums[name]['sum_r']
            beta_r = float(np.clip(gamma_r * g['w_insuf'], 0.0, beta_max))
            if beta_r > 0:
                g_r = (1 - beta_r) * g_r + beta_r * ref

            # hi <- P(r): simple repeat/reshape projection
            beta_hi = float(np.clip(gamma_hi * g['w_over'], 0.0, beta_max))
            if beta_hi > 0:
                # deterministic projection: repeat then trim
                proj = g_r.repeat(int(np.ceil(g_hi.numel() / g_r.numel())))[: g_hi.numel()]
                g_hi = (1 - beta_hi) * g_hi + beta_hi * proj

            # (C) Two-vector joint CAGrad
            rho = float(np.clip(eta * g['w_noise'], 0.0, rho_max))
            if rho > 1e-3:
                # build joint vectors [g_r, 0] and [0, g_hi]
                z1 = torch.cat([g_r, torch.zeros_like(g_hi)])
                z2 = torch.cat([torch.zeros_like(g_r), g_hi])
                # closed-form lambda*
                g11 = torch.dot(z1, z1)
                g22 = torch.dot(z2, z2)
                g12 = torch.dot(z1, z2)
                lam = (g11 - g12) / (g22 - g12 + self.eps)
                lam = lam.clamp(0, 1)
                z = z1 + lam * z2
                z = z / (z.norm() + self.eps)
                g_r_ca = z[: g_r.numel()]
                g_hi_ca = z[g_r.numel() :]
                g_r = (1 - rho) * g_r + rho * g_r_ca
                g_hi = (1 - rho) * g_hi + rho * g_hi_ca

            # Write back
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

        return {'alignment_threshold': thr, 'triggered_blocks': float(triggered)}
