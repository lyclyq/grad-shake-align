# src/lora_layers.py
from __future__ import annotations

import math
from typing import Dict, Iterator, List, Optional, Tuple

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """Drop-in replacement for nn.Linear with LoRA (A @ B) adaptation.
    Common init: A random, B zero => adaptation = 0 at step 0.
    """

    def __init__(self, base: nn.Linear, r: int, alpha: float, dropout: float = 0.0):
        super().__init__()
        self.base = base
        self.r = r
        self.scaling = alpha / r if r > 0 else 1.0
        self.dropout = nn.Dropout(dropout)

        in_dim = base.in_features
        out_dim = base.out_features

        self.lora_A = nn.Parameter(torch.empty(r, in_dim))
        self.lora_B = nn.Parameter(torch.zeros(out_dim, r))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.base(x)
        if self.r <= 0:
            return res
        dx = self.dropout(x)
        lora = (dx @ self.lora_A.T) @ self.lora_B.T
        return res + lora * self.scaling


class DualRankLoRALinear(nn.Module):
    """Dual-rank LoRA: rank-r core + rank-(R-r) extension.

    Naming (matches your LaTeX):
      - r branch : (U_r, V_r) ~ (A_r, B_r)
      - hi branch: (U_hi,V_hi)~ (A_hi,B_hi)

    Init:
      - A_r: Kaiming, B_r: 0
      - A_hi: Normal,  B_hi: 0
    => both branches contribute 0 at step 0, but have gradients.
    """

    def __init__(self, base: nn.Linear, r: int, R: int, alpha: float, dropout: float = 0.0):
        super().__init__()
        assert R > r >= 1
        self.base = base
        self.r = r
        self.R = R
        self.hi = R - r

        # scaling uses core convention alpha/r (consistent with your current code)
        self.scaling = alpha / r
        self.dropout = nn.Dropout(dropout)

        in_dim = base.in_features
        out_dim = base.out_features

        self.lora_A_r = nn.Parameter(torch.empty(r, in_dim))
        self.lora_B_r = nn.Parameter(torch.zeros(out_dim, r))

        self.lora_A_hi = nn.Parameter(torch.empty(self.hi, in_dim))
        self.lora_B_hi = nn.Parameter(torch.zeros(out_dim, self.hi))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.lora_A_r, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B_r)

        nn.init.normal_(self.lora_A_hi, std=0.02)
        nn.init.zeros_(self.lora_B_hi)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.base(x)
        dx = self.dropout(x)

        path_r = (dx @ self.lora_A_r.T) @ self.lora_B_r.T
        path_hi = (dx @ self.lora_A_hi.T) @ self.lora_B_hi.T
        return res + (path_r + path_hi) * self.scaling


def _iter_named_linears(model: nn.Module):
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            yield name, module


def inject_lora(
    model: nn.Module,
    mode: str,
    r: int,
    R: int,
    alpha: float,
    dropout: float,
    target_substrings: Optional[List[str]] = None,
) -> List[str]:
    """Replace selected nn.Linear modules with LoRA-wrapped modules."""
    assert mode in {"baseline", "ours"}

    replaced: List[str] = []

    def should_replace(name: str) -> bool:
        if not target_substrings:
            return True
        return any(s in name for s in target_substrings)

    # Need parent modules to replace children
    name_to_parent = {}
    for full_name, module in model.named_modules():
        for child_name, _child in module.named_children():
            name_to_parent[f"{full_name}.{child_name}".strip(".")] = (module, child_name)

    for name, lin in list(_iter_named_linears(model)):
        if not should_replace(name):
            continue
        parent = name_to_parent.get(name)
        if parent is None:
            continue
        pmod, attr = parent

        if mode == "baseline":
            wrapped = LoRALinear(lin, r=r, alpha=alpha, dropout=dropout)
        else:
            wrapped = DualRankLoRALinear(lin, r=r, R=R, alpha=alpha, dropout=dropout)

        setattr(pmod, attr, wrapped)
        replaced.append(name)

    return replaced


def lora_block_names(model: nn.Module) -> List[str]:
    names = []
    for name, m in model.named_modules():
        if hasattr(m, "lora_A_r") or hasattr(m, "lora_A"):
            names.append(name)
    return names


# -------------------------
# Debug helpers (NEW)
# -------------------------
def iter_dualrank_modules(model: nn.Module) -> Iterator[Tuple[str, DualRankLoRALinear]]:
    for name, m in model.named_modules():
        if hasattr(m, "lora_A_r") and hasattr(m, "lora_A_hi"):
            yield name, m  # type: ignore


@torch.no_grad()
def debug_check_dualrank_init(
    model: nn.Module,
    assert_hi_zero: bool = True,
    max_blocks_to_print: int = 3,
) -> Dict[str, Dict[str, float]]:
    """
    Prints/returns initialization checks for DualRankLoRALinear:
      - nested existence
      - ||B_hi|| == 0 (zero contribution)
    Returns: per_block norms dict for optional logging.
    """
    blocks = list(iter_dualrank_modules(model))
    print(f"[DBG][Init] dualrank_blocks={len(blocks)} assert_hi_zero={assert_hi_zero}")

    out: Dict[str, Dict[str, float]] = {}
    for i, (name, m) in enumerate(blocks):
        Ar = m.lora_A_r
        Br = m.lora_B_r
        Ahi = m.lora_A_hi
        Bhi = m.lora_B_hi

        n_Ar = float(Ar.norm().item())
        n_Br = float(Br.norm().item())
        n_Ahi = float(Ahi.norm().item())
        n_Bhi = float(Bhi.norm().item())

        out[name] = {
            "Ar_norm": n_Ar,
            "Br_norm": n_Br,
            "Ahi_norm": n_Ahi,
            "Bhi_norm": n_Bhi,
        }

        ok = (n_Bhi == 0.0)
        if assert_hi_zero and not ok:
            raise AssertionError(f"[DBG][Init] {name}: expected ||Bhi||=0 but got {n_Bhi}")

        if i < max_blocks_to_print:
            print(
                f"[DBG][Init][{name}] "
                f"Ar={tuple(Ar.shape)} Br={tuple(Br.shape)} "
                f"Ahi={tuple(Ahi.shape)} Bhi={tuple(Bhi.shape)} "
                f"||Ar||={n_Ar:.4f} ||Br||={n_Br:.4f} ||Ahi||={n_Ahi:.4f} ||Bhi||={n_Bhi:.4f} "
                f"{'OK' if ok else 'NOT_ZERO'}"
            )

    if len(blocks) > max_blocks_to_print:
        print(f"[DBG][Init] ... ({len(blocks) - max_blocks_to_print} more blocks omitted)")

    return out
