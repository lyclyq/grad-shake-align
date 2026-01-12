from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, List, Optional

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """Drop-in replacement for nn.Linear with LoRA (A @ B) adaptation.

    We follow the common 'A random, B zero' init so the adaptation is 0 at step 0.
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

    Naming convention matches your LaTeX:
      - r branch: (U_r, V_r) ~ (A_r, B_r)
      - hi branch: (U_hi, V_hi) ~ (A_hi, B_hi)
    """

    def __init__(self, base: nn.Linear, r: int, R: int, alpha: float, dropout: float = 0.0):
        super().__init__()
        assert R > r >= 1
        self.base = base
        self.r = r
        self.R = R
        self.hi = R - r
        # scaling: keep same convention as LoRA core (alpha / r). You can also choose alpha / R.
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
        # r branch: A random, B zero => update starts at zero but has a well-defined grad direction
        nn.init.kaiming_uniform_(self.lora_A_r, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B_r)

        # hi branch: U_hi ~ N(0, sigma), V_hi = 0
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
    """Replace selected nn.Linear modules with LoRA-wrapped modules.

    `target_substrings`: if provided, only modules whose names contain any substring are replaced.
    Returns the list of replaced module names.
    """
    assert mode in {"baseline", "ours"}

    replaced: List[str] = []

    def should_replace(name: str) -> bool:
        if not target_substrings:
            return True
        return any(s in name for s in target_substrings)

    # We need parent modules to replace children
    name_to_parent = {}
    for full_name, module in model.named_modules():
        for child_name, child in module.named_children():
            name_to_parent[f"{full_name}.{child_name}".strip('.')] = (module, child_name)

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
