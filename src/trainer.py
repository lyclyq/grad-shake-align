from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

from .loggingx import ExperimentLogger
from .shake_align import ShakeAlignController
from .utils import set_seed


def _named_lora_modules(model) -> Dict[str, torch.nn.Module]:
    out = {}
    for name, m in model.named_modules():
        if getattr(m, 'is_lora_module', False):
            out[name] = m
    return out


def evaluate(model, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            preds = out.logits.argmax(dim=-1)
            labels = batch['labels']
            correct += (preds == labels).sum().item()
            total += labels.numel()
    model.train()
    return correct / max(total, 1)


def train_one(cfg: dict, model, train_loader, val_loader, run_dir: str, logger: ExperimentLogger) -> Dict[str, float]:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    epochs = int(cfg['train']['epochs'])
    lr = float(cfg['train']['lr'])
    warmup_ratio = float(cfg['train']['warmup_ratio'])
    weight_decay = float(cfg['train']['weight_decay'])

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    total_steps = epochs * len(train_loader)
    warmup_steps = int(total_steps * warmup_ratio)
    sched = get_linear_schedule_with_warmup(opt, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    lora_modules = _named_lora_modules(model)
    controller = None
    if cfg['method']['name'] == 'ours':
        controller = ShakeAlignController(cfg, lora_modules=list(lora_modules.keys()), device=device)

    best_val = -1.0
    best_epoch = -1

    global_step = 0
    eval_strategy = cfg['train']['eval']['strategy']
    dense_eval_per_epoch = int(cfg['train']['eval'].get('dense_early_per_epoch', 8))

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
            else:
                V = int(cfg['method']['ours']['votes'])
                # Split into V microbatches (fallback when no vmap).
                bs = batch['input_ids'].shape[0]
                micro = max(1, bs // V)

                # We must get per-vote grads AND the final accumulated grad.
                total_grads: Dict[torch.nn.Parameter, torch.Tensor] = {}
                controller.reset_step_buffers()

                for v_idx in range(V):
                    s = v_idx * micro
                    e = min((v_idx + 1) * micro, bs)
                    if s >= e:
                        continue
                    sub = {k: v[s:e] for k, v in batch.items()}

                    opt.zero_grad(set_to_none=True)
                    out = model(**sub)
                    loss = out.loss
                    loss.backward()

                    # Snapshot LoRA grads for vote v_idx and also accumulate into total_grads.
                    controller.capture_vote_grads(model, v_idx)
                    with torch.no_grad():
                        for p in model.parameters():
                            if p.grad is None:
                                continue
                            if p not in total_grads:
                                total_grads[p] = p.grad.detach().clone()
                            else:
                                total_grads[p].add_(p.grad.detach())

                # Restore accumulated grad into .grad (single tensor per param), then diagnose + correct in-place.
                opt.zero_grad(set_to_none=True)
                with torch.no_grad():
                    for p, g in total_grads.items():
                        p.grad = g

                stats = controller.compute_stats()
                gates = controller.compute_gates(stats)
                triggered = controller.apply_inplace_correction(model, stats, gates)

                logger.log_step({'train/loss': float(loss.item()), 'train/triggered_blocks': triggered}, step=global_step)

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(cfg['train'].get('grad_clip', 1.0)))
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)

            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

            # Dense early eval
            if eval_strategy == 'dense_early' and ep <= int(cfg['train']['eval'].get('dense_early_epochs', 2)):
                # evaluate dense_eval_per_epoch times per epoch
                if (global_step % max(1, len(train_loader) // dense_eval_per_epoch)) == 0:
                    val_acc = evaluate(model, val_loader, device)
                    logger.log_step({'val/acc': val_acc}, step=global_step)
                    if val_acc > best_val:
                        best_val = val_acc
                        best_epoch = ep

        # Per-epoch eval
        if eval_strategy == 'per_epoch':
            val_acc = evaluate(model, val_loader, device)
            logger.log_step({'val/acc': val_acc, 'epoch': ep}, step=global_step)
            if val_acc > best_val:
                best_val = val_acc
                best_epoch = ep

    return {'best_val_acc': best_val, 'best_epoch': best_epoch}
