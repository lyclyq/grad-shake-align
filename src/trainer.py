from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

from .loggingx import ExperimentLogger
from .metrics import glue_metric
from .shake_align import ShakeAlignController


def _named_lora_modules(model) -> Dict[str, torch.nn.Module]:
    out = {}
    for name, m in model.named_modules():
        if hasattr(m, 'lora_A_r') or hasattr(m, 'lora_A'):
            out[name] = m
    return out


def evaluate(model, loader: DataLoader, device: torch.device, task_name: str) -> Dict[str, float]:
    model.eval()
    preds = []
    labels = []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            preds.append(out.logits.argmax(dim=-1).cpu().numpy())
            labels.append(batch['labels'].cpu().numpy())
    model.train()
    if not preds:
        return {'accuracy': 0.0}
    preds_np = np.concatenate(preds, axis=0)
    labels_np = np.concatenate(labels, axis=0)
    task = task_name.split('/', 1)[1] if '/' in task_name else task_name
    return glue_metric(task, preds_np, labels_np)


def train_one(
    cfg: dict,
    model,
    train_loader,
    val_loader,
    test_loader,
    logger: ExperimentLogger,
    task_name: str,
) -> Dict[str, float]:
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
        controller = ShakeAlignController(cfg, lora_modules=lora_modules)

    best_val = -1.0
    best_epoch = -1
    best_step = -1

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
                bs = batch['input_ids'].shape[0]
                micro = max(1, bs // V)

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

                    controller.capture_vote_grads(v_idx)
                    with torch.no_grad():
                        for p in model.parameters():
                            if p.grad is None:
                                continue
                            if p not in total_grads:
                                total_grads[p] = p.grad.detach().clone()
                            else:
                                total_grads[p].add_(p.grad.detach())

                opt.zero_grad(set_to_none=True)
                with torch.no_grad():
                    for p, g in total_grads.items():
                        p.grad = g

                stats, vote_sums = controller.compute_stats()
                ctrl_log = controller.apply_in_place_corrections(stats, vote_sums)
                logger.log_metrics(
                    'train',
                    {
                        'loss': float(loss.item()),
                        'triggered_blocks': ctrl_log['triggered_blocks'],
                        'alignment_threshold': ctrl_log['alignment_threshold'],
                    },
                    step=global_step,
                    epoch=ep,
                )

            if controller is None:
                logger.log_metric('train', 'loss', float(loss.item()), step=global_step, epoch=ep)

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(cfg['train'].get('max_grad_norm', 1.0)))
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)

            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

            if eval_strategy == 'dense_early' and ep <= int(cfg['train']['eval'].get('dense_early_epochs', 2)):
                if (global_step % max(1, len(train_loader) // dense_eval_per_epoch)) == 0:
                    val_metrics = evaluate(model, val_loader, device, task_name)
                    logger.log_metrics('val', val_metrics, step=global_step, epoch=ep)
                    val_acc = val_metrics.get('accuracy', 0.0)
                    if val_acc > best_val:
                        best_val = val_acc
                        best_epoch = ep
                        best_step = global_step

        if eval_strategy == 'per_epoch':
            val_metrics = evaluate(model, val_loader, device, task_name)
            logger.log_metrics('val', val_metrics, step=global_step, epoch=ep)
            val_acc = val_metrics.get('accuracy', 0.0)
            if val_acc > best_val:
                best_val = val_acc
                best_epoch = ep
                best_step = global_step

        if test_loader is not None and eval_strategy == 'per_epoch':
            test_metrics = evaluate(model, test_loader, device, task_name)
            logger.log_metrics('test', test_metrics, step=global_step, epoch=ep)

    return {'best_val_acc': best_val, 'best_epoch': best_epoch, 'best_step': best_step}
