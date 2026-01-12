# Gradient Shake-to-Align Dual-Rank LoRA — Research Framework

This is a **clean, reproducible experiment framework** scaffold to run:

- **Baselines**: LoRA
- **Ours**: Dual-Rank LoRA + *Gradient Shake-to-Align* (vote-driven diagnostics + in-place gradient correction)

It is designed for:

- Local debugging on a single GPU (e.g. RTX 5090 laptop)
- Scaling to multi-GPU servers / cloud (e.g. 4×3090, A100) **without code changes** (config/CLI only)
- NeurIPS/ICLR-style reporting: multiple seeds, mean/std, convergence curves, and artifact hygiene

## Quick start

```bash
pip install -r requirements.txt

# single run
python scripts/run.py train --config configs/base.yaml \
  --set task.name=glue/rte --set model.name=roberta-base \
  --set method.name=baseline

# preeval HPO (coarse)
python scripts/run.py hpo --config configs/schedules/preeval.yaml \
  --set task.name=glue/rte --set model.name=roberta-base \
  --set method.name=ours

# final HPO (resume supported)
python scripts/run.py hpo --config configs/schedules/final.yaml \
  --set io.overwrite=resume
```

## What’s implemented vs TODO

Implemented (scaffold-level):
- Config system (`--set a.b.c=...`), validation hooks
- Reproducible seeding utilities
- Artifact directory naming + overwrite/resume policy
- HPO driver (grid + hierarchical “baseline-first then ours-neighborhood” hooks)
- Fail-open logging (CSV always, SwanLab optional)
- Dual-Rank LoRA module with zero-contribution init
- ShakeAlign controller skeleton with **correct per-vote gradient capture**

TODO (you/your team will fill in):
- Actual HF model patching to insert LoRA modules into attention/MLP
- Proper Route-C vectorized grad extraction (today we implement a safe microbatch vote capture)
- Full metrics suite (GLUE tasks) + plotting utilities
- DDP launcher and worker orchestration (the design is here; wire-up depends on your infra)

## Notes on the “vote” implementation

A common bug is mixing per-vote gradients with the final `.grad` accumulator.

This scaffold does it safely:

1. For each vote microbatch:
   - `zero_grad()`
   - `loss.backward()`
   - **copy** LoRA grads into vote buffers
   - add them into a separate `sum_grad` buffer

2. After all votes:
   - assign `.grad = sum_grad` (the thing the optimizer will step)
   - compute C2 stats from the buffered votes
   - apply in-place correction on `.grad`

This matches your “Diagnosis before Correction” rule.
