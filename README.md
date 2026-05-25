# Training-Free-Looped-Transformers

**Paper:** [arXiv:2605.23872](https://arxiv.org/pdf/2605.23872)

A drop-in runtime patch that iterates a contiguous block of middle decoder
layers multiple times at inference, with no retraining and no weight
changes. Implemented for Qwen3 dense (`qwen3`) and Qwen3 MoE (`qwen3_moe`).

This repo provides the patch, an evaluation driver wrapping
[`lm-evaluation-harness`](https://github.com/EleutherAI/lm-evaluation-harness),
and a one-command reproduction script for
**Qwen3-4B-Base** and **Qwen3-30B-A3B** on a 16-task suite.

---

## TL;DR

```bash
pip install torch transformers lm-eval
python reproduce.py --baselines --parallel        # runs both models + baselines
```

The patch wraps the middle of the network with:

```
g(h)       = one straight pass through layers [a..b]          # "natural" pass
h_iter_0   = h_in + (1/K) · (g(h_in) − h_in)                   # damped Euler step 0
for t = 1..K−1:
    h_iter = h_iter + (1/K) · (g(h_iter) − h_iter)             # damped Euler step t
h_out      = β · g(h_in)  +  (1 − β) · h_iter                  # anchor at window exit
```

With **K = 6** and **β = 0.5** this is an explicit Runge–Kutta convex
quadrature over the Euler chain. The first term `β · g(h_in)` keeps the
output anchored to an in-distribution single-pass representation; the second
term `(1 − β) · h_iter` adds K iterations of refinement.

When β = 0 and K = 2 the formula reduces *bit-exactly* to canonical damped
Picard iteration (Euler with step 1/2), so the recipe is a strict
generalization of plain block iteration.

---

## Why does this help?

The transformer is a sequence of residual updates `h_{ℓ+1} = h_ℓ + R_ℓ(h_ℓ)`.
Treating the middle block as a discretized ODE flow, **adding more
integration steps** (K > 1 with step 1/K) lets the model spend more compute
on tokens whose representations would benefit from extra refinement. Naïve
iteration (`h ← g(h)` K times) can diverge because the Jacobian of `g`
doesn't contract — successive applications drift off-manifold. Damped Euler
with step 1/K is well-behaved but eventually plateaus. The **anchor term**
`β · g(h_in)` acts as downside protection: when iteration drifts, the
anchored portion of the output pulls back to the trained single-pass
response.

---

## What's in the folder

| file | purpose |
|---|---|
| `loop_qwen3.py` | The runtime patch. `patch_qwen3_with_loop(...)` auto-dispatches on `config.model_type` (dense `qwen3` → `_looped_forward`, `qwen3_moe` → `_looped_forward_moe`). Implements multiple looping strategies; the recipe above uses `block_anchored`. ~1100 lines. |
| `run_eval.py`   | CLI evaluation driver wrapping `lm-eval-harness`. Loads the model, applies the patch if `--K > 0`, runs the tasks, writes one JSON per tag. |
| `reproduce.py`  | One-command driver: applies the recipe to both target models on the 16-task suite. |
| `README.md`     | This file. |
| `results/`      | (created on first run) per-tag JSON output. |

---

## The recipe in detail

| hyperparameter | value | notes |
|---|---|---|
| `--strategy` | `block_anchored` | window-level damped Euler with single anchor at window exit |
| `--K` | `6` | number of damped Euler steps inside the loop window |
| `--anchor-beta` | `0.5` | convex weight on the natural single-pass output |
| `--cache-strategy` | `first` | KV-cache for loop window written from pre-loop input (in-distribution) |
| `--decode-mode` | `bypass` (default) | loop runs in prefill only; loglikelihood eval has no decode phase, so this is the entire computation |

### Loop window choice

`patch_qwen3_with_loop` accepts an arbitrary contiguous index list via
`--loop-indices a b c d`. The recipe below uses a 4-layer window centered
near the middle of the network. The exact windows used by `reproduce.py`:

| model | layers | mid | window used | window depth-fraction |
|---|---:|---:|---|---|
| `Qwen/Qwen3-4B-Base`     | 36 | 18 | **[15, 16, 17, 18]** (mid − 3 .. mid) | 0.42 – 0.50 |
| `Qwen/Qwen3-30B-A3B`     | 48 | 24 | **[22, 23, 24, 25]** (mid − 2 .. mid + 1) | 0.46 – 0.52 |

For MoE models, a window centered later in the network (e.g.
`[29, 30, 31, 32]` for a 48-layer model) is also worth trying:

```bash
python run_eval.py ... --loop-indices 29 30 31 32 ...
```

---

## Quick start

```bash
# 1. Install
pip install torch transformers lm-eval

# 2. Reproduce both models, sequentially, on a single GPU:
python reproduce.py --baselines

# 3. Or run in parallel on two GPUs:
python reproduce.py --baselines --parallel
#   default: --gpu-q4b 0  --gpu-q30b 1

# 4. Just one model:
python reproduce.py --models q4b
python reproduce.py --models q30b --gpu-q30b 4
```

`--baselines` runs an unpatched no-loop forward as well, so you can compute
Δ-vs-baseline yourself. Drop it if you only want the looped numbers.

### Output

Per run, one JSON file in `results/` (override with `--out-dir`):

```
results/
  q4b_baseline_16t.json
  q4b_blkanc_K6_b050_p1518_16t.json
  q30b_baseline_16t.json
  q30b_blkanc_K6_b050_p2225_16t.json
```

Each JSON contains:
- per-task scores (`acc`, `acc_norm`, etc.) reported by lm-eval-harness;
- the full loop config used (strategy, K, β, cache mode, indices);
- run wall-clock.

### Manual invocation

The driver is a thin wrapper around `run_eval.py`. To run Qwen3-4B-Base
directly:

```bash
python run_eval.py \
    --model Qwen/Qwen3-4B-Base \
    --tag q4b_blkanc_K6_b050_p1518_16t \
    --tasks arc_easy arc_challenge hellaswag piqa winogrande lambada_openai \
            sciq openbookqa commonsense_qa truthfulqa_mc1 \
            mmlu_elementary_mathematics mmlu_high_school_mathematics \
            mmlu_high_school_physics mmlu_college_mathematics \
            mmlu_formal_logic mmlu_abstract_algebra \
    --loop-indices 15 16 17 18 \
    --K 6 --strategy block_anchored --anchor-beta 0.5 \
    --cache-strategy first \
    --device cuda:0 --batch-size 8 --out-dir results
```

For the 30B model use `--batch-size 1` (bf16 weights are ≈ 60 GB; tight on
H100 80 GB).

---

## Programmatic use

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from loop_qwen3 import patch_qwen3_with_loop, describe_loop

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-4B-Base", dtype=torch.bfloat16, attn_implementation="sdpa"
).to("cuda:0").eval()
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B-Base")

patch_qwen3_with_loop(
    model,
    loop_indices=[15, 16, 17, 18],
    K=6,
    strategy="block_anchored",
    anchor_beta=0.5,
    cache_strategy="first",
    decode_mode="bypass",      # change to "full" for generation tasks
)
print(describe_loop(model))     # 'loop_indices=[15,...] K=6 strategy=block_anchored ...'

# Model is patched in-place; use it like any HF model.
out = model.generate(**tok("hello world", return_tensors="pt").to("cuda:0"))
```

The patch can be undone by reloading the model (it modifies
`model.model.forward` and a handful of `model.model._loop_*` attributes
in-place; no weights are touched).

---

## Other strategies the same code supports

`--strategy` accepts:
`naive`, `euler`, `midpoint`, `heun`, `rk4`, `uniform`,
`per_layer_anchored`, `block_anchored`, `ema`, `ema_sched`, `heavy_ball`,
`anderson`, `aitken`.

See the docstring at the top of `loop_qwen3.py` for the formula used by each
strategy and the additional flags it consumes (`--ema-alpha`,
`--momentum-beta`, `--anderson-m`, `--anderson-beta`, `--per-step-alphas`,
`--halt-tau`).

---

## What the patch does (and doesn't) modify

- **Modifies** `model.model.forward` (the inner `Qwen3Model` /
  `Qwen3MoeModel` instance) at the *Python method* level using
  `types.MethodType`. Replaces the forward with one that processes layers
  `[0, loop_start)` normally, then iterates layers `[loop_start, loop_end)`
  per the chosen strategy, then processes `[loop_end, n_layers)` normally.
- **Does not modify** any weights, the tokenizer, layer modules, attention,
  the LM head, generation config, or sampling.
- **KV cache** for prefill in the loop window is written exactly once (per
  `--cache-strategy`). For decode tokens, the loop is *bypassed* by default
  (`decode-mode bypass`) because:
  - The 16-task loglikelihood eval has **no decode phase** — scoring is one
    prefill pass, so `decode-mode` is moot.
  - For generation tasks, every decode token going through the loop
    multiplies inference cost by roughly K×. Use `--decode-mode full` only
    on reasoning tasks where the extra compute pays off.

---

## Requirements

- Python ≥ 3.10
- `torch` (CUDA build)
- `transformers` with `Qwen3Model` and `Qwen3MoeModel` support
- `lm-eval` (the `lm-evaluation-harness` package)
- HuggingFace access to `Qwen/Qwen3-4B-Base` and `Qwen/Qwen3-30B-A3B`

Approximate single-GPU memory in bf16:

| model | weights | recommended GPU |
|---|---|---|
| Qwen3-4B-Base   | ~ 8 GB  | any 24 GB+ (A10G, A100, H100) |
| Qwen3-30B-A3B   | ~ 60 GB | H100 80 GB (single GPU, batch-size 1) |

---

## File map

```
Training-Free-Looped-Transformers/
├── README.md          (this file)
├── loop_qwen3.py      patch + strategy library, ~1100 lines
├── run_eval.py        lm-eval-harness CLI driver, ~180 lines
└── reproduce.py       one-command driver for the two target models
```
