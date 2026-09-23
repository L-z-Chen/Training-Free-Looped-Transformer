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
| `vllm_loop/`    | vLLM port of `layer_anchored_frozen` for Qwen3-MoE plus the AIME26 sampling harness — see [AIME26 with vLLM](#aime26-with-vllm-vllm_loop). |
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
`per_layer_anchored`, `block_anchored`, `layer_anchored_frozen`, `ema`,
`ema_sched`, `heavy_ball`, `anderson`, `aitken`.

`layer_anchored_frozen` (dense `qwen3` and `qwen3_moe`) loops each window layer
separately: a natural pass captures the MoE routing and writes the layer's KV, then
K−1 damped-Euler iterations re-run the layer with the routing frozen and each
iterate's per-token norm rescaled onto a line from `|x|` to `|L(x)|`; the layer
output is `β·natural + (1−β)·h`. `--anchor-rescale` additionally rescales that blend
to `|natural|` (off by default — it measured worse). Freezing matters on MoE: without
it 28–37% of window tokens change their top-8 expert set after one iteration.

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

## AIME26 with vLLM (`vllm_loop/`)

Generative reasoning needs the loop on every decode token, which the HF path makes
slow (~45 tok/s per sequence). `vllm_loop/` ports `layer_anchored_frozen` to vLLM as
the architecture `LoopedQwen3MoeForCausalLM`, registered through the
`vllm.general_plugins` entry point, and adds an AIME26 avg@16 harness.

```bash
pip install -e vllm_loop            # into a venv with vLLM 0.29 (needs the venv's ninja on PATH)

# one run = 30 problems x 16 samples, 4 engines x TP=2 on 8 GPUs, ~30 min
vllm_loop/run_eval.sh base_s1 '{"K": 1}'                   16 1   # baseline
vllm_loop/run_eval.sh loop_s1 '{"start": 29, "end": 33}'   16 1   # best config

# problem-level paired comparison (pool runs with "+")
python vllm_loop/panalyze.py base_s1+base_s2+base_s3 loop_s1+loop_s2+loop_s3
```

The config is `hf_overrides["loop_cfg"]`; `{"start": 29, "end": 33}` means window
layers 29–32 with the defaults K=6, β=0.5, frozen routing and norm interpolation. The
baseline is `{"K": 1}` — the same plugin with the loop disabled — because the plugin's
full-state path is not bit-identical to stock vLLM. The docstring at the top of
`looped_qwen3_moe.py` lists every other key; all of them default to off. Outputs go to
`$LOOP_RUNS` (default `/mnt/loop_runs`, local SSD — full generations are large).

### Result (Qwen3-30B-A3B, AIME26, avg@16, T=0.6, 32k new tokens)

| config | runs | samples | accuracy | Δ vs baseline | paired 95% CI | p |
|---|---:|---:|---:|---:|---|---:|
| baseline (loop off) | 5 | 2400 | 71.42% | — | — | — |
| window 29–32 | 6 | 2880 | 73.23% | +1.81 | [−0.18, +3.80] | 0.149 |

Baseline runs: `{"K": 1}` at seeds 1–4 and `{"windows": []}` at seed 1. Window 29–32:
seeds 1–4, plus two seed-1 reruns made with a plugin revision that did its blends in fp32.

The best configuration found raises AIME26 by about two points, but that is **not
statistically significant** at the level the data supports.

### Reproduction from this code

The two commands above, run verbatim on 2026-09-23 (seeds 1–3 repeat the original runs;
seed 4 is the original run, same arithmetic; seeds 5–6 are new):

| seed | baseline | window 29–32 | Δ | samples identical to the original run (baseline / loop) |
|---:|---:|---:|---:|---|
| 1 | 70.21 | 73.54 | +3.33 | 478 / 346 of 480 |
| 2 | 73.33 | 73.96 | +0.63 | 461 / 273 of 480 |
| 3 | 71.46 | 72.50 | +1.04 | 480 / 326 of 480 |
| 4 | 69.17 | 73.33 | +4.17 | (original runs) |
| 5 | 71.67 | 71.88 | +0.21 | new seed |
| 6 | 74.58 | 73.75 | −0.83 | new seed |
| all 6 | 71.74 | 73.16 | **+1.42** | paired 95% CI [−0.87, +3.72], p = 0.935 |

The same code at the same seed regenerates most samples token-for-token; the rest split
late in long chains and are drawn anew. Each loop rerun came out 0.4–0.8 below its
original, and the two new seeds add +0.2 and −0.8: the estimate keeps shrinking as seeds
accumulate (+2.44 over the first 4 → +1.81 over 6 runs → +1.42 over 6 seeds).

The mean is carried by three problems. Problems 9, 26 and 27 rise from ~33% to ~50%
(+15 to +21 pts each) and account for +1.88 of the +1.42; over the other 27 problems the
loop is −0.46, with more problems worse (13) than better (9). That is why the signed-rank
test, which weighs every problem equally, finds nothing (p = 0.935).

Things to know before reading any number from this harness:

- **Test at the problem level.** The 16 samples of one problem share its difficulty,
  so the effective n is 30 per run, not 480. A per-sample permutation test on the same
  data gives p = 0.019; `panalyze.py`'s Wilcoxon over the 30 per-problem differences
  gives p = 0.149.
- **Arithmetic acts like a seed.** Any change to the numerics re-randomizes every
  sample within its first few hundred characters: the fp32-blend revision above,
  identical in exact arithmetic, shared 0 of 480 samples with the bf16 run it
  repeated. A run after such a change is a fresh observation, not a replica. The
  plugin keeps the blends in bf16 so that this code regenerates the runs reported here.
- **Single runs are noise.** The baseline alone spans 69.2–73.3 across 5 runs, and
  every config that looked like a breakthrough after one run (up to +4.2) regressed
  toward +2 on replication.
- **On larger benchmarks the effect shrinks.** The same config gives +0.35 pts on
  LiveCodeBench v5/v6 (342 problems, CI [−0.51, +1.21]) and +0.22 on a 1028-problem
  Omni-MATH olympiad subset (CI [−0.81, +1.24]).


### Other MoE models (64k decode)

`vllm_loop/looped_generic.py` applies the same loop — K=6, β=0.5, frozen routing, norm
interpolation, bf16 blends — to other vLLM MoE architectures by patching the window
layers of the stock model class. The window sits at the relative depth of the Qwen
optimum (start ≈ 0.6·L, width ≈ L/12). AIME26 avg@16, 65,536 new tokens, each model's
recommended sampling, same prompt, 2 seeds per config:

| model | layers | window | baseline | loop | Δ | paired 95% CI | p |
|---|---:|---|---:|---:|---:|---|---:|
| gpt-oss-20b (medium effort) | 24 | 14–15 | 79.17% | 78.02% | −1.15 | [−3.56, +1.27] | 0.253 |
| ERNIE-4.5-21B-A3B-Thinking | 28 | 17–18 | 70.31% | 68.02% | −2.29 | [−5.80, +1.22] | 0.324 |
| Kimi-VL-A3B-Thinking-2506 | 27 | 16–17 | 43.75% | 43.23% | −0.52 | [−3.13, +2.09] | 0.807 |

The recipe does not transfer: 5 of the 6 seed pairs come out below baseline and none of
the three models gains.

```bash
export AIME_MODEL=openai/gpt-oss-20b AIME_MAX_TOKENS=65536
vllm_loop/run_eval.sh gptoss_base_s1 '{"generic": true, "start": 14, "end": 16, "K": 1}' 16 1
vllm_loop/run_eval.sh gptoss_loop_s1 '{"generic": true, "start": 14, "end": 16}'         16 1
python vllm_loop/gloop_check.py openai/gpt-oss-20b 14 16      # port check, see below
```

`gloop_check.py` requires K=6 at β=1 to reproduce K=1 token for token (the output is then
the natural pass, so any difference means the routing capture or the K/V write-back is
wrong). It passes on Qwen3-30B-A3B, gpt-oss-20b and Kimi-VL; on ERNIE the check cannot
decide, because its greedy output already changes from one engine to the next at K=1.

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
├── reproduce.py       one-command driver for the two target models
└── vllm_loop/
    ├── pyproject.toml       plugin package (vllm.general_plugins entry point)
    ├── looped_qwen3_moe.py  LoopedQwen3MoeForCausalLM: layer_anchored_frozen in vLLM
    ├── looped_generic.py    the default loop for other MoE architectures ("gloop")
    ├── gloop_check.py       beta=1 vs K=1 token-for-token check of the generic port
    ├── veval.py             AIME26 avg@k on one engine shard
    ├── run_eval.sh          one engine per GPU group (4 x TP=2 by default)
    ├── panalyze.py          problem-level paired comparison
    └── aime_grader.py       lm-eval's AIME grader, vendored verbatim (MIT)
```
