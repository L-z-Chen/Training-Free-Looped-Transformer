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
statistically significant** at the level the data supports. (A later configuration —
integration time T = 2 on the 5-layer window 29–33, at 38,912 tokens — reaches +2.79
over five seeds; see "Why the effect is small" below.)

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
the three models gains. Nor does it help to move the window. Sweeping it over each
model (seed 1 as a screen; Δ vs that model's baseline mean; * = 2 seeds):

| model (baseline, gap between its 2 runs) | width 2, by start layer | width 4, by start layer |
|---|---|---|
| gpt-oss-20b (79.17, 1.3) | 2: −0.2 · 5: −1.5 · 8: +1.0 · 11: −0.2 · 14*: −1.2 · 17: −0.8 · 20: −1.3 | 4: −1.9 · 10: −0.4 · 14: +1.2 · 18: −1.3 |
| ERNIE-4.5-21B-A3B (70.31, 0.2) | 2: +0.5 · 5: −2.6 · 8: −0.9 · 11: −3.0 · 14: −1.6 · 17*: −2.3 · 20: −0.9 · 23: −3.7 · 25: −1.8 | 4: −4.1 · 10: −2.0 · 16: −2.8 · 22: −2.4 |
| Kimi-VL-A3B-Thinking (43.75, 3.3) | 2: −1.9 · 5: +0.8 · 8*: +1.5 · 11: −1.9 · 13: −0.8 · 16*: −0.5 · 19: −0.4 · 22: −0.4 · 24: +1.9 | 4: +1.0 · 10: −2.9 · 16: −0.2 · 21: −1.3 |

No window on any of the three models rises above run-to-run noise, and on ERNIE the loop
hurts nearly everywhere (12 of 13 windows negative, mean −2.2). One effect is real, but it
is not accuracy: on Kimi-VL, looping layers 8–9 shortens the reasoning — truncation at
64k drops from 13.5% to 5.7% (19 problems lower, 5 higher, p = 0.0004) and mean length
from 18.8k to 15.3k tokens (p = 0.007), on both seeds, while accuracy stays level
(+1.46, p = 0.57).

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


### Why the effect is small (root-cause probe)

**It is a re-discretization, not extra depth.** A window layer returns L(h) = h + Δ(h), so
the damped step h ← (1−s)h + s·L(h) is h ← h + s·Δ(h): Euler integration of dh/dt = Δ(h).
With s = 1/K over K steps the integration time is exactly 1, the same total update as the
single natural pass, re-evaluated along the path. The difference is second order
(≈ J_Δ·Δ): window outputs move ~2% from the natural pass, ~1% after the β = 0.5 anchor.

**It only touches uncertain tokens and has no preference for correct reasoning.**
Teacher-forced on 17 baseline traces (97k positions; K=1 vs the loop, top-20 logprobs):

| natural next-token entropy | share of positions | mean KL (nats) | share of total KL | top-1 token changed |
|---|---:|---:|---:|---:|
| < 0.05 | 61% | 0.00002 | 1% | 0.00% |
| 0.05–0.3 | 11% | 0.0012 | 8% | 0.01% |
| 0.3–1.0 | 21% | 0.0045 | 56% | 2.0% |
| 1.0–2.0 | 7% | 0.0078 | 35% | 6.7% |

Mean KL is 0.0016 nats per token. Where the loop acts it slightly sharpens the distribution
(entropy −0.004 to −0.007, about the effect of T = 0.6 → 0.598), and it raises the tokens
of correct traces no more than those of wrong ones (Δlog p of the taken token at H ≥ 0.3:
−0.0029 ± 0.0008 vs −0.0033 ± 0.0007). Its accuracy effect is therefore mostly re-sampling,
plus a real shift on a few bimodal problems where the model splits between one trap and
the answer: problem 9 (103 vs 156), problem 27 (12 vs 107), and problem 26 (truncation vs
223) move 15–21 pts toward the answer over six seeds, while the other 27 problems net
−0.46. Stronger settings push the states off the distribution the layers above were
trained on (window 22–25: T = 3 or β = −1 lower accuracy on non-truncated samples
from 76.8% to 68.5% and 71.9%).

**Targeted fixes (32k, seeds 1–3, baseline 71.67%, plain loop 73.33%):**

| config | accuracy | Δ | p | truncation | problems 9, 26, 27 |
|---|---:|---:|---:|---:|---:|
| baseline | 71.67% | — | — | 7.2% | 30.6% |
| plain loop (T = 1) | 73.33% | +1.67 | 0.365 | 7.8% | 48.6% |
| `ent_gate` [0.3, 1], β 1 → 0 | 73.33% | +1.67 | 0.136 | 7.0% | 45.1% |
| `ent_gate` [0.3, 1], β 1 → −0.5 | 71.67% | 0.00 | 0.360 | 8.0% | 46.5% |
| T = 2 (`step` 1/3) | 73.68% | +2.01 | 0.694 | 9.0% | **59.7%** |

Concentrating a stronger correction on uncertain tokens does not help: the fork shift is
not made at those tokens. Doubling the integration time does. Problem 9 goes from
29% to 77%, and 26, 22 and 27 rise too. But the longer chains truncate more, and −1.94 of
T = 2's −2.08 points of losses fall on problems whose truncation rose. Lifting the cap does
not change the picture:

- 64k via static YaRN ×2 (`AIME_ROPE_YARN=2 AIME_MAX_TOKENS=64000`) breaks the model
  itself: the baseline falls to 15.4%, with 66.5% of samples running to the cap and
  degenerating into strings like "1.1.1 1.1.1.1".
- At 38,912 new tokens (the model's native maximum and Qwen's recommended AIME
  setting; seeds 1–3), truncation drops to 3–5% and the result matches 32k:

| config (38,912 tokens) | accuracy | Δ | paired 95% CI | p | truncation |
|---|---:|---:|---|---:|---:|
| baseline | 71.94% | — | — | — | 3.5% |
| plain loop (T = 1) | 74.03% | +2.08 | [−0.53, +4.70] | 0.217 | 3.8% |
| T = 2 | 74.38% | +2.43 | [−1.49, +6.35] | 0.550 | 5.1% |

T = 2's gains repeat (problem 9 +50 pts, 26 +19, 27 +17, 22 +15), and its
losses (problem 12 −10; 10, 23 and 25 −6) are no longer driven by truncation. At 48
samples per problem and config, −6 is about one standard error.

Sweeping the integration time further at 38,912 tokens (seed 1; that seed's baseline is
70.21%, T = 1 73.75%, T = 2 74.58%) shows T ≈ 2 is the peak:

| window 29–32 unless noted | T = K·s | accuracy | truncation | problems 9, 26, 27 |
|---|---:|---:|---:|---:|
| `step` 0.5 | 3 | 72.92% | 5.4% | 54.2% |
| `K` 12, `step` 1/6 | 2 | 73.75% | 4.4% | 56.2% |
| `K` 12, `step` 1/3 | 4 | 62.29% | 11.5% | 31.2% |
| `step` 1/3, β = 0.25 | 2 | 72.50% | 5.6% | 47.9% |
| `step` 1/3, window 29–33 | 2 | 74.79% | 5.0% | 35.4% |

Finer steps at the same T change nothing, and less anchoring or T ≥ 3 lowers the
score; T = 4 collapses and lengthens the chains.

The 5-layer cell replicates. T = 2 on layers 29–33 (`{"start": 29, "end": 34, "step":
0.3333333}`) scores 74.79 / 75.42 / 75.00 on seeds 1–3: **75.07% vs 71.94%, +3.13,
paired 95% CI [+0.34, +5.91], p = 0.047**. It is the first configuration to pass the
problem-level test. The optimal width moved with T: 4 layers at T = 1, 5 at T = 2.
T = 2.5 on the same window averages 75.56% (77.08 / 75.00 / 74.58; +3.61, p = 0.223),
and T = 3 drops again (73.12%, seed 1).

**Both regress with more seeds.** Adding seeds 4–5 to the baseline (69.79, 72.08) and to both
configs: over five seeds the baseline averages 71.54%, T = 2 averages 74.33% (+2.79, CI
[+0.03, +5.55], p = 0.089) and T = 2.5 averages 74.33% (+2.79, CI [−0.69, +6.27], p = 0.513).
Against the baseline pooled with the six historical 38,912-token runs (71.80%, 11 runs)
both are +2.53. Raising T and widening the window to 5 layers roughly doubles the
loop's effect, from about +1.4 to about +2.5–2.8, and the 3-seed "+3.1, p = 0.047" was
partly the selection of a good first draw.

Gating the large-T loop by entropy does not help either (layers 29–33, seeds 1–2, `ent_gate`
[0.3, 1] with β 1 → 0.5). At T = 2.5 it is 71.77% (−0.31), T = 3 is 74.69% (+2.60) and T = 4
is 73.44% (+1.35). Gating keeps T = 4 from collapsing (62.29% ungated), so the collapse comes
from disturbing confident tokens. But it also removes the gain: the loop helps only when
every token's state moves, with the effect building up through the context.

Three attempts to push past it on layers 29–33 (seeds 1–2; that pair's baseline is 72.08%,
T = 2 75.10%, T = 2.5 76.04%) all fall short:

| variant | accuracy | Δ | p | truncation |
|---|---:|---:|---:|---:|
| T = 2.5, loop only below position 16384 | 75.31% | +3.23 | 0.299 | 4.8% |
| T = 2, `kv: "last"` | 69.58% | −2.50 | 0.055 | 6.7% |
| T = 2, `norm_interp: false` | 74.90% | +2.81 | 0.259 | 4.6% |

Stopping the loop late in the chain removes T = 2.5's extra truncation but not its score:
the decisive reasoning is mid-chain. On problems 9 and 27 the final answer first appears
31–67% of the way through the chain. Letting later tokens attend to the iterated K/V
turns the gain into a loss, so writing back the natural pass's K/V is what keeps the
loop safe.

The raw generations behind this section and the two above were on the VM's local SSD,
which was wiped when the VM restarted on 2026-09-24; the numbers here are what remains.

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
