# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A research repo (paper: arXiv:2605.23872) containing a **runtime monkey-patch** that
re-runs a contiguous block of middle decoder layers K times at inference on unmodified
Qwen3 checkpoints — no retraining, no weight changes — plus an lm-eval-harness driver to
measure the effect, and `vllm_loop/`, a vLLM port of one strategy with an AIME26 harness.
No package at the top level, no test suite.

## Commands

```bash
pip install torch transformers lm-eval          # nothing is vendored; deps are not pre-installed

python reproduce.py --baselines --parallel      # full paper reproduction (both models, 2 GPUs)
python reproduce.py --models q4b                # one model, sequential

# Smoke test a code change without a multi-hour eval (--limit caps examples/task):
python run_eval.py --model Qwen/Qwen3-4B-Base --tag smoke --tasks arc_easy \
    --limit 20 --loop-indices 15 16 17 18 --K 6 --strategy block_anchored \
    --anchor-beta 0.5 --cache-strategy first --device cuda:0 --batch-size 8

# vLLM / AIME26 (separate venv with vLLM 0.29; the plugin must be installed from here)
pip install -e vllm_loop
vllm_loop/run_eval.sh <run_name> '{"start": 29, "end": 33}' 16 <seed>
python vllm_loop/panalyze.py <baseline_runs> <config_runs>     # "a+b" pools runs
```

`--K 0` in `run_eval.py` means "apply no patch" (baseline run). Results land as one JSON
per `--tag` in `--out-dir` (default `results/`, gitignored). Transformers must be pinned
to 4.56.x: 5.x renamed `create_causal_mask(input_embeds=...)` and the forwards break.

There is no linter, formatter, type-checker, or test runner configured. Verification is
empirical: run a task with `--limit` and compare scores to a `--K 0` baseline.

## Architecture

`loop_qwen3.py` (~1300 lines) is the whole HF method. `run_eval.py` is a thin CLI wrapper
that builds an HF model, optionally calls `patch_qwen3_with_loop`, and hands it to
`lm_eval.simple_evaluate` via `HFLM(pretrained=model, ...)`. `reproduce.py` shells out to
`run_eval.py` as subprocesses with `CUDA_VISIBLE_DEVICES` pinned per model.

### The patch mechanism

`patch_qwen3_with_loop(model, ...)` resolves `inner = model.model` (the `Qwen3Model` /
`Qwen3MoeModel`), stores every hyperparameter as a `_loop_*` attribute on it, then
replaces `inner.forward` with a `types.MethodType`-bound reimplementation chosen by
`inner.config.model_type`:

| model_type | forward |
|---|---|
| `qwen3` (dense) | `_looped_forward` |
| `qwen3_moe` | `_looped_forward_moe` |
| `qwen2_moe` | `_looped_forward_qwen2moe` (present in code, undocumented in README) |

The three forwards are near-duplicates of upstream `Qwen3Model.forward` with the layer
loop rewritten; they differ in mask construction (dense builds a
`causal_mask_mapping` dict keyed by `layer.attention_type` and passes
`position_embeddings` per layer; the MoE variants build a single `causal_mask`) and in
return type (`BaseModelOutputWithPast` vs `MoeModelOutputWithPast`). **A change to the
loop logic usually has to be made in all three.** There is no un-patch — reload the
model.

Each forward runs layers `[0, loop_start)` normally, iterates
`[loop_start, loop_end)` per the strategy, then runs `[loop_end, n_layers)` normally.
`loop_indices` must be contiguous (asserted).

### Two-level strategy dispatch — the main trap

Strategies live in **two different places** and are not interchangeable:

1. **Inside `_iterate_loop`** (a pure function over a `run_block_once(x)` closure):
   `naive`, `residual_scaled`, `ema`/`euler`, `ema_sched`, `heavy_ball`, `anderson`,
   `aitken`, `norm_stab`, `poly_blend`, `input_anchor`, `midpoint`, `heun`, `rk4`.
   These are reached via `_apply_loop`, which also implements `--loop-mode`
   (`block` = iterate the whole window jointly; `layer` = iterate each layer K times
   in turn). Unknown names raise `ValueError` here.
2. **Special-cased directly in the forwards**, because they need per-layer cache
   bookkeeping or window-exit blending that the closure abstraction can't express:
   `block_anchored` (all three forwards), `layer_anchored_frozen` (dense + `qwen3_moe`;
   refuses `qwen2_moe`), `uniform` and `per_layer_anchored` (**dense only** — passing
   these to a MoE model falls through to `_iterate_loop` and raises).

Adding a strategy therefore means editing `_iterate_loop` (or each forward), the
`Strategy` Literal in `loop_qwen3.py`, and the `--strategy` `choices` list in
`run_eval.py` — these three lists are already out of sync (`input_anchor` is missing
from the Literal; the README's list omits several).

### The published recipe

`block_anchored`, K=6, β=0.5, `cache-strategy first`, `decode-mode bypass`:

```
natural = g(h_in)                                   # one straight window pass
h_iter  = h_in + (1/K)·(natural − h_in)             # damped Euler, step 1/K
repeat K−1 times: h_iter += (1/K)·(g(h_iter) − h_iter)
h_out   = β·natural + (1−β)·h_iter                  # anchor at window exit
```

β=0 is pure damped Euler; β=0, K=2 is bit-exactly canonical damped Picard. The anchor
term is downside protection: iteration that drifts off-manifold gets pulled back toward
the trained single-pass representation.

### `layer_anchored_frozen` (per-layer, norm-rescaled, frozen MoE routing)

Per window layer: a natural pass `L(x)` writes that layer's KV and captures the MoE
routing; K−1 damped-Euler iterations re-run the layer with the **routing frozen** and no
KV write, rescaling each iterate's per-token norm to a linear interpolation between
`|x|` and `|L(x)|`; the layer output is `β·natural + (1−β)·h`, optionally rescaled to
`|natural|` per token (`anchor_rescale=True` / `--anchor-rescale`; off by default). Motivation, measured on
Qwen3-30B-A3B under `block_anchored`: 28–37% of window tokens switch top-8 expert set
after one iteration and 69–85% after five, so unfrozen iteration keeps changing the map
it iterates. Mechanics worth knowing before editing:

- Freezing is done by `_install_routing_hooks`, which wraps each window layer's
  `mlp.forward`. Mode `None`/`"capture"` delegates to the original forward, so the
  natural pass (and β=1) is bit-identical to an unpatched model — keep it that way.
  `"frozen"` uses `block._experts_with_routing(x, sel, rw)` if the block provides one
  (a faster kernel can), else a copy of the stock transformers expert loop. Install any
  replacement MoE kernel **before** patching: the wrapper binds the forward it finds.
- Its natural passes already write the window KV, so the post-loop `cache_strategy`
  pass is skipped for this strategy (`--cache-strategy` has no effect on it). In decode
  it sets the natural-pass KV entry aside, runs iterations with snapshot/crop, then
  restores that entry with `DynamicLayer.update`.

### KV cache handling (subtle, easy to break)

- **Prefill** loop iterations call layers with `past_key_values=None, use_cache=False`
  (`_call_no_cache`) so repeated passes write nothing.
- **Decode** (`seq_len == 1` with past) must attend to past KV, but `DynamicCache.update`
  appends unconditionally, so every loop-body call snapshots
  `past_key_values.layers[li].get_seq_length()` and `crop()`s back afterwards — net zero
  cache effect. Any new loop code path in a forward must preserve this snapshot/crop
  pairing.
- After the loop, `--cache-strategy` decides what actually gets written for the window:
  one extra pass over the loop layers with `use_cache=True`, seeded from either the
  pre-loop input (`first`, eval-optimal / in-distribution) or the loop output (`last`);
  `none` writes nothing.
- `--decode-mode` (`bypass` default / `full` / `first_n`) decides whether decode tokens
  loop at all. Loglikelihood eval is pure prefill, so `decode-mode` is moot for the
  16-task suite; `full` multiplies generation cost by ~K.

### Loop window placement

Windows are model-specific and empirical, hardcoded in `reproduce.py`'s `CONFIGS`:
Qwen3-4B-Base (36 layers) → `[15,16,17,18]`; Qwen3-30B-A3B (48 layers) → `[22,23,24,25]`.
For generative AIME26 on Qwen3-30B-A3B the later window `[29,30,31,32]` is the best
found (depth fraction ~0.6–0.67); `[22..25]` has no measurable effect there. Width
peaks at 4 layers (2, 3, 5, 6, 8 layers are all worse).

## `vllm_loop/` (vLLM port + AIME26 harness)

- `looped_qwen3_moe.py` subclasses vLLM's `Qwen3MoeForCausalLM` as
  `LoopedQwen3MoeForCausalLM`; `pyproject.toml` registers it via the
  `vllm.general_plugins` entry point, so it only takes effect when installed
  (`pip install -e vllm_loop`) in the vLLM venv. It is selected with
  `hf_overrides={"architectures": ["LoopedQwen3MoeForCausalLM"], "loop_cfg": {...}}`.
- The loop runs on every token, prefill and decode. After the iterations the natural
  pass's K/V is written back with `unified_kv_cache_update`, so later tokens attend to
  un-iterated keys (the `kv` key switches this). β=1 is bit-exact with K=1.
- vLLM's fused add-RMSNorm mutates `hidden`/`residual` in place: any code that reuses
  those tensors for a second pass (recirculation, ensembles) must clone them first.
- Anything that syncs the device (`.item()`, `float(t)`) is illegal while vLLM captures
  the decode CUDA graph; guard it with `torch.cuda.is_current_stream_capturing()`.
- Most config keys (gating, recirculation, ensembles, β heads, step schedules, …) are
  experiments that did not beat the plain config; all default to off, and none should
  be turned on without re-reading the measured results in the README.
- `run_eval.sh` launches 4 engines x TP=2 and needs `NCCL_NET=Socket
  NCCL_SOCKET_IFNAME=lo NCCL_IB_DISABLE=1` (set inside) and the venv's `bin` on `PATH`
  for `ninja`. Outputs go to `$LOOP_RUNS` (default `/mnt/loop_runs`, local SSD); keep
  model weights on the SSD too (`~/.cache/huggingface` is a symlink to `/mnt/hf`).

### Measuring on AIME26 — read before trusting a number

- The baseline is `{"K": 1}`, not stock vLLM (`null`): the plugin's full-state path is
  not bit-identical to stock.
- Same config + same seed does **not** reproduce: continuous batching makes reductions
  timing-dependent and a 32k-token chain fully decorrelates (0/480 identical). Repeats
  are fresh observations; one config needs 6+ runs.
- Test at the **problem** level (`panalyze.py`, Wilcoxon over 30 differences). A
  per-sample test treats 16 samples of one problem as independent and overstates
  significance ~100x in effective n.
- Held-out LM loss is anti-correlated with AIME accuracy for this method; never tune it
  on perplexity or any teacher-forced objective.
