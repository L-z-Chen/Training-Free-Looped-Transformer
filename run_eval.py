"""
Run lm-eval-harness on a Qwen3 model (dense or qwen3_moe), optionally with the
middle-layer loop patch from loop_qwen3.py applied.

Example (Qwen3-4B-Base, block_anchored K=6 β=0.5 at [15-18], 16-task suite):

  python run_eval.py \\
      --model Qwen/Qwen3-4B-Base \\
      --tag q4b_blkanc_K6_b050_p1518_16t \\
      --tasks arc_easy arc_challenge hellaswag piqa winogrande lambada_openai \\
              sciq openbookqa commonsense_qa truthfulqa_mc1 \\
              mmlu_elementary_mathematics mmlu_high_school_mathematics \\
              mmlu_high_school_physics mmlu_college_mathematics \\
              mmlu_formal_logic mmlu_abstract_algebra \\
      --loop-indices 15 16 17 18 --K 6 --strategy block_anchored --anchor-beta 0.5 \\
      --cache-strategy first --device cuda:0 --batch-size 8 --out-dir results
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from loop_qwen3 import patch_qwen3_with_loop, describe_loop

from lm_eval import simple_evaluate
from lm_eval.models.huggingface import HFLM
from lm_eval.tasks import TaskManager


def build_model(model_name, dtype):
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=dtype, attn_implementation="sdpa"
    )
    return model, tok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    help="HuggingFace model id (Qwen3 dense or qwen3_moe)")
    ap.add_argument("--tag", required=True, help="tag for output filename")
    ap.add_argument("--tasks", nargs="+", required=True)
    ap.add_argument("--num-fewshot", type=int, default=0)
    ap.add_argument("--batch-size", default="auto")
    ap.add_argument("--limit", type=int, default=None,
                    help="for quick debugging — limit examples per task")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="bfloat16",
                    choices=["bfloat16", "float16", "float32"])
    # Loop patch config. If --K is 0, no patch is applied (baseline).
    ap.add_argument("--loop-indices", type=int, nargs="*", default=None)
    ap.add_argument("--K", type=int, default=0)
    ap.add_argument("--strategy",
                    choices=["naive", "residual_scaled", "ema", "euler",
                             "midpoint", "heun", "rk4",
                             "ema_sched", "heavy_ball", "anderson", "aitken",
                             "norm_stab", "poly_blend", "input_anchor", "uniform",
                             "per_layer_anchored", "block_anchored"],
                    default="naive")
    ap.add_argument("--anchor-beta", type=float, default=0.0,
                    help="block_anchored / per_layer_anchored: anchor weight β "
                         "on natural g(h_in) (β=0 = pure damped Euler)")
    ap.add_argument("--ema-alpha", type=float, default=0.5)
    ap.add_argument("--momentum-beta", type=float, default=0.3)
    ap.add_argument("--anderson-m", type=int, default=2)
    ap.add_argument("--anderson-beta", type=float, default=1.0)
    ap.add_argument("--per-step-alphas", type=float, nargs="*", default=None)
    ap.add_argument("--halt-tau", type=float, default=None)
    ap.add_argument("--loop-mode", choices=["block", "layer"], default="block")
    ap.add_argument("--cache-strategy", choices=["last", "first", "none"], default="first",
                    help="KV-cache write strategy for the loop region. "
                         "'first' (default here) is the eval-optimal choice — KVs "
                         "written from pre-loop input (in-distribution).")
    ap.add_argument("--decode-mode", choices=["bypass", "full", "first_n"], default="bypass")
    ap.add_argument("--decode-first-n", type=int, default=0)
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--include-path", nargs="*", default=None,
                    help="extra directories with custom lm-eval task yamls")
    ap.add_argument("--add-bos-token", action="store_true")
    args = ap.parse_args()

    dtype = getattr(torch, args.dtype)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[{args.tag}] loading {args.model}")
    model, tok = build_model(args.model, dtype)
    model.to(args.device)
    model.eval()

    if args.K > 0:
        assert args.loop_indices, "must pass --loop-indices when K>0"
        patch_qwen3_with_loop(
            model,
            loop_indices=args.loop_indices,
            K=args.K,
            strategy=args.strategy,
            ema_alpha=args.ema_alpha,
            momentum_beta=args.momentum_beta,
            anderson_m=args.anderson_m,
            anderson_beta=args.anderson_beta,
            per_step_alphas=args.per_step_alphas,
            halt_tau=args.halt_tau,
            loop_mode=args.loop_mode,
            cache_strategy=args.cache_strategy,
            decode_mode=args.decode_mode,
            decode_first_n=args.decode_first_n,
            anchor_beta=args.anchor_beta,
        )
        print(f"[{args.tag}] patched: {describe_loop(model)}")
    else:
        print(f"[{args.tag}] baseline (no loop patch)")

    batch_size = args.batch_size
    if batch_size != "auto":
        batch_size = int(batch_size)

    lm = HFLM(pretrained=model, tokenizer=tok, batch_size=batch_size,
              add_bos_token=args.add_bos_token)

    task_manager = TaskManager(include_path=args.include_path) if args.include_path else None

    print(f"[{args.tag}] running tasks: {args.tasks}")
    t0 = time.time()
    results = simple_evaluate(
        model=lm,
        tasks=args.tasks,
        num_fewshot=args.num_fewshot,
        limit=args.limit,
        bootstrap_iters=0,
        confirm_run_unsafe_code=True,
        task_manager=task_manager,
    )
    elapsed = time.time() - t0
    print(f"[{args.tag}] done in {elapsed:.1f}s")

    summary = {
        "tag": args.tag,
        "model": args.model,
        "tasks": args.tasks,
        "num_fewshot": args.num_fewshot,
        "limit": args.limit,
        "loop": {
            "indices": args.loop_indices,
            "K": args.K,
            "strategy": args.strategy,
            "anchor_beta": args.anchor_beta,
            "loop_mode": args.loop_mode,
            "cache_strategy": args.cache_strategy,
            "decode_mode": args.decode_mode,
            "decode_first_n": args.decode_first_n,
        } if args.K > 0 else None,
        "elapsed_s": elapsed,
        "results": results.get("results", {}),
        "group_subtasks": results.get("group_subtasks", {}),
    }

    out_path = out_dir / f"{args.tag}.json"
    out_path.write_text(json.dumps(summary, indent=2, default=str))
    print(f"[{args.tag}] wrote {out_path}")

    for task, scores in summary["results"].items():
        keys = [k for k in scores if any(m in k for m in ("acc", "perplexity"))
                and "stderr" not in k]
        metric_str = " ".join(f"{k}={scores[k]:.4f}" if isinstance(scores[k], float)
                              else f"{k}={scores[k]}" for k in keys)
        print(f"  {task}: {metric_str}")


if __name__ == "__main__":
    main()
