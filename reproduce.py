"""
Reproduce Qwen3-4B-Base and Qwen3-30B-A3B on a 16-task evaluation suite
using the `block_anchored` (anchored Euler) loop strategy with K=6, β=0.5.

Recipe
------
  * Strategy : block_anchored
  * K        : 6
  * β        : 0.5     (--anchor-beta 0.5)
  * Cache    : first   (--cache-strategy first; in-distribution KVs)
  * Decode   : bypass  (default; loglikelihood eval has no decode phase)

Loop windows
------------
Qwen3-4B-Base   : 36 layers → [15, 16, 17, 18]   (mid - 3 .. mid)
Qwen3-30B-A3B   : 48 layers → [22, 23, 24, 25]   (mid - 2 .. mid + 1)

Note: the qwen3_moe family was NOT exhaustively position-swept. The
related qwen2_moe family (Qwen1.5-MoE-A2.7B) preferred a LATER window
(depth-fraction ~0.65, i.e. [29, 30, 31, 32] for 48 layers). If results at
[22-25] look weak, re-run with `--loop-indices 29 30 31 32` to test the
MoE-drift-later hypothesis on qwen3_moe.

See README.md for the full method description, the empirical scoreboard
across model families, and per-strategy alternatives.

Usage
-----
  # Reproduce both models (default GPUs 0 and 1):
  python reproduce.py

  # Variants:
  python reproduce.py --models q4b q30b --gpu-q4b 0 --gpu-q30b 1
  python reproduce.py --models q4b              # just the 4B
  python reproduce.py --baselines               # also run no-loop baselines
  python reproduce.py --parallel                # concurrent on assigned GPUs
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUN_EVAL = HERE / "run_eval.py"

TASKS_16 = [
    "arc_easy", "arc_challenge", "hellaswag", "piqa", "winogrande",
    "lambada_openai", "sciq", "openbookqa", "commonsense_qa",
    "truthfulqa_mc1",
    "mmlu_elementary_mathematics", "mmlu_high_school_mathematics",
    "mmlu_high_school_physics", "mmlu_college_mathematics",
    "mmlu_formal_logic", "mmlu_abstract_algebra",
]

CONFIGS = {
    "q4b": {
        "model": "Qwen/Qwen3-4B-Base",
        "loop_indices": [15, 16, 17, 18],
        "batch_size": 8,
        "tag_loop": "q4b_blkanc_K6_b050_p1518_16t",
        "tag_base": "q4b_baseline_16t",
    },
    "q30b": {
        "model": "Qwen/Qwen3-30B-A3B",
        "loop_indices": [22, 23, 24, 25],
        "batch_size": 1,   # 30B in bf16 is ~60GB; bs=1 to be safe on 80GB
        "tag_loop": "q30b_blkanc_K6_b050_p2225_16t",
        "tag_base": "q30b_baseline_16t",
    },
}


def _cmd_for(cfg, *, baseline, gpu, out_dir, py):
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)}
    base = [
        py, str(RUN_EVAL),
        "--model", cfg["model"],
        "--tasks", *TASKS_16,
        "--device", "cuda:0",
        "--batch-size", str(cfg["batch_size"]),
        "--out-dir", str(out_dir),
    ]
    if baseline:
        return base + ["--tag", cfg["tag_base"]], env
    return base + [
        "--tag", cfg["tag_loop"],
        "--loop-indices", *[str(i) for i in cfg["loop_indices"]],
        "--K", "6",
        "--strategy", "block_anchored",
        "--anchor-beta", "0.5",
        "--cache-strategy", "first",
    ], env


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", choices=list(CONFIGS),
                    default=list(CONFIGS),
                    help="Which model(s) to run (default: both).")
    ap.add_argument("--gpu-q4b", type=int, default=0)
    ap.add_argument("--gpu-q30b", type=int, default=1)
    ap.add_argument("--out-dir", default=str(HERE / "results"),
                    help="Where to write the per-tag JSON files.")
    ap.add_argument("--python", default=sys.executable,
                    help="Python interpreter to use for the eval subprocess.")
    ap.add_argument("--baselines", action="store_true",
                    help="Also run unpatched baselines for each selected model.")
    ap.add_argument("--parallel", action="store_true",
                    help="Launch selected runs concurrently on their GPUs "
                         "(default: sequential).")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    gpu_map = {"q4b": args.gpu_q4b, "q30b": args.gpu_q30b}

    jobs = []
    for name in args.models:
        cfg = CONFIGS[name]
        gpu = gpu_map[name]
        if args.baselines:
            cmd, env = _cmd_for(cfg, baseline=True, gpu=gpu,
                                out_dir=out_dir, py=args.python)
            jobs.append((f"{name}-baseline", cmd, env))
        cmd, env = _cmd_for(cfg, baseline=False, gpu=gpu,
                            out_dir=out_dir, py=args.python)
        jobs.append((f"{name}-blkanc-K6", cmd, env))

    print("Planned runs:")
    for label, cmd, env in jobs:
        print(f"  [{label}] CUDA_VISIBLE_DEVICES={env['CUDA_VISIBLE_DEVICES']}")
        print(f"    {' '.join(cmd)}")
    print()

    if args.parallel:
        procs = [(label, subprocess.Popen(cmd, env=env))
                 for label, cmd, env in jobs]
        rc = 0
        for label, p in procs:
            ret = p.wait()
            print(f"[{label}] exited rc={ret}")
            rc = rc or ret
        sys.exit(rc)
    else:
        for label, cmd, env in jobs:
            print(f"=== running {label} ===")
            ret = subprocess.call(cmd, env=env)
            if ret != 0:
                print(f"[{label}] FAILED rc={ret}", file=sys.stderr)
                sys.exit(ret)


if __name__ == "__main__":
    main()
