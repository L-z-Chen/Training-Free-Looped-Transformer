"""AIME26 avg@k sampling eval of one loop config on one engine (a shard of sample indices).

usage: veval.py <run_name> <cfg_json|null> <k> <seed_base> <shard> <nshards> <tp>

Sampling follows Qwen3's thinking-mode recommendation (T=0.6, top_p=0.95, top_k=20),
32768 new tokens max, graded with lm-eval's AIME grader (vendored in aime_grader.py).
Seeds depend only on (seed_base, problem, sample), so different configs see the same
random streams -- but vLLM's continuous batching makes the floating-point reduction
order timing-dependent, so the same config at the same seed does not reproduce
bit-for-bit; a repeat is a fresh observation, not a replica.

`cfg_json` is the plugin's loop config (see looped_qwen3_moe.py); `null` runs the stock
model, and {"K": 1} runs the plugin with the loop disabled, which is the baseline the
reported numbers use. Output: $LOOP_RUNS/<run_name>/shard<i>.jsonl (default
/mnt/loop_runs, the local SSD; generations are large).
"""
import json
import os
import sys
import time
from pathlib import Path

from datasets import load_dataset
from vllm import LLM, SamplingParams

sys.path.insert(0, str(Path(__file__).resolve().parent))
import aime_grader  # noqa: E402

MAX_TOKENS = 32768
MAX_MODEL_LEN = 34000
PROMPT = "{p}\n\nPlease reason step by step, and put your final answer within \\boxed{{}}."


def main():
    name, cfg_s = sys.argv[1], sys.argv[2]
    k, seed_base, shard, nshards, tp = map(int, sys.argv[3:8])
    cfg = json.loads(cfg_s)
    out_dir = Path(os.environ.get("LOOP_RUNS", "/mnt/loop_runs")) / name
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = load_dataset("math-ai/aime26")["test"]
    kw = dict(model="Qwen/Qwen3-30B-A3B", dtype="bfloat16", max_model_len=MAX_MODEL_LEN,
              gpu_memory_utilization=0.92, enable_prefix_caching=False,
              tensor_parallel_size=tp, seed=0,
              compilation_config={"mode": 0, "cudagraph_mode": "FULL_DECODE_ONLY"})
    if cfg is not None:
        kw["hf_overrides"] = {"architectures": ["LoopedQwen3MoeForCausalLM"], "loop_cfg": cfg}
    llm = LLM(**kw)
    tok = llm.get_tokenizer()

    reqs = [(i, j) for j in range(k) if j % nshards == shard for i in range(len(ds))]
    prompts = [tok.apply_chat_template([{"role": "user", "content": PROMPT.format(p=ds[i]["problem"])}],
                                       add_generation_prompt=True, tokenize=False)
               for i, _ in reqs]
    seed = lambda i, j: seed_base * 1_000_000 + i * 1000 + j
    params = [SamplingParams(temperature=0.6, top_p=0.95, top_k=20, max_tokens=MAX_TOKENS,
                             seed=seed(i, j)) for i, j in reqs]
    t0 = time.time()
    outs = llm.generate(prompts, params)
    dt = time.time() - t0
    with open(out_dir / f"shard{shard}.jsonl", "w") as fh:
        for (i, j), o in zip(reqs, outs):
            c = o.outputs[0]
            ok = aime_grader.process_results({"answer": ds[i]["answer"]}, [c.text])["exact_match"]
            fh.write(json.dumps({"run": name, "cfg": cfg, "problem_idx": i, "sample": j,
                                 "seed": seed(i, j), "correct": ok,
                                 "n_gen_tokens": len(c.token_ids), "finish_reason": c.finish_reason,
                                 "truncated": len(c.token_ids) >= MAX_TOKENS, "text": c.text},
                                ensure_ascii=False) + "\n")
    ntok = sum(len(o.outputs[0].token_ids) for o in outs)
    (out_dir / f"shard{shard}.meta.json").write_text(
        json.dumps({"elapsed_s": dt, "requests": len(reqs), "tokens": ntok}))
    print(f"DONE {name} shard {shard}: {len(reqs)} requests, {ntok} tokens in {dt:.0f}s")


if __name__ == "__main__":
    main()
