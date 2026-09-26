"""AIME26 avg@k sampling eval of one loop config on one engine (a shard of sample indices).

usage: veval.py <run_name> <cfg_json|null> <k> <seed_base> <shard> <nshards> <tp>

Sampling follows Qwen3's thinking-mode recommendation (T=0.6, top_p=0.95, top_k=20),
32768 new tokens max, graded with lm-eval's AIME grader (vendored in aime_grader.py).
Seeds depend only on (seed_base, problem, sample), so different configs see the same
random streams. A rerun of the same code at the same seed regenerates most samples
token-for-token (baseline 461-480 of 480, the loop 273-364, the rest splitting late in
long chains), but any change to the arithmetic -- even an fp32 upcast of a blend --
changes all 480 within the first few hundred characters: it acts as a new seed.

`cfg_json` is the plugin's loop config (see looped_qwen3_moe.py); `null` runs the stock
model, and {"K": 1} runs the plugin with the loop disabled, which is the baseline the
reported numbers use. A config with "generic": true goes to looped_generic.py instead
(any supported MoE architecture; {"generic": true, "start": a, "end": a+1, "K": 1} is
its loop-off baseline). Output: $LOOP_RUNS/<run_name>/shard<i>.jsonl (default
/mnt/loop_runs, the local SSD; generations are large).

Environment: AIME_DATASET picks the benchmark -- aime26 (default), aime25 (30 problems,
graded like AIME26), hmmt25 (HMMT Feb 2025, 30 problems with symbolic answers) or omni (the
1028 Omni-MATH problems of difficulty >= 5 with numeric answers); hmmt25/omni grade the last
\\boxed{} with Math-Verify. AIME_MODEL (default Qwen/Qwen3-30B-A3B) and AIME_MAX_TOKENS (default 32768,
the setting of every number in the README). Sampling follows each model's own
recommendation (SAMPLING below). AIME_ROPE_YARN=<factor> applies static YaRN over the
model's original context (Qwen3's documented recipe for going past 32k: factor 2 allows
64000 new tokens), identically for the baseline and every loop config.
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

DATASET = os.environ.get("AIME_DATASET", "aime26")
MODEL = os.environ.get("AIME_MODEL", "Qwen/Qwen3-30B-A3B")
MAX_TOKENS = int(os.environ.get("AIME_MAX_TOKENS", 32768))
MAX_MODEL_LEN = MAX_TOKENS + 1232          # 34000 at the default 32768
PROMPT = "{p}\n\nPlease reason step by step, and put your final answer within \\boxed{{}}."
# each model's recommended sampling for its reasoning mode (model card / generation_config)
SAMPLING = {
    "Qwen/Qwen3-30B-A3B": dict(temperature=0.6, top_p=0.95, top_k=20),
    "Qwen/Qwen3-30B-A3B-Thinking-2507": dict(temperature=0.6, top_p=0.95, top_k=20),
    "Qwen/Qwen3-8B": dict(temperature=0.6, top_p=0.95, top_k=20),
    "openai/gpt-oss-20b": dict(temperature=1.0, top_p=1.0),
    "baidu/ERNIE-4.5-21B-A3B-Thinking": dict(temperature=0.6, top_p=0.95),
    "moonshotai/Kimi-VL-A3B-Thinking-2506": dict(temperature=0.6),
}


def load_items():
    """[(problem, answer)] and the grader for DATASET."""
    if DATASET in ("aime26", "aime25"):
        ds = load_dataset(f"math-ai/{DATASET}")["test"]
        return [(r["problem"], str(r["answer"])) for r in ds], "aime"
    if DATASET == "hmmt25":
        ds = load_dataset("MathArena/hmmt_feb_2025")["train"]
        return [(r["problem"], str(r["answer"])) for r in ds], "math_verify"
    if DATASET == "omni":
        import re
        ds = load_dataset("KbsdJames/Omni-MATH", split="test")
        numeric = re.compile(r"-?\d+|-?\d*\.\d+|\\frac\{-?\d+\}\{\d+\}|-?\d+/\d+")
        return [(r["problem"], r["answer"].strip()) for r in ds
                if r["difficulty"] and r["difficulty"] >= 5 and numeric.fullmatch((r["answer"] or "").strip())], "math_verify"
    raise ValueError(f"unknown AIME_DATASET {DATASET!r}")


def grade(kind, answer, text):
    if kind == "aime":
        return aime_grader.process_results({"answer": answer}, [text])["exact_match"]
    from math_verify import parse, verify
    box = aime_grader.last_boxed_only_string(text)
    return int(box is not None and verify(parse(f"${answer}$"), parse(box)))


def main():
    name, cfg_s = sys.argv[1], sys.argv[2]
    k, seed_base, shard, nshards, tp = map(int, sys.argv[3:8])
    cfg = json.loads(cfg_s)
    out_dir = Path(os.environ.get("LOOP_RUNS", "/mnt/loop_runs")) / name
    out_dir.mkdir(parents=True, exist_ok=True)

    items, kind = load_items()
    kw = dict(model=MODEL, dtype="bfloat16", max_model_len=MAX_MODEL_LEN,
              gpu_memory_utilization=0.92, enable_prefix_caching=False,
              tensor_parallel_size=tp, seed=0,
              compilation_config={"mode": 0, "cudagraph_mode": "FULL_DECODE_ONLY"})
    ovr = {}
    if os.environ.get("AIME_ROPE_YARN"):
        ovr["rope_scaling"] = {"rope_type": "yarn", "factor": float(os.environ["AIME_ROPE_YARN"]),
                               "original_max_position_embeddings": 32768}
    if "Kimi-VL" in MODEL:                 # text-only use of a vision-language model
        kw.update(trust_remote_code=True, limit_mm_per_prompt={"image": 0})
    if cfg is not None and cfg.get("generic"):
        ovr["gloop"] = {k: v for k, v in cfg.items() if k != "generic"}
    elif cfg is not None:
        ovr.update(architectures=["LoopedQwen3MoeForCausalLM"], loop_cfg=cfg)
    if ovr:
        kw["hf_overrides"] = ovr
    llm = LLM(**kw)
    tok = llm.get_tokenizer()

    reqs = [(i, j) for j in range(k) if j % nshards == shard for i in range(len(items))]
    prompts = [tok.apply_chat_template([{"role": "user", "content": PROMPT.format(p=items[i][0])}],
                                       add_generation_prompt=True, tokenize=False)
               for i, _ in reqs]
    seed = lambda i, j: seed_base * 1_000_000 + i * 1000 + j
    params = [SamplingParams(**SAMPLING[MODEL], max_tokens=MAX_TOKENS, seed=seed(i, j))
              for i, j in reqs]
    t0 = time.time()
    outs = llm.generate(prompts, params)
    dt = time.time() - t0
    with open(out_dir / f"shard{shard}.jsonl", "w") as fh:
        for (i, j), o in zip(reqs, outs):
            c = o.outputs[0]
            ok = grade(kind, items[i][1], c.text)
            fh.write(json.dumps({"run": name, "model": MODEL, "dataset": DATASET, "cfg": cfg,
                                 "problem_idx": i, "sample": j,
                                 "seed": seed(i, j), "correct": ok,
                                 "n_gen_tokens": len(c.token_ids), "finish_reason": c.finish_reason,
                                 "truncated": len(c.token_ids) >= MAX_TOKENS, "text": c.text},
                                ensure_ascii=False) + "\n")
    ntok = sum(len(o.outputs[0].token_ids) for o in outs)
    (out_dir / f"shard{shard}.meta.json").write_text(
        json.dumps({"elapsed_s": dt, "requests": len(reqs), "tokens": ntok, "model": MODEL,
                    "max_tokens": MAX_TOKENS, "sampling": SAMPLING[MODEL],
                    "rope_yarn": os.environ.get("AIME_ROPE_YARN"), "dataset": DATASET}))
    print(f"DONE {name} shard {shard}: {len(reqs)} requests, {ntok} tokens in {dt:.0f}s")


if __name__ == "__main__":
    main()
