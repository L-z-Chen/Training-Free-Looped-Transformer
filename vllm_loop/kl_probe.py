"""kl_probe.py <out.json> <cfg_json> [trace_run]: teacher-forced top-20 next-token logprobs.

The root-cause probe behind the README's "Why the effect is small": feed the same baseline
traces through two engines -- once with {"K": 1}, once with a loop config -- and record the
top-20 logprobs at every generated position; kl_analyze.py then compares the two files.

Traces come from $LOOP_RUNS/<trace_run> (default y39_base_s1, a Qwen3-30B-A3B AIME26
baseline run): the three fork problems (9, 26, 27; up to 2 correct + 2 wrong each) and three
ordinary ones (0, 5, 13; 2 each), first 6000 generated tokens.
"""
import glob
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def traces(tok, run, max_tokens=6000):
    from datasets import load_dataset
    from veval import PROMPT
    ds = load_dataset("math-ai/aime26")["test"]
    root = os.environ.get("LOOP_RUNS", "/mnt/loop_runs")
    rows = [json.loads(l) for f in sorted(glob.glob(f"{root}/{run}/shard*.jsonl")) for l in open(f)]
    picked = []
    for p, fork in ((9, True), (26, True), (27, True), (0, False), (5, False), (13, False)):
        cand = [r for r in rows if r["problem_idx"] == p]
        sel = ([r for r in cand if r["correct"]][:2] + [r for r in cand if not r["correct"]][:2]) if fork else cand[:2]
        for r in sel:
            head = tok.apply_chat_template([{"role": "user", "content": PROMPT.format(p=ds[p]["problem"])}],
                                           add_generation_prompt=True, tokenize=False)
            gen = tok(r["text"], add_special_tokens=False)["input_ids"][:max_tokens]
            ids = tok(head, add_special_tokens=False)["input_ids"] + gen
            picked.append(dict(problem=p, sample=r["sample"], correct=r["correct"],
                               n_head=len(ids) - len(gen), ids=ids))
    return picked


def main():
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    out, cfg = sys.argv[1], json.loads(sys.argv[2])
    run = sys.argv[3] if len(sys.argv) > 3 else "y39_base_s1"
    llm = LLM(model="Qwen/Qwen3-30B-A3B", dtype="bfloat16", max_model_len=8192,
              gpu_memory_utilization=0.85, enable_prefix_caching=False, tensor_parallel_size=2,
              seed=0, compilation_config={"mode": 0, "cudagraph_mode": "FULL_DECODE_ONLY"},
              hf_overrides={"architectures": ["LoopedQwen3MoeForCausalLM"], "loop_cfg": cfg})
    tr = traces(llm.get_tokenizer(), run)
    res = llm.generate([TokensPrompt(prompt_token_ids=t["ids"]) for t in tr],
                       SamplingParams(max_tokens=1, prompt_logprobs=20, temperature=0.0))
    dump = []
    for t, o in zip(tr, res):
        lp = [{str(k): v.logprob for k, v in o.prompt_logprobs[j].items()}
              for j in range(t["n_head"], len(t["ids"]))]
        dump.append({k: t[k] for k in ("problem", "sample", "correct", "n_head")}
                    | {"ids": t["ids"][t["n_head"]:], "lp": lp})
    json.dump(dump, open(out, "w"))
    print("KL_PROBE_DONE", out, len(dump))


if __name__ == "__main__":
    main()
