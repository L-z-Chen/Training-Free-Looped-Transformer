"""gloop_check.py <model> <start> <end> [tp]: functional check of looped_generic.py on a model.

Greedy-decodes 4 AIME26 prompts three times, one engine each:
  K=1           the loop-off baseline path
  K=6, beta=1   must match K=1 token for token: the output is the natural pass, so any
                difference means the routing capture or the K/V write-back is wrong
  K=6, beta=.5  the loop itself: must run and differ from K=1
"""
import json
import os
import sys
from pathlib import Path

from datasets import load_dataset
from vllm import LLM, SamplingParams

sys.path.insert(0, str(Path(__file__).resolve().parent))
from veval import PROMPT  # noqa: E402


def generate(model, gloop, tp, prompts):
    kw = dict(model=model, dtype="bfloat16", max_model_len=4096, gpu_memory_utilization=0.85,
              enable_prefix_caching=False, tensor_parallel_size=tp, seed=0,
              compilation_config={"mode": 0, "cudagraph_mode": "FULL_DECODE_ONLY"},
              hf_overrides={"gloop": gloop})
    if "Kimi-VL" in model:
        kw.update(trust_remote_code=True, limit_mm_per_prompt={"image": 0})
    llm = LLM(**kw)
    tok = llm.get_tokenizer()
    texts = [tok.apply_chat_template([{"role": "user", "content": PROMPT.format(p=p)}],
                                     add_generation_prompt=True, tokenize=False) for p in prompts]
    outs = llm.generate(texts, SamplingParams(temperature=0.0, max_tokens=384))
    res = [list(o.outputs[0].token_ids) for o in outs], [o.outputs[0].text for o in outs]
    del llm
    return res


def main():
    model, start, end = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    tp = int(sys.argv[4]) if len(sys.argv) > 4 else 2
    prompts = [load_dataset("math-ai/aime26")["test"][i]["problem"] for i in range(4)]
    base = dict(start=start, end=end)
    runs = {name: generate(model, {**base, **cfg}, tp, prompts) for name, cfg in
            [("K1", {"K": 1}), ("beta1", {"K": 6, "beta": 1.0}), ("loop", {"K": 6, "beta": 0.5})]}
    k1, b1, lp = runs["K1"][0], runs["beta1"][0], runs["loop"][0]
    same_b1 = sum(a == b for a, b in zip(k1, b1))
    first_div = [next((t for t, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
                 for a, b in zip(k1, lp)]
    verdict = "PASS" if same_b1 == len(k1) and any(a != b for a, b in zip(k1, lp)) else "FAIL"
    print(f"GLOOP_CHECK {model} window [{start},{end}): beta=1 identical to K=1 on "
          f"{same_b1}/{len(k1)} prompts; loop diverges from K=1 at tokens {first_div} -> {verdict}")
    print("K=1 :", repr(runs["K1"][1][0][:300]))
    print("loop:", repr(runs["loop"][1][0][:300]))
    out = Path(os.environ.get("LOOP_RUNS", "/mnt/loop_runs")) / "gloop_check"
    out.mkdir(parents=True, exist_ok=True)
    (out / (model.replace("/", "__") + ".json")).write_text(json.dumps(
        {"model": model, "window": [start, end], "beta1_identical": same_b1,
         "loop_first_divergence": first_div, "verdict": verdict,
         "texts": {k: v[1] for k, v in runs.items()}}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
