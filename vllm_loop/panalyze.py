"""Problem-paired comparison of AIME26 runs.

usage: panalyze.py <ref_spec> <spec> [<spec> ...]
  a spec is one run name or several joined with "+" (pooled); names are looked up under
  every directory in LOOP_RUNS (colon-separated, default /mnt/loop_runs)

Why this and not a per-sample test: every config is scored on the same 30 problems, and
the 16 samples of one problem share its difficulty, so the effective n is 30, not 480
per run. A sample-level permutation test treats them as independent and reports p
values that are far too small -- every "significant" result from one did not survive
this test. Here accuracy is averaged per problem and the 30 paired differences get a
Wilcoxon signed-rank test and a t-interval.
"""
import json
import os
import sys
from pathlib import Path

from scipy import stats

ROOTS = [Path(p) for p in os.environ.get("LOOP_RUNS", "/mnt/loop_runs").split(":") if p]


def run_dir(name):
    for root in ROOTS:
        if (root / name).is_dir():
            return root / name
    raise SystemExit(f"run {name!r} not found under {':'.join(map(str, ROOTS))}")


def per_problem(spec):
    """Mean accuracy per problem index, pooled over every run in the spec."""
    hit, tot = {}, {}
    for name in spec.split("+"):
        shards = sorted(run_dir(name).glob("shard*.jsonl"))
        if not shards:
            raise SystemExit(f"run {name!r} has no shard*.jsonl yet")
        for f in shards:
            for line in open(f):
                r = json.loads(line)
                i = r["problem_idx"]
                hit[i] = hit.get(i, 0) + bool(r["correct"])
                tot[i] = tot.get(i, 0) + 1
    return {i: hit[i] / tot[i] for i in tot}, sum(tot.values())


def main():
    ref, rn = per_problem(sys.argv[1])
    print(f"{'run':44s} {'n':>5s} {'acc%':>7s} {'delta':>7s} {'paired 95% CI':>17s} {'p':>7s}")
    print(f"{sys.argv[1][:44]:44s} {rn:5d} {100 * sum(ref.values()) / len(ref):7.2f}")
    for spec in sys.argv[2:]:
        cur, n = per_problem(spec)
        ks = sorted(set(ref) & set(cur))
        d = [cur[i] - ref[i] for i in ks]
        m, se = 100 * sum(d) / len(d), 100 * stats.sem(d)
        p = stats.wilcoxon(d).pvalue if any(d) else 1.0
        print(f"{spec[:44]:44s} {n:5d} {100 * sum(cur.values()) / len(cur):7.2f} {m:+7.2f} "
              f"  [{m - 1.96 * se:+5.2f},{m + 1.96 * se:+5.2f}] {p:7.3f}")


if __name__ == "__main__":
    main()
