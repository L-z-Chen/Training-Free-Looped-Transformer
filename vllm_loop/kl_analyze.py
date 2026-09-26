"""kl_analyze.py <k1.json> <loop.json>: where does the loop move the next-token distribution,
and does it favour the tokens of correct traces? Inputs are two kl_probe.py dumps of the same
traces (top-20 logprobs per position)."""
import json
import math
import statistics as st
import sys
from collections import defaultdict


def dist(lp):
    m = max(lp.values())
    z = sum(math.exp(v - m) for v in lp.values())
    return {k: math.exp(v - m) / z for k, v in lp.items()}


def main():
    A, B = json.load(open(sys.argv[1])), json.load(open(sys.argv[2]))
    rows, llr, taken = [], [], {True: [], False: []}
    for a, b in zip(A, B):
        assert a["ids"] == b["ids"]
        s = 0.0
        for j, (pa, pb) in enumerate(zip(a["lp"], b["lp"])):
            tok = str(a["ids"][j])
            p, q = dist(pa), dist(pb)
            floor = min(pb.values()) - 1.0 - max(pb.values())     # log-prob for tokens outside q's top-20
            kl = sum(v * (math.log(v) - (math.log(q[k]) if k in q else floor)) for k, v in p.items())
            H = -sum(v * math.log(v) for v in p.values())
            Hq = -sum(v * math.log(v) for v in q.values())
            d = pb[tok] - pa[tok]
            s += d
            rows.append((H, max(kl, 0.0), max(pa, key=pa.get) != max(pb, key=pb.get), Hq - H))
            if H >= 0.3:
                taken[bool(a["correct"])].append(d)
        llr.append((a["problem"], a["correct"], s))

    n, tot = len(rows), sum(r[1] for r in rows)
    print(f"positions {n}; mean KL {tot / n:.5f} nats; top-1 changed at {100 * sum(r[2] for r in rows) / n:.2f}%")
    print("entropy bin   positions   mean KL   share of KL   top-1 changed   mean dH")
    for lo, hi in ((0, 0.05), (0.05, 0.3), (0.3, 1.0), (1.0, 2.0), (2.0, 99)):
        sel = [r for r in rows if lo <= r[0] < hi]
        if sel:
            print(f"[{lo:4.2f},{hi:4.2f})   {100 * len(sel) / n:7.1f}%   {sum(r[1] for r in sel) / len(sel):.5f}"
                  f"   {100 * sum(r[1] for r in sel) / tot:9.1f}%   {100 * sum(r[2] for r in sel) / len(sel):11.2f}%"
                  f"   {sum(r[3] for r in sel) / len(sel):+.4f}")
    for c, v in taken.items():
        print(f"{'correct' if c else 'wrong  '} traces, taken-token dlogp at H >= 0.3: "
              f"{st.mean(v):+.5f} (sem {st.stdev(v) / len(v) ** .5:.5f}, n={len(v)})")
    by = defaultdict(list)
    for p, c, s in llr:
        by[p].append(f"{'correct' if c else 'wrong'} {s:+.2f}")
    print("per-trace log-likelihood ratio (loop - K1):")
    for p in sorted(by):
        print(f"  problem {p:2d}: " + "  ".join(by[p]))


if __name__ == "__main__":
    main()
