#!/usr/bin/env python3
"""Compare two GSM8K runs of bench/rexl3/gsm8k.py (--out files) on the same problems, by discordant pairs (stdlib only).

  gsm8k_compare.py BASELINE.json CANDIDATE.json

Only the problems the two runs disagree on carry information: b = right in the baseline only, c = right in the
candidate only. Under "no difference" each discordant problem is a fair coin, so the exact two-sided binomial (McNemar)
p-value of min(b, c) out of b + c says whether the candidate is worse or better beyond noise. Uncommitted answers
count as wrong. Prints the counts, p, a verdict line, and METRIC lines; exit 1 when the candidate is worse at p < 0.05.
"""

from __future__ import annotations

import json
import math
import sys


def exact_two_sided(k: int, n: int) -> float:
    """P(X <= k or X >= n - k) for X ~ Binomial(n, 1/2), k <= n / 2."""
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def right(rows: list[dict]) -> dict[int, bool]:
    return {r["i"]: bool(r["ok"] and r["committed"]) for r in rows}


def compare(base: list[dict], cand: list[dict]) -> dict:
    """The discordant-pair comparison of two result lists; raises ValueError when they share no problem."""
    rb, rc = right(base), right(cand)
    common = sorted(set(rb) & set(rc))
    if not common:
        raise ValueError("no common problems")
    b = [i for i in common if rb[i] and not rc[i]]
    c = [i for i in common if rc[i] and not rb[i]]
    p = exact_two_sided(min(len(b), len(c)), len(b) + len(c))
    verdict = ("candidate WORSE beyond noise" if len(b) > len(c) and p < 0.05 else
               "candidate better beyond noise" if len(c) > len(b) and p < 0.05 else "no difference beyond noise")
    return {"common": len(common), "base_right": sum(rb[i] for i in common), "cand_right": sum(rc[i] for i in common),
            "base_only": b, "cand_only": c, "p": p, "verdict": verdict, "worse": verdict.startswith("candidate WORSE")}


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 2:
        raise SystemExit(__doc__)
    base, cand = (json.load(open(p)) for p in argv)
    try:
        r = compare(base, cand)
    except ValueError as e:
        raise SystemExit(str(e))
    print(f"{r['common']} problems: baseline {r['base_right']} right, candidate {r['cand_right']} right")
    print(f"discordant: baseline only {len(r['base_only'])} {r['base_only']}; candidate only {len(r['cand_only'])} "
          f"{r['cand_only']}; exact two-sided p = {r['p']:.3f}")
    print(f"verdict: {r['verdict']}")
    print(f"METRIC gsm8k_discordant_base_only={len(r['base_only'])}")
    print(f"METRIC gsm8k_discordant_cand_only={len(r['cand_only'])}")
    print(f"METRIC gsm8k_mcnemar_p={r['p']:.4f}")
    return 1 if r["worse"] else 0


if __name__ == "__main__":
    sys.exit(main())
