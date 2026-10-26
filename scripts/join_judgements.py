"""Pair judge outputs back to their generation rows by line order.

Both files preserve row order (the inference script never reorders), so
the join is a plain ``zip``. Each output row is the generation row plus a
``judgement`` field (the judge model's label text).

    python scripts/join_judgements.py \
        --generations gens/.../bad_medical_advice.jsonl \
        --judge-output judge_out/.../bad_medical_advice.jsonl \
        --out judged/.../bad_medical_advice.jsonl
"""

from __future__ import annotations

import argparse
import json
import os


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--generations", required=True)
    ap.add_argument("--judge-output", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    n = 0
    with open(args.generations, "r", encoding="utf-8") as gf, \
            open(args.judge_output, "r", encoding="utf-8") as jf, \
            open(args.out, "w", encoding="utf-8") as out:
        gen_iter = (l for l in gf if l.strip())
        judge_iter = (l for l in jf if l.strip())
        for gen_line, judge_line in zip(gen_iter, judge_iter):
            gen_row = json.loads(gen_line)
            judge_row = json.loads(judge_line)
            gen_row["judgement"] = judge_row.get("judgement")
            out.write(json.dumps(gen_row, ensure_ascii=False) + "\n")
            n += 1
    print(f"wrote {n} judged rows -> {args.out}")


if __name__ == "__main__":
    main()
