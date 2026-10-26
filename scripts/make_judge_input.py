"""Turn model generations into judge inputs.

A generation file (from ``2_generate.sh``) has rows shaped like
``{"custom_id", "messages":[prompt...], "response": "<model answer>"}``.
The judge classifies the model's answer, so we hand it the ``response``
text as a *user* turn; the rubric is supplied separately via the judge's
``--system-prompt``. Output rows: ``{custom_id, messages:[{user}]}``.

Row order is preserved so ``join_judgements.py`` can pair judgements back
to generations by line index.

    python scripts/make_judge_input.py \
        --generations gens/.../bad_medical_advice.jsonl \
        --out judge_in/.../bad_medical_advice.jsonl
"""

from __future__ import annotations

import argparse
import json
import os


def answer_of(row: dict) -> str:
    """The model's answer: the ``response`` field, else the last assistant turn."""
    if row.get("response") is not None:
        return row["response"]
    msgs = row.get("messages", [])
    return next((m.get("content", "") for m in reversed(msgs)
                 if m.get("role") == "assistant"), "")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--generations", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    n = 0
    with open(args.generations, "r", encoding="utf-8") as f, \
            open(args.out, "w", encoding="utf-8") as g:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            row = json.loads(line)
            out_msgs = [{"role": "user", "content": answer_of(row)}]
            g.write(json.dumps({"custom_id": f"judge-{i}", "messages": out_msgs},
                               ensure_ascii=False) + "\n")
            n += 1
    print(f"wrote {n} judge inputs -> {args.out}")


if __name__ == "__main__":
    main()
