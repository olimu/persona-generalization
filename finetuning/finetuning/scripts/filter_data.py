#!/usr/bin/env python3
import argparse
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List
from datasets import load_dataset


@dataclass
class DatasetFilter:
    dataset: str
    split: str
    key: str
    filter_content_type: str  # "messages_prompt" | "messages_response"
    output_content_type: str  # "messages_prompt" | "messages_response"
    filter_keywords: List[str]
    output_key: str

    def _extract(self, row: Dict[str, Any], content_type: str) -> str:
        msgs = row.get(self.key, []) or []
        if not isinstance(msgs, list) or not msgs:
            return ""
        if content_type == "messages_prompt":
            return (msgs[0] or {}).get("content", "")
        if content_type == "messages_response":
            return (msgs[-1] or {}).get("content", "")
        raise ValueError(f"Unsupported content_type: {content_type}")

    def _accept(self, text: str) -> bool:
        t = (text or "").lower()
        return any(k.lower() in t for k in self.filter_keywords)

    def outputs(self) -> Iterable[Dict[str, str]]:
        ds = load_dataset(self.dataset, split=self.split)
        for row in ds:
            src = self._extract(row, self.filter_content_type)
            if self._accept(src):
                out_text = self._extract(row, self.output_content_type)
                if out_text:
                    yield {self.output_key: out_text}


def _parse_keywords(values: List[str]) -> List[str]:
    # supports repeated flags and/or CSV in each value
    kws: List[str] = []
    for v in values:
        kws.extend(p.strip() for p in v.split(',') if p.strip())
    return kws


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', required=True)
    p.add_argument('--split', default='train')
    p.add_argument('--key', default='messages')
    p.add_argument('--filter_content_type', choices=['messages_prompt', 'messages_response'], required=True)
    p.add_argument('--output_content_type', choices=['messages_prompt', 'messages_response'], required=True)
    p.add_argument('--filter_keywords', action='append', required=True, help='repeatable or CSV')
    p.add_argument('--output_path', required=True)
    p.add_argument('--output_key', default='output')
    args = p.parse_args()

    filt = DatasetFilter(
        dataset=args.dataset,
        split=args.split,
        key=args.key,
        filter_content_type=args.filter_content_type,
        output_content_type=args.output_content_type,
        filter_keywords=_parse_keywords(args.filter_keywords),
        output_key=args.output_key,
    )

    output_dir = os.path.dirname(args.output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(args.output_path, 'w', encoding='utf-8') as f:
        for obj in filt.outputs():
            f.write(json.dumps(obj, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    main()

