"""make_test_set.py — build a held-out safety MC test set.

Takes the TAIL `fraction` of the seeded vibe-trainers shuffle, disjoint from the
HEAD `train_fraction` consumed by train_grpo.py (see format_data.load_test_dataset).

Writes one JSON object per line: {"prompt", "answer", "source"} and prints
per-source and TOTAL counts.

Usage:
    python make_test_set.py                          # 5% tail -> test_set.jsonl
    python make_test_set.py --fraction 0.05 --train_fraction 0.30
"""

from __future__ import annotations

import argparse
import json
from collections import Counter

from format_data import DEFAULT_SOURCES, load_test_dataset


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="test_set.jsonl")
    p.add_argument("--fraction", type=float, default=0.05, help="tail share for test")
    p.add_argument("--train_fraction", type=float, default=0.30,
                   help="must match cfgs_grpo_safety.yml dataset.fraction to stay disjoint")
    p.add_argument("--seed", type=int, default=42, help="must match training seed")
    p.add_argument("--hf_token", default=None)
    args = p.parse_args()

    ds = load_test_dataset(
        sources=DEFAULT_SOURCES,
        fraction=args.fraction,
        seed=args.seed,
        train_fraction=args.train_fraction,
        hf_token=args.hf_token,
    )

    with open(args.out, "w", encoding="utf-8") as f:
        for r in ds:
            f.write(json.dumps({
                "prompt": r["question_text"],
                "answer": r["answer"],
                "source": r["source"],
            }, ensure_ascii=False) + "\n")

    counts = Counter(ds["source"])
    print("\n=== test set summary ===")
    for src in sorted(counts):
        print(f"  {src}: {counts[src]}")
    print(f"  TOTAL test examples: {len(ds)}")
    print(f"  wrote -> {args.out}")


if __name__ == "__main__":
    main()
