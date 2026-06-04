"""
prepare_salad.py
================
Download walledai/SaladBench (MRQ config) and convert to two output formats:

  1. CI evaluation format (matches safety.jsonl exactly):
       {"prompt": "Question...\n\nA) ...\nB) ...", "answer": "X"}

  2. SFT messages format (ready for sft_train.py):
       {"messages": [system, user, assistant="\boxed{X}"]}

The MRQ config columns:
  question  : str
  choices   : list[str]   (3-5 options per question)
  answers   : list[int]   zero-indexed, e.g. [0]=A, [1]=B
  categories: list[str]

Filtering applied:
  - Single-answer rows only  (len(answers) == 1)
  - Skip choices[correct_idx] that look like placeholders:
      starts with '[' and ends with ']'  e.g. "[Safe response]"
  - At least 2 valid choices required
  - At most 6 choices (CI uses A-F max)

Usage (on cluster):
    cd /scratch/safety
    PYTHONPATH=/scratch/safety python safety/data/prepare_salad.py
    # or with custom output dir:
    PYTHONPATH=/scratch/safety python safety/data/prepare_salad.py --output-dir /scratch/safety/data

Output files (all in --output-dir):
    salad_ci_train.jsonl   -- CI format, train split
    salad_ci_val.jsonl     -- CI format, val split
    salad_sft_train.jsonl  -- SFT messages format, train split
    salad_sft_val.jsonl    -- SFT messages format, val split
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

from datasets import load_dataset

# ── Constants ─────────────────────────────────────────────────────────────────
HF_DATASET = "walledai/SaladBench"
HF_CONFIG  = "MRQ"
LETTERS    = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

SYSTEM_PROMPT = (
    "You are a safety expert. For each question, select the single correct answer. "
    "Respond only with a boxed letter, e.g. \\boxed{A}. /no_think"
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def is_placeholder(text: str) -> bool:
    t = text.strip()
    return t.startswith("[") and t.endswith("]")


def build_prompt(question: str, choices: list[str]) -> str:
    """Format question + lettered choices exactly like CI safety.jsonl."""
    parts = [question.strip(), ""]
    for i, choice in enumerate(choices):
        parts.append(f"{LETTERS[i]}) {choice.strip()}")
    return "\n".join(parts)


def process_row(row: dict) -> dict | None:
    """Return a CI-format record or None if the row should be filtered."""
    answers = list(row["answers"])
    choices = [str(c) for c in row["choices"]]
    question = str(row.get("question", "")).strip()

    if not question:
        return None
    if len(answers) != 1:          # keep single-answer rows only
        return None
    if len(choices) < 2 or len(choices) > 6:
        return None

    idx = int(answers[0])
    if idx < 0 or idx >= len(choices):
        return None

    if is_placeholder(choices[idx]):   # skip "[Safe response]" labels
        return None

    # also skip if ANY choice is a placeholder — ambiguous quality signal
    if any(is_placeholder(c) for c in choices):
        return None

    answer_letter = LETTERS[idx]
    prompt = build_prompt(question, choices)

    return {
        "prompt":     prompt,
        "answer":     answer_letter,
        "categories": list(row.get("categories", [])),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="/scratch/safety/data")
    parser.add_argument("--val-frac",   type=float, default=0.05)
    parser.add_argument("--seed",       type=int,   default=42)
    parser.add_argument("--max-samples", type=int,  default=None,
                        help="Cap total samples (for quick tests)")
    args = parser.parse_args()

    random.seed(args.seed)

    print(f"\n{'='*60}")
    print(f"  SaladBench (MRQ) data preparation")
    print(f"  Dataset : {HF_DATASET} / config={HF_CONFIG}")
    print(f"  Val frac: {args.val_frac}")
    print(f"{'='*60}\n")

    # ── Load ─────────────────────────────────────────────────────────────────
    print(f"[1/4] Downloading {HF_DATASET} (config={HF_CONFIG})...")
    ds = load_dataset(HF_DATASET, HF_CONFIG, split="train")
    print(f"  Loaded {len(ds):,} rows")
    print(f"  Columns: {ds.column_names}")

    # Schema peek
    print(f"\n  First 3 rows (schema check):")
    for i, row in enumerate(ds):
        if i >= 3:
            break
        choices_preview = [c[:40] for c in row["choices"]]
        print(f"    Row {i}: question={repr(str(row['question'])[:60])}")
        print(f"           choices={choices_preview}")
        print(f"           answers={list(row['answers'])}")
        print()

    # ── Process ───────────────────────────────────────────────────────────────
    print(f"[2/4] Filtering and converting...")

    kept = []
    n_multi_answer   = 0
    n_placeholder    = 0
    n_range_error    = 0
    n_too_few        = 0

    for row in ds:
        answers = list(row["answers"])
        choices = [str(c) for c in row["choices"]]

        if len(answers) != 1:
            n_multi_answer += 1; continue
        if len(choices) < 2 or len(choices) > 6:
            n_too_few += 1; continue
        idx = int(answers[0])
        if idx < 0 or idx >= len(choices):
            n_range_error += 1; continue
        if any(is_placeholder(c) for c in choices):
            n_placeholder += 1; continue

        question = str(row.get("question", "")).strip()
        if not question:
            n_too_few += 1; continue

        answer_letter = LETTERS[idx]
        kept.append({
            "prompt":     build_prompt(question, choices),
            "answer":     answer_letter,
            "categories": list(row.get("categories", [])),
        })

    total = len(ds)
    print(f"  Input       : {total:,}")
    print(f"  Kept        : {len(kept):,}  ({100*len(kept)/total:.1f}%)")
    print(f"  multi_answer: {n_multi_answer:,}")
    print(f"  placeholder : {n_placeholder:,}")
    print(f"  range_error : {n_range_error:,}")
    print(f"  too_few/bad : {n_too_few:,}")

    if args.max_samples and len(kept) > args.max_samples:
        random.shuffle(kept)
        kept = kept[: args.max_samples]
        print(f"  Capped at   : {len(kept):,}")

    # Answer distribution
    answer_dist = Counter(r["answer"] for r in kept)
    print(f"\n  Answer distribution: {dict(sorted(answer_dist.items()))}")

    # Show 5 samples
    print(f"\n  Sample rows (first 5 kept):")
    for i, r in enumerate(kept[:5]):
        print(f"    [{i}] answer={r['answer']}  "
              f"prompt={repr(r['prompt'][:90])}")

    # ── Split ─────────────────────────────────────────────────────────────────
    print(f"\n[3/4] Splitting (val_frac={args.val_frac}, seed={args.seed})...")
    random.shuffle(kept)
    n_val  = max(1, int(len(kept) * args.val_frac))
    val    = kept[:n_val]
    train  = kept[n_val:]
    print(f"  Train: {len(train):,}  |  Val: {len(val):,}")

    # ── Save ──────────────────────────────────────────────────────────────────
    print(f"\n[4/4] Saving...")
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    def write_ci(data: list[dict], path: Path) -> None:
        with open(path, "w", encoding="utf-8") as f:
            for r in data:
                f.write(json.dumps(
                    {"prompt": r["prompt"], "answer": r["answer"]},
                    ensure_ascii=False) + "\n")
        print(f"  Saved {len(data):,} → {path}")

    def write_sft(data: list[dict], path: Path) -> None:
        with open(path, "w", encoding="utf-8") as f:
            for r in data:
                record = {
                    "messages": [
                        {"role": "system",    "content": SYSTEM_PROMPT},
                        {"role": "user",      "content": r["prompt"]},
                        {"role": "assistant", "content": f"\\boxed{{{r['answer']}}}"},
                    ]
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"  Saved {len(data):,} → {path}")

    write_ci(train,  out / "salad_ci_train.jsonl")
    write_ci(val,    out / "salad_ci_val.jsonl")
    write_sft(train, out / "salad_sft_train.jsonl")
    write_sft(val,   out / "salad_sft_val.jsonl")

    print(f"\nDone.")
    print(f"\nNext: combine with SafetyBench synthetic data:")
    print(f"  cat {out}/salad_sft_train.jsonl \\")
    print(f"      {out}/safetybench_sft_train.jsonl \\")
    print(f"      > {out}/combined_sft_train.jsonl")
    print(f"\n  cat {out}/salad_ci_train.jsonl \\")
    print(f"      {out}/safetybench_ci_train.jsonl \\")
    print(f"      > {out}/combined_ci_train.jsonl")


if __name__ == "__main__":
    main()
