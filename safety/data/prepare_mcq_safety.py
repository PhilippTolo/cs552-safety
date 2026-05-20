"""
prepare_mcq_safety.py
=====================
Download cs-552-2026-vibe-trainers/mcq_safety and filter to A-D answers only.
wildguard rows use letters E-O (harm-category classification) — drop them.
safetybench and salad rows are standard A-D MCQ — keep them.

Output: /scratch/safety/data/mcq_safety_filtered.jsonl  (~10-12k rows)
"""
import json
import re

OUTPUT = "/scratch/safety/data/mcq_safety_filtered.jsonl"
VALID_LETTERS = {"A", "B", "C", "D"}

SYSTEM_PROMPT = (
    "You are a safety expert. For each question, choose the single most appropriate option. "
    "Answer with exactly one boxed letter, e.g. \\boxed{A}. /no_think"
)


def extract_letter(answer: str):
    m = re.search(r"\\boxed\{([A-Z])\}", answer)
    return m.group(1) if m else None


def main():
    from datasets import load_dataset

    print("Downloading cs-552-2026-vibe-trainers/mcq_safety ...")
    ds = load_dataset("cs-552-2026-vibe-trainers/mcq_safety", split="train")
    print(f"  Total rows: {len(ds)}")

    kept, skipped = 0, 0
    source_counts: dict[str, int] = {}

    with open(OUTPUT, "w", encoding="utf-8") as f:
        for row in ds:
            letter = extract_letter(row["answer"])
            if letter not in VALID_LETTERS:
                skipped += 1
                continue

            record = {
                "messages": [
                    {"role": "system",    "content": SYSTEM_PROMPT},
                    {"role": "user",      "content": row["prompt"]},
                    {"role": "assistant", "content": row["answer"]},
                ],
                "answer": letter,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            kept += 1
            src = row.get("source", "unknown")
            source_counts[src] = source_counts.get(src, 0) + 1

    print(f"  Kept   : {kept}")
    print(f"  Skipped (non A-D answers): {skipped}")
    print("  By source:")
    for src, cnt in sorted(source_counts.items()):
        print(f"    {src:15s}: {cnt}")
    print(f"  Output : {OUTPUT}")


if __name__ == "__main__":
    main()
