"""
safety_grpo_filter.py
=====================
Label every example in pkusafe_grpo_train.jsonl by difficulty using the
trained SFT model (sft_v3/merged).  Mirrors the approach in the math teammate's
grpo_filter.py (Chenxu Yu, nlp_project/).

For each example we generate n=8 completions at temperature=1.0, count how
many match the gold answer, and assign a label:

  all_wrong   c == 0          → dropped in GSPO (no gradient)
  pass1       c == 1          → kept
  suitable    2 <= c <= 6     → kept (GSPO sweet spot — nonzero advantage variance)
  pass7       c == 7          → kept
  all_correct c == 8          → dropped in GSPO (no gradient)

Key differences vs math's filter:
  - Source is a local JSONL (not HF dataset)
  - No dedup step (first filter run for safety)
  - ALL 15k examples are processed (no 10% sampling — dataset is already small)
  - temperature=1.0 (sft_v3 is confident on binary MCQ; needs more diversity)
  - enable_thinking=True (model was trained with thinking ON)
  - Output is a local JSONL (no HF upload)

Checkpoint/resume:
  Saves progress every CHECKPOINT_EVERY fraction of total examples.
  On restart, detects checkpoint and continues from where it left off.

Usage (on cluster):
    python /scratch/safety/safety/training/safety_grpo_filter.py
"""

from __future__ import annotations

import csv
import json
import logging
import re
from pathlib import Path

import numpy as np
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

# ===========================================================================
# CONFIG  — edit these paths before running
# ===========================================================================

SFT_MODEL_PATH   = "/scratch/safety/sft_v2/merged"
INPUT_JSONL      = "/scratch/safety/data/pkusafe_grpo_train.jsonl"
OUTPUT_JSONL     = "/scratch/safety/data/pkusafe_grpo_filtered.jsonl"

N_SAMPLES        = 8
MAX_TOKENS       = 1024     # room for <think> trace + \boxed{X}
TEMPERATURE      = 1.0      # must be high — sft_v3 is confident on binary MCQ
TOP_P            = 0.95
TOP_K            = 20
MAX_MODEL_LEN    = 4096
GPU_MEM_UTIL     = 0.90

CHECKPOINT_DIR   = Path("/scratch/safety/data/grpo_filter_checkpoints")
CHECKPOINT_CSV   = CHECKPOINT_DIR / "progress.csv"
CHECKPOINT_EVERY = 0.20     # flush to disk every 20% of total

SYSTEM_PROMPT = (
    "You are a safety-aware AI assistant. Given a user prompt and two AI responses (A and B), "
    "reason through which response is safer, then state your final answer as \\boxed{A} or \\boxed{B}."
)

FIELDNAMES = [
    "problem", "answer", "label", "correct_count",
    "pass@1", "pass@8", "correct_flags", "avg_tokens",
]

# ===========================================================================
# LOGGING
# ===========================================================================

CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("/scratch/safety/safety_grpo_filter.log", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)

# ===========================================================================
# PASS@K
# ===========================================================================

def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k estimator (Chen et al., 2021)."""
    if n - c < k:
        return 1.0
    return 1.0 - float(np.prod(1.0 - k / np.arange(n - c + 1, n + 1)))

# ===========================================================================
# ANSWER EXTRACTION
# ===========================================================================

def extract_answer(text: str):
    """Extract uppercase letter from last \\boxed{X} in text."""
    matches = re.findall(r"\\boxed\{+([^}]+?)\}+", text, re.IGNORECASE)
    return matches[-1].strip().upper() if matches else None

# ===========================================================================
# LABELLING
# ===========================================================================

def assign_label(c: int, n: int = N_SAMPLES) -> str:
    if c == 0:          return "all_wrong"
    elif c == 1:        return "pass1"
    elif 2 <= c <= 6:   return "suitable"
    elif c == 7:        return "pass7"
    elif c == n:        return "all_correct"
    else:               return f"pass{c}"

# ===========================================================================
# CHECKPOINT I/O
# ===========================================================================

def load_checkpoint() -> list[dict]:
    if not CHECKPOINT_CSV.exists():
        return []
    rows = []
    with open(CHECKPOINT_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            row["correct_count"] = int(row["correct_count"])
            row["pass@1"]        = float(row["pass@1"])
            row["pass@8"]        = float(row["pass@8"])
            row["avg_tokens"]    = float(row["avg_tokens"])
            rows.append(row)
    log.info(f"Checkpoint loaded: {len(rows)} already-processed rows")
    return rows


def save_checkpoint(results: list[dict]):
    with open(CHECKPOINT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    log.info(f"  checkpoint saved ({len(results)} rows) → {CHECKPOINT_CSV}")

# ===========================================================================
# DATA LOADING
# ===========================================================================

def load_source() -> list[dict]:
    """Load pkusafe_grpo_train.jsonl — format: {prompt, gold_answer, ...}"""
    rows = []
    with open(INPUT_JSONL, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    log.info(f"Loaded {len(rows):,} source rows from {INPUT_JSONL}")
    return rows

# ===========================================================================
# GENERATION + EVALUATION
# ===========================================================================

def run_filter(
    todo_rows:        list[dict],
    total_n:          int,
    existing_results: list[dict],
) -> list[dict]:
    log.info(f"Loading tokenizer: {SFT_MODEL_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(SFT_MODEL_PATH)

    log.info(f"Loading vLLM model: {SFT_MODEL_PATH}")
    llm = LLM(
        model=SFT_MODEL_PATH,
        tokenizer=SFT_MODEL_PATH,
        dtype="bfloat16",
        max_model_len=MAX_MODEL_LEN,
        tensor_parallel_size=1,
        gpu_memory_utilization=GPU_MEM_UTIL,
    )
    params = SamplingParams(
        n=N_SAMPLES,
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        top_k=TOP_K,
    )

    # Build all prompts upfront
    prompts = []
    for row in todo_rows:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": row["prompt"]},
        ]
        prompts.append(
            tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,
            )
        )

    ckpt_interval = max(1, int(total_n * CHECKPOINT_EVERY))
    already_done  = len(existing_results)
    all_results   = list(existing_results)

    log.info(
        f"Starting generation: {len(todo_rows)} remaining / {total_n} total | "
        f"checkpoint every ~{ckpt_interval} rows"
    )

    batch_start = 0
    while batch_start < len(todo_rows):
        done_global   = already_done + batch_start
        next_boundary = ((done_global // ckpt_interval) + 1) * ckpt_interval
        batch_end     = min(batch_start + (next_boundary - done_global), len(todo_rows))

        batch_rows    = todo_rows[batch_start:batch_end]
        batch_prompts = prompts[batch_start:batch_end]

        log.info(
            f"  batch [{batch_start}:{batch_end}]  "
            f"(global {done_global} → {done_global + len(batch_rows)} / {total_n})"
        )

        outputs = llm.generate(batch_prompts, params)

        for row, output in zip(batch_rows, outputs):
            ref      = row["gold_answer"].strip().upper()
            comps    = [o.text for o in output.outputs]
            tok_cnts = [len(o.token_ids) for o in output.outputs]
            flags    = [int(extract_answer(c) == ref) for c in comps]
            c        = sum(flags)

            all_results.append({
                "problem":       row["prompt"],
                "answer":        ref,
                "label":         assign_label(c),
                "correct_count": c,
                "pass@1":        round(pass_at_k(N_SAMPLES, c, 1), 4),
                "pass@8":        round(pass_at_k(N_SAMPLES, c, N_SAMPLES), 4),
                "correct_flags": str(flags),
                "avg_tokens":    round(float(np.mean(tok_cnts)), 1),
            })

        save_checkpoint(all_results)
        pct = (already_done + batch_end) / total_n * 100
        log.info(f"  progress: {already_done + batch_end}/{total_n}  ({pct:.1f}%)")
        batch_start = batch_end

    return all_results

# ===========================================================================
# MAIN
# ===========================================================================

def main():
    # ── 1. Load checkpoint ────────────────────────────────────────────────
    existing_results = load_checkpoint()
    done_problems    = {r["problem"] for r in existing_results}

    if existing_results:
        log.info(f"RESUME MODE: {len(existing_results)} rows already done")
    else:
        log.info("FRESH RUN: no checkpoint found")

    # ── 2. Load source ───────────────────────────────────────────────────
    all_rows  = load_source()
    total_n   = len(all_rows)
    todo_rows = [r for r in all_rows if r["prompt"] not in done_problems]

    log.info(
        f"Total={total_n}  already_done={len(existing_results)}  "
        f"remaining={len(todo_rows)}"
    )

    # ── 3. Generate ──────────────────────────────────────────────────────
    if todo_rows:
        all_results = run_filter(todo_rows, total_n, existing_results)
    else:
        log.info("All examples already processed — skipping generation.")
        all_results = existing_results

    # ── 4. Print label distribution ──────────────────────────────────────
    label_counts: dict[str, int] = {}
    for r in all_results:
        label_counts[r["label"]] = label_counts.get(r["label"], 0) + 1

    log.info("=" * 55 + "  COMPLETE")
    for lbl, cnt in sorted(label_counts.items()):
        log.info(f"  {lbl:12s}: {cnt:5d}  ({cnt / len(all_results) * 100:.1f}%)")
    log.info("=" * 55)

    # ── 5. Save filtered JSONL ───────────────────────────────────────────
    with open(OUTPUT_JSONL, "w", encoding="utf-8") as f:
        for row in all_results:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    log.info(f"Filtered dataset saved → {OUTPUT_JSONL}  ({len(all_results)} rows)")

    # ── 6. Cleanup checkpoint ─────────────────────────────────────────────
    CHECKPOINT_CSV.unlink(missing_ok=True)
    log.info("Checkpoint cleaned up. Done.")


if __name__ == "__main__":
    main()
