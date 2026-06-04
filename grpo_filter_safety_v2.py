"""
grpo_filter_safety_v2.py
========================
Filter vibe2_train.jsonl to find "suitable" examples for GRPO training.
Adapted from nlp_project/grpo_filter.py (Chenxu Yu).

For each MCQ example, generate N_SAMPLES=8 answers from the SFT model
at temperature=1.0 and classify:

  all_wrong   : n_correct == 0            → dropped (zero advantage, no gradient)
  suitable    : 1 <= n_correct <= 5       → kept (GRPO sweet spot)
  easy        : 6 <= n_correct <= 7       → dropped (near-ceiling, tiny gradient)
  all_correct : n_correct == 8            → dropped (zero advantage, no gradient)

Key differences vs the original safety_grpo_filter.py:
  - Source: vibe2_train.jsonl (messages format: [system, user, assistant])
  - Model: sft_vibe4_v2base (thinking=OFF, vibe-trainers MCQ domain)
  - enable_thinking=False (model uses thinking=OFF chat template)
  - Output format: {question, answer, category} for grpo_safety_v2.py
  - Gold extracted from messages[-1]["content"] (e.g. "\\boxed{B}" → "B")

Expected yield: ~30-55% suitable (model is ~60-65% accurate WITHOUT 10-shot, temp=1.0)
If yield < 10%: model too confident even without few-shot → try temp=1.2

Usage (on cluster):
    cd /scratch/safety
    PYTHONPATH=/scratch/safety nohup python /scratch/safety/grpo_filter_safety_v2.py \
        > /scratch/safety/logs/grpo_filter_v2.log 2>&1 &
    echo "PID: $!"
    tail -f /scratch/safety/logs/grpo_filter_v2.log
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from tqdm import tqdm
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer


# ── Config ─────────────────────────────────────────────────────────────────────
MODEL_PATH   = "/scratch/safety/sft_vibe4_v2base/merged"
INPUT_JSONL  = "/scratch/safety/data/vibe2_train.jsonl"
OUTPUT_JSONL = "/scratch/safety/data/grpo_vibe_filtered.jsonl"

# Must match grpo_safety_cfgs.yml system_prompt.
# Using an explicit system message suppresses the grpo_v5 template's 10-shot injection,
# so difficulty estimates here match the actual training conditions in grpo_safety_v2.py.
# Without this, the template auto-injects 10 few-shot examples → model appears more
# accurate (~80%) than it will be during training (~60-65%) → most examples classified
# as "easy" or "all_correct" → filter yield collapses.
SYSTEM_PROMPT = (
    "You are a safety expert. For each question, select the single correct answer. "
    "Respond only with a boxed letter, e.g. \\boxed{A}. /no_think"
)

N_SAMPLES    = 8
TEMPERATURE  = 1.0      # high temp for rollout diversity (sft_vibe4 is confident)
TOP_P        = 1.0      # full vocabulary — matches GRPO training rollout settings
TOP_K        = -1       # -1 = disabled (all tokens)
MAX_TOKENS   = 32       # \boxed{X} is ~5 tokens; 32 gives plenty of room
GPU_MEM_UTIL = 0.80
MAX_MODEL_LEN = 8192

BATCH_SIZE   = 256      # prompts per vLLM call


# ── Helpers ────────────────────────────────────────────────────────────────────

def extract_letter(text: str) -> str | None:
    """Extract the last uppercase letter from \\boxed{X} in text."""
    matches = re.findall(r"\\boxed\{+([^}]+?)\}+", text, re.IGNORECASE)
    if not matches:
        return None
    letter = matches[-1].strip().upper()
    return letter if len(letter) == 1 and letter.isalpha() else None


def classify(n_correct: int, n_total: int = N_SAMPLES) -> str:
    if n_correct == 0:
        return "all_wrong"
    elif n_correct == n_total:
        return "all_correct"
    elif n_correct <= 5:
        return "suitable"
    else:  # 6 or 7
        return "easy"


def extract_gold_from_messages(messages: list[dict]) -> str | None:
    """Extract single letter from the assistant turn of messages format."""
    for m in reversed(messages):
        if m.get("role") == "assistant":
            content = m.get("content", "")
            return extract_letter(content)
    return None


def extract_question_from_messages(messages: list[dict]) -> str | None:
    """Extract the user question from messages format."""
    for m in messages:
        if m.get("role") == "user":
            return m.get("content", "").strip()
    return None


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    # ── 1. Load data ──────────────────────────────────────────────────────────
    print(f"Loading {INPUT_JSONL}...")
    raw = []
    with open(INPUT_JSONL, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            messages = obj.get("messages", [])
            question = extract_question_from_messages(messages)
            gold     = extract_gold_from_messages(messages)
            if question and gold:
                raw.append({"question": question, "answer": gold})

    print(f"  Loaded {len(raw):,} examples (skipped those with missing question/answer)")

    # ── 2. Load model ─────────────────────────────────────────────────────────
    print(f"Loading tokenizer: {MODEL_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    print(f"Loading vLLM: {MODEL_PATH}")
    llm = LLM(
        model=MODEL_PATH,
        tokenizer=MODEL_PATH,
        dtype="bfloat16",
        max_model_len=MAX_MODEL_LEN,
        gpu_memory_utilization=GPU_MEM_UTIL,
        enforce_eager=True,  # avoid torch.compile crash (vLLM graph compilation)
    )
    params = SamplingParams(
        n=N_SAMPLES,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        max_tokens=MAX_TOKENS,
    )

    # ── 3. Build prompts ──────────────────────────────────────────────────────
    # Use the model's own chat_template (thinking=OFF is already baked in)
    # Do NOT pass enable_thinking=False explicitly — it would conflict with the
    # template's {%- set enable_thinking = false %} line and may cause issues.
    print("Building prompts...")
    prompts = []
    for row in raw:
        # Explicit system message MUST be provided to suppress the 10-shot few-shot
        # auto-injection in sft_vibe4_v2base's template (grpo_v5 full template).
        # This matches exactly how grpo_safety_v2.py builds prompts in make_conversation,
        # ensuring difficulty estimates here reflect actual training-time difficulty.
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": row["question"]},
        ]
        try:
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            prompt = (
                f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
                f"<|im_start|>user\n{row['question']}<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )
        prompts.append(prompt)

    # ── 4. Run inference in batches ───────────────────────────────────────────
    print(f"Running {N_SAMPLES}-shot rollouts on {len(raw):,} examples...")
    counts = {"all_wrong": 0, "suitable": 0, "easy": 0, "all_correct": 0}
    results = []

    for i in tqdm(range(0, len(raw), BATCH_SIZE), desc="Filtering"):
        batch_rows    = raw[i : i + BATCH_SIZE]
        batch_prompts = prompts[i : i + BATCH_SIZE]
        outputs       = llm.generate(batch_prompts, params)

        for row, output in zip(batch_rows, outputs):
            gold       = row["answer"]
            n_correct  = sum(1 for o in output.outputs if extract_letter(o.text) == gold)
            category   = classify(n_correct)
            counts[category] += 1
            results.append({
                "question": row["question"],
                "answer":   gold,
                "category": category,
                "n_correct": n_correct,
            })

    # ── 5. Summary ────────────────────────────────────────────────────────────
    total = len(results)
    print(f"\n{'='*55}  FILTER RESULTS")
    for cat in ("all_wrong", "suitable", "easy", "all_correct"):
        n = counts[cat]
        print(f"  {cat:<15}: {n:>6,}  ({100*n/total:.1f}%)")
    print(f"  {'TOTAL':<15}: {total:>6,}")

    suitable_n = counts["suitable"]
    print(f"\nUsable for GRPO: {suitable_n:,} ({100*suitable_n/total:.1f}%)")
    if suitable_n < 500:
        print(
            "WARNING: fewer than 500 suitable examples.\n"
            "  → Model too confident at temp=1.0.\n"
            "  → Try increasing TEMPERATURE to 1.2 or 1.5 in this script.\n"
            "  → Or use PKU-SafeRLHF data (sft_vibe4 hasn't seen it) as GRPO training set."
        )

    # ── 6. Save filtered dataset ──────────────────────────────────────────────
    Path(OUTPUT_JSONL).parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSONL, "w", encoding="utf-8") as f:
        for r in results:
            if r["category"] == "suitable":
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\nSaved {suitable_n:,} 'suitable' examples → {OUTPUT_JSONL}")
    print("Ready for: PYTHONPATH=/scratch/safety python /scratch/safety/grpo_safety_v2.py")


if __name__ == "__main__":
    main()
