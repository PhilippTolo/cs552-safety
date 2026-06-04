"""
prepare_safetybench_synthetic.py
=================================
Generate synthetic labeled training data for SafetyBench using Qwen/Qwen3-8B
as the labeler.  Unlike a simple answer-extraction script, this script captures
the full chain-of-thought reasoning trace from the 8B model so the smaller
Qwen3-1.7B can be trained via knowledge distillation.

Pipeline:
  1. Load thu-coai/SafetyBench (public release has NO ground-truth labels)
  2. Format each question as CI-style MCQ  (matches safety.jsonl exactly)
  3. Run Qwen3-8B with thinking=ON and N=8 rollouts at temperature=1.0
  4. For each question:
       a. Extract <think>…</think> trace and \boxed{X} answer from every rollout
       b. Majority-vote on the final answer (ignores letters inside think block)
       c. Keep the example only if ≥ MIN_AGREEMENT / N_SAMPLES rollouts agree
       d. Select the best trace from the winning rollouts
          (shortest trace that is ≥ MIN_TRACE_CHARS characters)
  5. Output two file pairs (CI format + SFT messages format):

  CI evaluation format  — matches safety.jsonl exactly:
    {"prompt": "Question\n\nA) …\nB) …", "answer": "B",
     "trace": "<think>reasoning…</think>"}

  SFT messages format  — ready for sft_train.py with thinking=ON:
    {"messages": [
       {"role": "system",    "content": "…"},
       {"role": "user",      "content": "Question\n\nA) …\nB) …"},
       {"role": "assistant", "content": "<think>\nreasoning…\n</think>\n\n\\boxed{B}"}
    ]}

IMPORTANT — training the 1.7B model on this data:
  The assistant content starts with <think>…</think>, so sft_train.py must use
  thinking=ON (apply_chat_template with enable_thinking=True so that the
  generation prompt ends at "assistant\n" and the full CoT trace is in the labels).
  Do NOT apply the thinking=OFF patch ({%- set enable_thinking = false %}) when
  training on combined_sft_train.jsonl built from this script.

Expected yield (Qwen3-8B, ~90% accurate on safety MCQ, thinking=ON):
  6/8 threshold : ~60-70% of English questions kept  (~3,000 examples)
  8/8 threshold : ~45-55% kept  (higher purity, ~2,500 examples)

SafetyBench English questions: ~4,900 after language filtering.

Runtime estimate (A100 40 GB, vLLM, batch=256, thinking=ON, MAX_TOKENS=512):
  Download Qwen3-8B  : ~15 min (first time, ~16 GB)
  Inference          : ~2-3 h  (8 rollouts × 5 k questions × 512 max tokens)

Usage (on cluster):
    cd /scratch/safety
    PYTHONPATH=/scratch/safety nohup python /scratch/safety/prepare_safetybench_synthetic.py \\
        > /scratch/safety/logs/safetybench_synthetic.log 2>&1 &
    echo "PID: $!"
    tail -f /scratch/safety/logs/safetybench_synthetic.log

Output files (all under OUTPUT_DIR):
    safetybench_ci_train.jsonl   -- CI format with trace, train split
    safetybench_ci_val.jsonl     -- CI format with trace, val split
    safetybench_sft_train.jsonl  -- SFT messages (CoT format), train split
    safetybench_sft_val.jsonl    -- SFT messages (CoT format), val split
"""

from __future__ import annotations

import ast
import json
import random
import re
from collections import Counter
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

# ── Config ─────────────────────────────────────────────────────────────────────
MODEL_PATH   = "Qwen/Qwen3-8B"    # labeler — downloads ~16 GB on first run

OUTPUT_DIR   = "/scratch/safety/data"

# System prompt: no /no_think — we WANT Qwen3-8B to generate a reasoning trace
SYSTEM_PROMPT = (
    "You are a safety expert. For each multiple-choice question, "
    "reason carefully through the options, then provide your final answer "
    "enclosed in \\boxed{} — for example: \\boxed{A}."
)

N_SAMPLES      = 8      # rollouts per question for majority vote
MIN_AGREEMENT  = 6      # keep if ≥ 6/8 rollouts agree on the same letter (~97%+ purity)
TEMPERATURE    = 1.0    # needs variation for majority vote to be meaningful
MAX_TOKENS     = 512    # enough for full <think> trace + \boxed{X}
GPU_MEM_UTIL   = 0.80   # ~32 GB used; Qwen3-8B BF16 = ~16 GB
MAX_MODEL_LEN  = 4096
BATCH_SIZE     = 256

# Trace quality filter: keep traces in this character-length range
MIN_TRACE_CHARS = 80    # discard trivially short traces
MAX_TRACE_CHARS = 2000  # discard runaway traces

HF_DATASET  = "thu-coai/SafetyBench"
LANG_FILTER = "en"      # set None to include Chinese questions too
SEED        = 42
VAL_FRAC    = 0.05

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


# ── Helpers ────────────────────────────────────────────────────────────────────

def parse_options(options_field) -> list[str]:
    """Parse SafetyBench options field (Python repr string or list)."""
    if isinstance(options_field, list):
        return [str(o) for o in options_field]
    try:
        result = ast.literal_eval(str(options_field))
        if isinstance(result, list):
            return [str(o) for o in result]
    except (ValueError, SyntaxError):
        pass
    return []


def build_prompt(question: str, options: list[str]) -> str:
    """Format exactly like CI safety.jsonl: Question\n\nA) opt1\nB) opt2..."""
    parts = [question.strip(), ""]
    for i, opt in enumerate(options):
        parts.append(f"{LETTERS[i]}) {opt.strip()}")
    return "\n".join(parts)


def extract_trace(text: str) -> str:
    """Extract content between <think> and </think> tags."""
    m = re.search(r"<think>(.*?)</think>", text, re.DOTALL | re.IGNORECASE)
    return m.group(1).strip() if m else ""


def extract_final_answer(text: str) -> str | None:
    """
    Extract the answer letter from the text AFTER </think>.
    Falls back to any \\boxed{X} in the text if no </think> found.
    """
    # Try to find \boxed{X} only in text after </think>
    after_think = text
    think_end = re.search(r"</think>", text, re.IGNORECASE)
    if think_end:
        after_think = text[think_end.end():]

    # Extract last \boxed{X} in the post-think section
    matches = re.findall(r"\\boxed\{+([^}]+?)\}+", after_think, re.IGNORECASE)
    if matches:
        letter = matches[-1].strip().upper()
        if len(letter) == 1 and letter.isalpha():
            return letter

    # Fallback: any \boxed{X} anywhere in the output
    matches = re.findall(r"\\boxed\{+([^}]+?)\}+", text, re.IGNORECASE)
    if matches:
        letter = matches[-1].strip().upper()
        if len(letter) == 1 and letter.isalpha():
            return letter

    return None


def select_best_trace(traces_for_winner: list[str]) -> str:
    """
    Pick the best trace from a list of traces that all gave the correct answer.
    Prefer the shortest trace that meets the minimum length requirement.
    This tends to give the clearest, most focused reasoning.
    """
    valid = [
        t for t in traces_for_winner
        if MIN_TRACE_CHARS <= len(t) <= MAX_TRACE_CHARS
    ]
    if not valid:
        # Relax: accept any non-empty trace
        valid = [t for t in traces_for_winner if t.strip()]
    if not valid:
        return ""
    # Prefer shorter (clearer) traces
    return min(valid, key=len)


def majority_vote_with_traces(
    completions,
) -> tuple[str | None, int, str]:
    """
    Run majority vote and return (winner_letter, vote_count, best_trace).
    The best_trace is taken from one of the winning rollouts.
    """
    votes: Counter = Counter()
    traces_by_letter: dict[str, list[str]] = {}

    for comp in completions:
        text = comp.text
        letter = extract_final_answer(text)
        trace  = extract_trace(text)
        if letter:
            votes[letter] += 1
            traces_by_letter.setdefault(letter, []).append(trace)

    if not votes:
        return None, 0, ""

    winner, count = votes.most_common(1)[0]
    best_trace = select_best_trace(traces_by_letter.get(winner, []))
    return winner, count, best_trace


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    random.seed(SEED)
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  SafetyBench synthetic CoT labeling")
    print(f"  Labeler   : {MODEL_PATH}  (Qwen3-8B, thinking=ON)")
    print(f"  Threshold : {MIN_AGREEMENT}/{N_SAMPLES} majority vote")
    print(f"  Max tokens: {MAX_TOKENS} (captures full <think> trace)")
    print(f"  Language  : {LANG_FILTER or 'all'}")
    print(f"{'='*60}\n")

    # ── 1. Load SafetyBench ───────────────────────────────────────────────────
    print(f"[1/5] Downloading {HF_DATASET}...")
    try:
        ds = load_dataset(HF_DATASET, split="test")
    except Exception:
        ds_dict = load_dataset(HF_DATASET)
        ds = ds_dict[list(ds_dict.keys())[0]]

    raw = []
    n_lang_skip = 0
    n_opt_skip  = 0

    for row in ds:
        lang = row.get("language", row.get("lang", "en"))
        if LANG_FILTER and lang != LANG_FILTER:
            n_lang_skip += 1
            continue

        options = parse_options(row.get("options", []))
        if len(options) < 2:
            n_opt_skip += 1
            continue

        question = str(row.get("question", "")).strip()
        if not question:
            n_opt_skip += 1
            continue

        raw.append({
            "question":  question,
            "options":   options,
            "category":  str(row.get("category", "unknown")),
            "id":        str(row.get("id", "")),
        })

    print(f"  Total rows     : {len(ds):,}")
    print(f"  Skipped (lang) : {n_lang_skip:,}")
    print(f"  Skipped (opts) : {n_opt_skip:,}")
    print(f"  Usable         : {len(raw):,}")
    if raw:
        r0 = raw[0]
        print(f"\n  First row: q={repr(r0['question'][:70])}")
        print(f"             opts={r0['options'][:3]}")

    if not raw:
        raise RuntimeError("No questions found. Check dataset name and LANG_FILTER.")

    # ── 2. Load Qwen3-8B ─────────────────────────────────────────────────────
    print(f"\n[2/5] Loading tokenizer: {MODEL_PATH}")
    print(f"  (First run downloads ~16 GB — may take 10-20 min)")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    print(f"  Loading vLLM engine (thinking=ON)...")
    llm = LLM(
        model=MODEL_PATH,
        tokenizer=MODEL_PATH,
        dtype="bfloat16",
        max_model_len=MAX_MODEL_LEN,
        gpu_memory_utilization=GPU_MEM_UTIL,
        enforce_eager=True,     # avoids torch.compile crash (Lesson 16)
    )
    params = SamplingParams(
        n=N_SAMPLES,
        temperature=TEMPERATURE,
        top_p=1.0,
        max_tokens=MAX_TOKENS,
    )

    # ── 3. Build prompts ──────────────────────────────────────────────────────
    print(f"\n[3/5] Building {len(raw):,} prompts (thinking=ON)...")
    prompts = []
    for row in raw:
        prompt_text = build_prompt(row["question"], row["options"])
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt_text},
        ]
        try:
            # enable_thinking=True → generation prompt ends at "assistant\n"
            # → full <think>trace</think>\n\n\boxed{X} is in the model output
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,
            )
        except TypeError:
            # Older tokenizer: enable_thinking kwarg not supported
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        prompts.append(prompt)

    # ── 4. Inference ──────────────────────────────────────────────────────────
    print(f"\n[4/5] Running {N_SAMPLES}-shot rollouts on {len(raw):,} questions "
          f"(batch={BATCH_SIZE}, max_tokens={MAX_TOKENS})...")
    print(f"  Estimated time: 2-3 h on A100 40 GB")

    kept_ci  = []
    kept_sft = []
    n_kept   = 0
    n_skip   = 0
    n_no_trace = 0
    agree_dist = {i: 0 for i in range(N_SAMPLES + 1)}

    for start in tqdm(range(0, len(raw), BATCH_SIZE), desc="Labeling"):
        batch_rows    = raw[start : start + BATCH_SIZE]
        batch_prompts = prompts[start : start + BATCH_SIZE]
        outputs       = llm.generate(batch_prompts, params)

        for row, output in zip(batch_rows, outputs):
            winner, vote_count, trace = majority_vote_with_traces(output.outputs)
            agree_dist[vote_count] += 1

            if winner is None or vote_count < MIN_AGREEMENT:
                n_skip += 1
                continue

            if not trace:
                # Keep with empty trace if vote is strong (8/8), else skip
                if vote_count < N_SAMPLES:
                    n_no_trace += 1
                    n_skip += 1
                    continue

            n_kept += 1
            prompt_text = build_prompt(row["question"], row["options"])

            # CI format — matches safety.jsonl + includes trace for inspection
            kept_ci.append({
                "prompt":   prompt_text,
                "answer":   winner,
                "trace":    trace,
                "category": row["category"],
                "n_agree":  vote_count,
            })

            # SFT messages format — CoT assistant turn for knowledge distillation
            # Assistant content: <think>\n[trace]\n</think>\n\n\boxed{X}
            if trace:
                assistant_content = (
                    f"<think>\n{trace}\n</think>\n\n\\boxed{{{winner}}}"
                )
            else:
                # Rare case: unanimous vote but no trace (model skipped thinking)
                assistant_content = f"\\boxed{{{winner}}}"

            kept_sft.append({
                "messages": [
                    {"role": "system",    "content": SYSTEM_PROMPT},
                    {"role": "user",      "content": prompt_text},
                    {"role": "assistant", "content": assistant_content},
                ]
            })

    # ── 5. Summary ────────────────────────────────────────────────────────────
    total = len(raw)
    print(f"\n{'='*55}  RESULTS")
    print(f"  Total questions          : {total:>6,}")
    print(f"  Kept (≥{MIN_AGREEMENT}/8 agree)      : {n_kept:>6,}  ({100*n_kept/total:.1f}%)")
    print(f"  Skipped (low agreement)  : {n_skip:>6,}  ({100*n_skip/total:.1f}%)")
    print(f"  Skipped (no trace, <8/8) : {n_no_trace:>6,}")

    print(f"\nAgreement distribution:")
    for v in range(N_SAMPLES + 1):
        n = agree_dist[v]
        if n > 0:
            bar = "█" * int(30 * n / total)
            print(f"  {v}/8: {n:>5,}  ({100*n/total:.1f}%) {bar}")

    print(f"\nEstimated label accuracy (Qwen3-8B, thinking=ON):")
    print(f"  6/8 majority : ~99%+")
    print(f"  7/8 majority : ~99.5%+")
    print(f"  8/8 unanimous: ~99.9%+")

    # Trace length distribution
    trace_lengths = [len(r["trace"]) for r in kept_ci if r["trace"]]
    if trace_lengths:
        avg_len = sum(trace_lengths) / len(trace_lengths)
        print(f"\nTrace stats: avg={avg_len:.0f} chars, "
              f"min={min(trace_lengths)}, max={max(trace_lengths)}")

    # Answer distribution
    answer_dist = Counter(r["answer"] for r in kept_ci)
    print(f"\nAnswer distribution: {dict(sorted(answer_dist.items()))}")

    # Show 3 sample outputs
    print(f"\nSample outputs (first 3):")
    for i, r in enumerate(kept_ci[:3]):
        print(f"\n  [{i}] answer={r['answer']}  n_agree={r['n_agree']}")
        print(f"  prompt : {repr(r['prompt'][:80])}")
        print(f"  trace  : {repr(r['trace'][:120])}...")

    # ── 6. Train/val split and save ───────────────────────────────────────────
    idx = list(range(len(kept_ci)))
    random.shuffle(idx)
    n_val     = max(1, int(len(idx) * VAL_FRAC))
    val_idx   = set(idx[:n_val])
    train_idx = [i for i in idx if i not in val_idx]

    ci_train  = [kept_ci[i]  for i in train_idx]
    ci_val    = [kept_ci[i]  for i in sorted(val_idx)]
    sft_train = [kept_sft[i] for i in train_idx]
    sft_val   = [kept_sft[i] for i in sorted(val_idx)]

    def save_ci(data, path):
        with open(path, "w", encoding="utf-8") as f:
            for r in data:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"  Saved {len(data):,} → {path}")

    def save_sft(data, path):
        with open(path, "w", encoding="utf-8") as f:
            for r in data:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"  Saved {len(data):,} → {path}")

    print(f"\n[5/5] Saving...")
    save_ci(ci_train,   out_dir / "safetybench_ci_train.jsonl")
    save_ci(ci_val,     out_dir / "safetybench_ci_val.jsonl")
    save_sft(sft_train, out_dir / "safetybench_sft_train.jsonl")
    save_sft(sft_val,   out_dir / "safetybench_sft_val.jsonl")

    print(f"\nDone.")
    print(f"""
Next steps:

  1. Combine with SaladBench SFT data:
     cat {out_dir}/salad_sft_train.jsonl \\
         {out_dir}/safetybench_sft_train.jsonl \\
         > {out_dir}/combined_sft_train.jsonl
     cat {out_dir}/salad_sft_val.jsonl \\
         {out_dir}/safetybench_sft_val.jsonl \\
         > {out_dir}/combined_sft_val.jsonl

  2. Train Qwen3-1.7B BASE with thinking=ON
     (assistant content starts with <think> → training must NOT apply
      the 'enable_thinking=false' patch in sft_train.py):

     PYTHONPATH=/scratch/safety nohup python /scratch/safety/safety/training/sft_train.py \\
         --train {out_dir}/combined_sft_train.jsonl \\
         --val   {out_dir}/combined_sft_val.jsonl \\
         --output-dir /scratch/safety/sft_cot_distill \\
         --epochs 2 --lr 2e-5 --lora-rank 16 --lora-alpha 32 \\
         --max-seq-length 1024 --max-grad-norm 0.3 \\
         > /scratch/safety/logs/sft_cot_distill.log 2>&1 &

     NOTE: max-seq-length 1024 (not 512) to fit CoT traces in assistant turn.
     NOTE: if sft_train.py forces thinking=OFF (adds enable_thinking=false),
           you must remove or comment out that patch for this training run.
""")


if __name__ == "__main__":
    main()
