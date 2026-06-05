"""
Path B — Step 1: Generate correct CoT traces from Qwen3-1.7B base (thinking=ON).

Reads MMLU safety examples, runs vLLM inference with enable_thinking=True on the
BASE model (not the fine-tuned one), and keeps only examples where the predicted
answer matches the gold label AND the reasoning block is non-trivial.

Output JSONL assistant turn format:
    <think>
    [reasoning]
    </think>

    \\boxed{X}

Typical yield: ~55–65% of input examples.

Usage:
    python generate_cot_traces.py \\
        --input  /scratch/safety/data/mmlu_safety_train.jsonl \\
        --output /scratch/safety/data/mmlu_safety_cot.jsonl

After this, run the nuclear SFT:
    python /scratch/safety/safety/training/sft_train.py \\
        --model /scratch/safety/sft_mcq_v2/merged \\
        --train /scratch/safety/data/mmlu_safety_cot.jsonl \\
        --val   /scratch/safety/data/pkusafe_sft_val.jsonl \\
        --output-dir /scratch/safety/sft_nuclear \\
        --lora-r 64 --lora-alpha 128 --lr 1e-5 --epochs 3 \\
        --batch-size 4 --grad-accum 8 --max-seq-length 2048 \\
        --max-grad-norm 0.3
"""

import argparse
import json
import re
from pathlib import Path

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

# System prompt used for GENERATION (encourages reasoning).
# Also written into the output JSONL so the training distribution matches
# inference after the template is updated to this prompt.
REASONING_SYSTEM_PROMPT = (
    "You are a safety expert. For each multiple-choice question, reason step by step "
    "through the options, then give your final answer as a single letter inside "
    "\\boxed{}, e.g. \\boxed{A}. Do not include anything after the \\boxed{}."
)

# Minimum reasoning length (chars) to keep a trace.
MIN_THINKING_CHARS = 80


def extract_boxed(text: str) -> str | None:
    m = re.search(r"\\boxed\{+([A-Z])\}+", text)
    return m.group(1) if m else None


def get_gold(ex: dict) -> str:
    """Gold answer from a top-level 'answer' field, or fall back to the
    \\boxed{X} letter in the assistant message (messages-format inputs)."""
    gold = ex.get("answer", "")
    if not gold:
        msgs = ex.get("messages", [])
        if len(msgs) >= 3:
            gold = extract_boxed(msgs[-1].get("content", "")) or ""
    return gold


def build_prompts(examples: list[dict], tokenizer) -> list[str]:
    """
    Build generation prompts using the BASE model tokenizer with enable_thinking=True.
    The prompt will end with  <|im_start|>assistant\\n<think>\\n  so vLLM generates
    inside the think block — output.text begins with the reasoning (no opening <think>).
    """
    prompts = []
    for ex in examples:
        user_content = ex["messages"][1]["content"]
        messages = [
            {"role": "system", "content": REASONING_SYSTEM_PROMPT},
            {"role": "user",   "content": user_content},
        ]
        try:
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,   # Qwen3 default — appends <think>\n to prompt
            )
        except TypeError:
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        prompts.append(prompt)
    return prompts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  default="/scratch/safety/data/mmlu_safety_train.jsonl")
    parser.add_argument("--output", default="/scratch/safety/data/mmlu_safety_cot.jsonl")
    parser.add_argument("--model",  default="Qwen/Qwen3-1.7B",
                        help="Base model — must support enable_thinking (Qwen3 family)")
    parser.add_argument("--temperature", type=float, default=0.6,
                        help="Low enough for mostly correct outputs; high enough for diverse traces")
    parser.add_argument("--max-tokens", type=int, default=1536,
                        help="Max tokens per completion — reasoning traces are ~200-600 tokens")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    args = parser.parse_args()

    # ── Load examples ─────────────────────────────────────────────────────────
    raw = Path(args.input).read_text(encoding="utf-8").splitlines()
    examples = [json.loads(l) for l in raw if l.strip()]
    print(f"Loaded {len(examples)} examples from {args.input}")

    # ── Build prompts ─────────────────────────────────────────────────────────
    print(f"Loading tokenizer from {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    prompts = build_prompts(examples, tokenizer)

    # ── vLLM inference ────────────────────────────────────────────────────────
    print("Starting vLLM...")
    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=4096,
        enforce_eager=True,     # avoids torch.compile graph-split crash (Lesson 16)
    )
    sampling = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        stop=["<|im_end|>"],
    )

    print(f"Generating traces for {len(prompts)} prompts...")
    outputs = llm.generate(prompts, sampling)

    # ── Filter and save ───────────────────────────────────────────────────────
    kept = 0
    skipped_wrong   = 0
    skipped_no_box  = 0
    skipped_no_think = 0

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as fout:
        for ex, out in zip(examples, outputs):
            # The prompt ended with "<think>\n", so output.text begins INSIDE the
            # think block — we find "</think>" to split reasoning from the answer.
            raw_text = out.outputs[0].text

            think_end = raw_text.find("</think>")
            if think_end == -1:
                skipped_no_think += 1
                continue

            thinking = raw_text[:think_end].strip()
            if len(thinking) < MIN_THINKING_CHARS:
                skipped_no_think += 1
                continue

            rest = raw_text[think_end + len("</think>"):].strip()
            pred = extract_boxed(rest)
            if pred is None:
                skipped_no_box += 1
                continue

            gold = get_gold(ex)
            if pred != gold:
                skipped_wrong += 1
                continue

            assistant_content = f"<think>\n{thinking}\n</think>\n\n\\boxed{{{pred}}}"

            # Output system message matches REASONING_SYSTEM_PROMPT so that
            # sft_train.py's apply_chat_template sees an existing system message
            # and does NOT inject the /no_think template system prompt.
            # After training, update chat_template.jinja to use this prompt too.
            record = {
                "messages": [
                    {"role": "system",    "content": REASONING_SYSTEM_PROMPT},
                    {"role": "user",      "content": ex["messages"][1]["content"]},
                    {"role": "assistant", "content": assistant_content},
                ],
                "category": ex.get("category", ""),
                "answer": gold,
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            kept += 1

    total = len(examples)
    print(f"\n── Results ────────────────────────────────")
    print(f"  Kept (correct + has reasoning) : {kept:>5} / {total}  ({100*kept/total:.1f}%)")
    print(f"  Skipped — wrong answer         : {skipped_wrong:>5}")
    print(f"  Skipped — no \\boxed{{}}          : {skipped_no_box:>5}")
    print(f"  Skipped — reasoning too short  : {skipped_no_think:>5}")
    print(f"  Output: {out_path}")
    print()
    print("Next: verify a few traces look sensible, then run sft_train.py (see docstring).")
    print("After training+merging, update chat_template.jinja to remove '/no_think' and")
    print("replace the system prompt with REASONING_SYSTEM_PROMPT before pushing to HF.")


if __name__ == "__main__":
    main()
