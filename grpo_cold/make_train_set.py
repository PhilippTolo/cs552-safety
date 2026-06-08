"""make_train_set.py — difficulty-filter the training prompts for GRPO.

The cold-from-BASE run stalled because BASE + golden nuggets is over-confident on
easy safety MCQ: all 8 rollouts per prompt come out identical → zero group
variance → zero advantage → zero gradient (frac_reward_zero_std≈1.0, grad_norm≈0).

Fix (mirrors nlp_project's grpo_filter.py): run BASE + nuggets n=8 over the HEAD
training slice, keep ONLY "learnable" prompts where 1..n-1 of the rollouts are
correct (drop all-correct AND all-wrong). Every surviving prompt then guarantees
within-group variance, so GRPO gets real gradient on every step.

Writes ``grpo_cold/train_filtered.jsonl`` ({"question_text","answer","source"});
point cfgs_grpo_safety.yml `dataset.local_file` at it.

Usage (on cluster, GPU must be free):
    PYTHONPATH=/scratch/safety python grpo_cold/make_train_set.py \
        --out grpo_cold/train_filtered.jsonl --n 8 --temperature 1.1
"""

from __future__ import annotations

import argparse
import json

from transformers import AutoTokenizer

from extract_answer import extract_boxed_answer, normalize_letter
from format_data import SYSTEM_PROMPT, load_mc_dataset


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-1.7B")
    p.add_argument("--out", default="train_filtered.jsonl")
    p.add_argument("--fraction", type=float, default=0.30,
                   help="HEAD share to scan (match cfg dataset.fraction)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n", type=int, default=8, help="rollouts per prompt")
    p.add_argument("--temperature", type=float, default=1.1,
                   help="higher → more exploration → more learnable splits")
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--top_k", type=int, default=20)
    p.add_argument("--max_tokens", type=int, default=512)
    p.add_argument("--max_model_len", type=int, default=4096)
    p.add_argument("--gpu_mem", type=float, default=0.90)
    p.add_argument("--keep_lo", type=int, default=1, help="min correct to keep")
    p.add_argument("--keep_hi", type=int, default=None, help="max correct to keep (default n-1)")
    p.add_argument("--hf_token", default=None)
    args = p.parse_args()
    keep_hi = args.keep_hi if args.keep_hi is not None else args.n - 1

    from vllm import LLM, SamplingParams

    base = load_mc_dataset(fraction=args.fraction, seed=args.seed, hf_token=args.hf_token)
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    prompts = [
        tok.apply_chat_template(
            [{"role": "system", "content": SYSTEM_PROMPT},
             {"role": "user", "content": r["question_text"]}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        for r in base
    ]
    gold = [r["answer"] for r in base]

    llm = LLM(model=args.model, max_model_len=args.max_model_len,
              gpu_memory_utilization=args.gpu_mem, dtype="bfloat16",
              trust_remote_code=True, enforce_eager=True)
    sampling = SamplingParams(n=args.n, temperature=args.temperature,
                              top_p=args.top_p, top_k=args.top_k, max_tokens=args.max_tokens)
    outputs = llm.generate(prompts, sampling)

    buckets = {"all_wrong": 0, "learnable": 0, "all_correct": 0}
    kept = []
    for r, g, out in zip(base, gold, outputs):
        gl = normalize_letter(g)
        n_correct = sum(
            1 for o in out.outputs
            if normalize_letter(extract_boxed_answer(o.text)) == gl
        )
        if n_correct == 0:
            buckets["all_wrong"] += 1
        elif n_correct == args.n:
            buckets["all_correct"] += 1
        else:
            buckets["learnable"] += 1
        if args.keep_lo <= n_correct <= keep_hi:
            kept.append({"question_text": r["question_text"],
                         "answer": r["answer"], "source": r.get("source", "?")})

    with open(args.out, "w", encoding="utf-8") as f:
        for k in kept:
            f.write(json.dumps(k, ensure_ascii=False) + "\n")

    total = sum(buckets.values())
    print("\n=== difficulty buckets (BASE + nuggets, n=%d, temp=%.2f) ===" % (args.n, args.temperature))
    for k, v in buckets.items():
        print(f"  {k:<12} {v:>6}  ({v / total * 100:5.1f}%)")
    print(f"  kept (correct in [{args.keep_lo},{keep_hi}]): {len(kept)} -> {args.out}")
    if len(kept) < 300:
        print("  WARNING: <300 learnable prompts. Raise --temperature or --fraction, "
              "or widen the keep range. GRPO needs enough varied prompts to learn.")


if __name__ == "__main__":
    main()
