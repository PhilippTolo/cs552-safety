"""eval_checkpoint.py — score a GRPO checkpoint on a safety MC test set.

Loads the base model (optionally with a LoRA adapter checkpoint), runs vLLM
batch inference in NON-THINKING mode using the shared SYSTEM_PROMPT (golden
nuggets), extracts the boxed letter, and reports pass@1 (and pass@n if n>1)
overall and per source — mirroring how the course CI evaluates safety_model.

Point --test_set at either:
  * test_set.jsonl  (held-out vibe-trainers tail from make_test_set.py), or
  * the course safety.jsonl  (the real 10-question CI proxy).

Examples:
    python eval_checkpoint.py --adapter ./grpo_safety_checkpoints/checkpoint-200
    python eval_checkpoint.py --adapter ./grpo_safety_checkpoints/final --n 8 --temperature 0.7
    python eval_checkpoint.py --model cs-552-2026-ma-que/safety_model \
        --test_set /scratch/standard-project-m2-ma-que/validation_samples/safety.jsonl
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict

from transformers import AutoTokenizer

from extract_answer import extract_boxed_answer, normalize_letter, parse_gold
from format_data import SYSTEM_PROMPT


def load_records(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def build_prompts(tokenizer, records: list[dict], system_prompt: str) -> list[str]:
    """Chat-template each question in non-thinking mode (matches training)."""
    prompts = []
    for r in records:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": r["prompt"]},
        ]
        prompts.append(tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
        ))
    return prompts


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-1.7B", help="base or merged model")
    p.add_argument("--adapter", default=None, help="LoRA checkpoint dir (optional)")
    p.add_argument("--test_set", default="test_set.jsonl")
    p.add_argument("--n", type=int, default=1, help="samples per question")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top_p", type=float, default=1.0)
    p.add_argument("--top_k", type=int, default=-1)
    p.add_argument("--max_tokens", type=int, default=512)
    p.add_argument("--max_model_len", type=int, default=4096)
    p.add_argument("--max_lora_rank", type=int, default=64)
    p.add_argument("--gpu_mem", type=float, default=0.90)
    p.add_argument("--out", default=None, help="optional: write per-item predictions jsonl")
    args = p.parse_args()

    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    records = load_records(args.test_set)
    print(f"[eval] {len(records)} test items from {args.test_set}")

    tokenizer = AutoTokenizer.from_pretrained(args.adapter or args.model, trust_remote_code=True)
    prompts = build_prompts(tokenizer, records, SYSTEM_PROMPT)

    llm = LLM(
        model=args.model,
        enable_lora=bool(args.adapter),
        max_lora_rank=args.max_lora_rank,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_mem,
        dtype="bfloat16",
        trust_remote_code=True,
        enforce_eager=True,        # avoid vLLM CUDA-graph compile crash (Lesson 16)
    )
    sampling = SamplingParams(
        n=args.n,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_tokens,
    )
    lora_req = LoRARequest("ckpt", 1, args.adapter) if args.adapter else None

    outputs = llm.generate(prompts, sampling, lora_request=lora_req)

    correct1 = total = correctn = 0
    fmt_ok = 0
    per_src = defaultdict(lambda: [0, 0, 0])   # source -> [pass1, passn, total]
    out_rows = []
    for rec, out in zip(records, outputs):
        gold = parse_gold(rec["answer"])
        preds = [normalize_letter(extract_boxed_answer(o.text)) for o in out.outputs]
        ok1 = gold is not None and preds[0] == gold
        okn = gold is not None and any(pp == gold for pp in preds)

        total += 1
        correct1 += ok1
        correctn += okn
        fmt_ok += sum(pp is not None for pp in preds)
        src = rec.get("source", "?")
        per_src[src][0] += ok1
        per_src[src][1] += okn
        per_src[src][2] += 1

        if args.out:
            out_rows.append({**rec, "pred": preds[0], "all_preds": preds,
                             "correct": bool(ok1), "completion": out.outputs[0].text})

    print("\n=== results ===")
    print(f"{'source':<14}{'pass@1':>10}" + (f"{'pass@'+str(args.n):>10}" if args.n > 1 else "") + f"{'n':>8}")
    for src in sorted(per_src):
        c1, cn, t = per_src[src]
        line = f"{src:<14}{c1 / t:>10.3f}" + (f"{cn / t:>10.3f}" if args.n > 1 else "") + f"{t:>8}"
        print(line)
    overall = f"{'ALL':<14}{correct1 / total:>10.3f}" + (f"{correctn / total:>10.3f}" if args.n > 1 else "") + f"{total:>8}"
    print(overall)
    print(f"format_rate: {fmt_ok / (total * args.n):.3f}  (fraction of completions with a boxed letter)")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            for r in out_rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"\n[eval] wrote per-item predictions -> {args.out}")


if __name__ == "__main__":
    main()
