# grpo_cold — GRPO on Qwen3-1.7B for safety MCQ (cold, from BASE)

GRPO **directly on `Qwen/Qwen3-1.7B`** (no SFT first) on
**`cs-552-2026-vibe-trainers/mcq_safety`** (SafetyBench + SALAD-Bench), sources
`safetybench, salad`, answers `A–D`. **Non-thinking** inference. A **golden-nugget
system prompt** (6 worked examples, one per CI category) bootstraps the
`\boxed{}` format in-context. Reward = **format + correctness** (no length term).

This is the safety twin of `nlp_project/multi_lin/` — same algorithm (GSPO/DAPO,
`beta=0`), same reward shape, same files.

## Why cold-from-BASE works here (when GRPO-from-SFT failed before)

Every previous safety GRPO run started from a confident SFT checkpoint (sft_v2,
grpo_v5) and hit `frac_reward_zero_std=1.0` — all 8 rollouts identical → zero
group variance → zero gradient (project CLAUDE.md GRPO lessons). **BASE is not
overconfident**, so its rollouts stay diverse → real GRPO signal. The golden
nuggets supply the output format from in-context demonstration, so GRPO spends
its gradient on safety *knowledge*, not on learning `\boxed{}` from the format
penalty alone.

## Files

| File | Purpose |
|---|---|
| `format_data.py` | vibe-trainers `mcq_safety` → records; `SYSTEM_PROMPT` with golden nuggets; train (head) + held-out test (tail) loaders. |
| `extract_answer.py` | Nested-brace `\boxed{}` parser, `normalize_letter`, `parse_gold` (handles bare letter *or* `\boxed{X}`). |
| `reward_fns.py` | `format_reward` (+0.2 / −0.5) + `correctness_reward` (+1.0). No length reward. |
| `cfgs_grpo_safety.yml` | All hyperparameters. |
| `train_grpo.py` | GRPO trainer (LoRA, non-thinking prompts, optional vLLM, bnb mock). |
| `make_test_set.py` | Held-out test set (tail, disjoint from training) → `test_set.jsonl`. |
| `eval_checkpoint.py` | Score a checkpoint (pass@1 / pass@n, per source) on `test_set.jsonl` **or** the course `safety.jsonl`. |
| `ship_model.py` | Merge LoRA + **bake** golden-nugget system prompt + non-thinking into `chat_template.jinja` + write `generation_config.json` + verify. |

## Usage

```bash
# 0. sanity-check the golden-nugget prompt and the data formatting
python grpo_cold/format_data.py --print_system          # prints SYSTEM_PROMPT
python grpo_cold/format_data.py --out sample.jsonl --per_source 5

# 1. auth (token in cs552-safety/CLAUDE.md, or `hf auth login`)
export WANDB_API_KEY=...      # or pass --no_wandb
hf auth login

# 2. train (cold from BASE; HF generation, vLLM off by default)
cd /scratch/safety
PYTHONPATH=/scratch/safety nohup python grpo_cold/train_grpo.py \
    --config grpo_cold/cfgs_grpo_safety.yml \
    > /scratch/safety/logs/grpo_cold.log 2>&1 &
echo "PID: $!"; tail -f /scratch/safety/logs/grpo_cold.log

# 2b. faster: with a vLLM rollout server
python grpo_cold/train_grpo.py --vllm true
```

LoRA adapter is saved to `./grpo_safety_checkpoints/final`.

**Health checks (first ~20 steps):**
- `reward` rising from ~0.0–0.4 toward ~0.7+ → learning.
- `frac_reward_zero_std` well below 1.0 → rollouts diverse (BASE should give this).
  If it pins at 1.0 from step 1, raise `rollout.temperature` or check the data.
- Format reward should go positive almost immediately (golden nuggets demonstrate
  the box). If it stays negative, the system prompt isn't reaching the prompt.

## Held-out test & evaluating checkpoints

```bash
python grpo_cold/make_test_set.py --train_fraction 0.30   # match cfg dataset.fraction
# pass@1 (greedy) on a checkpoint
python grpo_cold/eval_checkpoint.py --adapter ./grpo_safety_checkpoints/final
# pass@1 + pass@8 with sampling, dump predictions
python grpo_cold/eval_checkpoint.py --adapter ./grpo_safety_checkpoints/final \
    --n 8 --temperature 0.7 --out preds.jsonl
# the real CI proxy (10 course questions)
python grpo_cold/eval_checkpoint.py --adapter ./grpo_safety_checkpoints/final \
    --test_set /scratch/standard-project-m2-ma-que/validation_samples/safety.jsonl
```

## Shipping the `safety_model` checkpoint

The CI calls only `apply_chat_template(messages, add_generation_prompt=True)` and
passes neither our system prompt nor `enable_thinking`. Both must be baked in:

```bash
PYTHONPATH=/scratch/safety python grpo_cold/ship_model.py \
    --adapter /scratch/safety/grpo_safety_checkpoints/final \
    --base Qwen/Qwen3-1.7B \
    --output /scratch/safety/grpo_cold/merged
# verify the printed checks: golden nugget present, user question present,
# prompt tail ends with the empty </think> block (non-thinking active).

# validate before pushing — only push if pass@1 > 76% (the grpo_v5 floor)
python grpo_cold/eval_checkpoint.py --model /scratch/safety/grpo_cold/merged \
    --test_set /scratch/standard-project-m2-ma-que/validation_samples/safety.jsonl

hf upload cs-552-2026-ma-que/safety_model /scratch/safety/grpo_cold/merged .
# restore floor if it regresses:
# hf upload cs-552-2026-ma-que/safety_model /scratch/safety/grpo_v5/merged .
```

## Notes / knobs

- `dataset.sources` drops `wildguard` by default (harm-category rows pollute the
  2–4 option MCQ distribution; see project CLAUDE.md §20). `MAX_LETTER` in
  `format_data.py` caps answers at `D`.
- `beta: 0.0` (DAPO, no KL). Raise to ~0.04 if the policy degenerates.
- `learning_rate: 5e-6` is the teammate's proven default. Lesson 50 suggests
  `1e-5–3e-5` for LoRA GRPO if convergence is slow — bump and re-run if reward
  plateaus early.
- Golden nuggets live **only** in `SYSTEM_PROMPT` (one source of truth) and are
  baked into the shipped template by `ship_model.py`. Edit them in
  `format_data.py` and everything downstream follows.
- **Floor**: never push anything scoring ≤ 76% pass@1 — `grpo_v5/merged` stays
  the fallback.
```
