"""
grpo_safety_v2.py
=================
GRPO fine-tuning for safety MCQ — adapted from nlp_project/grpo.py (Chenxu Yu).

Key upgrades vs grpo_train.py:
  - OmegaConf YAML config (grpo_safety_cfgs.yml)
  - vLLM colocate mode for fast rollouts (3-5× speedup)
  - Dr.GRPO  : loss_type="dr_grpo", scale_rewards=false  — removes biased normalisation
  - DAPO     : epsilon_high=0.28 > epsilon=0.2           — prevents entropy collapse
  - KL-free  : beta=0.0                                  — no frozen reference model needed
  - Full-vocab sampling: top_k=-1, top_p=1.0             — maximum rollout diversity
  - Separate reward functions with reward_weights         — easier to tune

Usage (on cluster):
    cd /scratch/safety
    PYTHONPATH=/scratch/safety nohup python /scratch/safety/grpo_safety_v2.py \
        > /scratch/safety/logs/grpo_v2.log 2>&1 &
    echo "PID: $!"
    tail -f /scratch/safety/logs/grpo_v2.log

Kill condition: if "frac_reward_zero_std=1.0" from step 1
  → model too confident at temp=1.0
  → re-run grpo_filter_safety_v2.py with the base model to check actual yield

After training:
    python /scratch/safety/safety/training/merge_lora.py \
        --adapter /scratch/safety/grpo_safety_v2/lora_adapter \
        --base /scratch/safety/sft_vibe4_v2base/merged \
        --output /scratch/safety/grpo_safety_v2/merged
    cp /scratch/safety/sft_vibe4_v2base/merged/chat_template.jinja \
       /scratch/safety/grpo_safety_v2/merged/
    cp /scratch/safety/sft_vibe4_v2base/merged/generation_config.json \
       /scratch/safety/grpo_safety_v2/merged/
    python /scratch/safety/shared/test_vllm_base.py \
        --model /scratch/safety/grpo_safety_v2/merged \
        --validation-file /scratch/standard-project-m2-ma-que/validation_samples/safety.jsonl
    # Push if pass@1 > 76% AND format_rate >= 90%
    hf upload cs-552-2026-ma-que/safety_model /scratch/safety/grpo_safety_v2/merged .
"""

import os
import sys
import types

# bitsandbytes is incompatible with CUDA 12.8 on this cluster (Lesson 1).
# Pre-populate sys.modules with mock modules BEFORE any peft/trl import so
# peft falls through to the standard (non-quantised) LoRA path.
# Must use types.ModuleType — MagicMock causes __spec__ AttributeError in TRL.
if "bitsandbytes" not in sys.modules:
    _bnb = types.ModuleType("bitsandbytes")
    for _s in ["nn", "optim", "functional", "cextension", "cuda_setup",
               "utils", "research", "research.nn", "research.nn.modules"]:
        _sub = types.ModuleType(f"bitsandbytes.{_s}")
        sys.modules[f"bitsandbytes.{_s}"] = _sub
        setattr(_bnb, _s.split(".")[0], _sub)
    class _Stub:
        pass
    _bnb.nn.Linear8bitLt = _Stub
    _bnb.nn.Linear4bit   = _Stub
    _bnb.nn.Int8Params   = _Stub
    _bnb.cextension.COMPILED_WITH_CUDA = False
    sys.modules["bitsandbytes"] = _bnb

import torch
from omegaconf import OmegaConf
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import GRPOConfig, GRPOTrainer
from peft import LoraConfig

from safety.training.reward_functions import correctness_reward, format_reward
from safety.training.utils import find_latest_checkpoint


# ── Configuration ──────────────────────────────────────────────────────────────
config = OmegaConf.load(
    os.path.join(os.path.dirname(__file__), "grpo_safety_cfgs.yml")
)

grpo_args_dict  = OmegaConf.to_container(config.grpo_args,   resolve=True)
lora_args_dict  = OmegaConf.to_container(config.lora_config, resolve=True)
SYSTEM_PROMPT   = config.system_prompt

# ── Resume from checkpoint ─────────────────────────────────────────────────────
resume_from = find_latest_checkpoint(config.grpo_args.output_dir)
if resume_from:
    print(f"[Resume] Resuming from: {resume_from}")
else:
    print("[Resume] No checkpoint found — starting fresh.")

# ── Model & Tokenizer ──────────────────────────────────────────────────────────
print(f"[1/5] Loading tokenizer: {config.model_name}")
tokenizer = AutoTokenizer.from_pretrained(config.model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

# Ensure thinking=OFF: sft_vibe4_v2base already has this in its template, but
# we patch defensively — GRPOTrainer's internal apply_chat_template calls would
# otherwise inject <think>\n before each rollout, causing the model to spend its
# entire 64-token budget on reasoning instead of \boxed{X} → reward=0 → no gradient.
_OFF = "{%- set enable_thinking = false %}"
if tokenizer.chat_template and _OFF not in tokenizer.chat_template:
    tokenizer.chat_template = _OFF + "\n" + tokenizer.chat_template
    print("  Patched chat_template: enable_thinking=false")

print(f"[2/5] Loading model: {config.model_name}")
model = AutoModelForCausalLM.from_pretrained(
    config.model_name,
    torch_dtype=torch.bfloat16,
    attn_implementation="eager",   # FA2 has NaN risk with Qwen3 BF16 on safety data
)
model.config.pad_token_id            = tokenizer.pad_token_id
model.config.bos_token_id            = tokenizer.bos_token_id
model.generation_config.bos_token_id = tokenizer.bos_token_id

if hasattr(model, "print_trainable_parameters"):
    model.print_trainable_parameters()

# ── Dataset ────────────────────────────────────────────────────────────────────

def make_conversation(batch: dict) -> dict:
    """
    Build the 'prompt' column expected by GRPOTrainer (batched map).

    Input columns (from grpo_filter_safety_v2.py output):
      question : str — the MCQ question text (options already embedded)
      answer   : str — single uppercase letter (A/B/C/D)

    GRPOTrainer applies apply_chat_template to 'prompt', so we pass
    messages in OpenAI list-of-dicts format. The 'answer' column is
    forwarded unchanged to the reward functions as **kwargs.
    """
    return {
        "prompt": [
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": question},
            ]
            for question in batch["question"]
        ]
    }


print("[3/5] Loading dataset...")
import json
raw = []
with open(config.grpo_dataset, encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            raw.append(json.loads(line))

# Filter to "suitable" only (grpo_filter already does this, but belt-and-suspenders)
suitable = [r for r in raw if r.get("category") == "suitable"]
print(f"  Total: {len(raw):,}  |  Suitable: {len(suitable):,}")

if not suitable:
    raise RuntimeError(
        f"No 'suitable' examples found in {config.grpo_dataset}.\n"
        "Run grpo_filter_safety_v2.py first to generate the filtered dataset."
    )

# 99% train / 1% eval
n_eval  = max(8, len(suitable) // 100)
n_train = len(suitable) - n_eval
train_raw = suitable[:n_train]
eval_raw  = suitable[n_train:]
print(f"  Train: {len(train_raw):,}  |  Eval: {len(eval_raw):,}")

train_ds = Dataset.from_list(train_raw).map(make_conversation, batched=True)
eval_ds  = Dataset.from_list(eval_raw).map(make_conversation, batched=True)

# ── Trainer setup ──────────────────────────────────────────────────────────────
print("[4/5] Building GRPOConfig and LoraConfig...")
grpo_config = GRPOConfig(**grpo_args_dict)
lora_config = LoraConfig(**lora_args_dict)

trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    args=grpo_config,
    peft_config=lora_config,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    # Order MUST match reward_weights in grpo_safety_cfgs.yml: [1.0, 0.5]
    # reward = 1.0 × correctness_reward + 0.5 × format_reward
    reward_funcs=[correctness_reward, format_reward],
)

# ── Diagnostics ────────────────────────────────────────────────────────────────
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Device: {torch.cuda.get_device_name(0)}")

if hasattr(trainer.model, "print_trainable_parameters"):
    trainer.model.print_trainable_parameters()
else:
    trainable = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in trainer.model.parameters())
    print(f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.4f}%)")

# ── Train ──────────────────────────────────────────────────────────────────────
print("[5/5] Starting GRPO training...")
print(
    f"  Base model     : {config.model_name}\n"
    f"  Dataset        : {config.grpo_dataset} ({len(suitable):,} suitable examples)\n"
    f"  LoRA           : r={lora_args_dict['r']}, alpha={lora_args_dict['lora_alpha']}\n"
    f"  Algorithm      : Dr.GRPO + DAPO (beta=0, epsilon_high={grpo_args_dict.get('epsilon_high', '?')})\n"
    f"  Rollout temp   : {grpo_args_dict.get('temperature', '?')} (top_k=-1, top_p=1.0)\n"
    f"  max_completion : {grpo_args_dict.get('max_completion_length', '?')} tokens\n"
    f"  Reward weights : correctness=1.0, format=0.5\n"
    f"  Expected output: /scratch/safety/grpo_safety_v2/\n"
)
print(
    "HEALTH CHECK (watch first 20 steps):\n"
    "  GOOD: frac_reward_zero_std < 0.5 → real gradient signal\n"
    "  KILL: frac_reward_zero_std = 1.0 from step 1 → model too confident at temp=1.0\n"
    "        → re-run filter or increase temperature in cfgs.yml"
)

trainer.train(resume_from_checkpoint=resume_from)

# ── Save ───────────────────────────────────────────────────────────────────────
save_dir = os.path.join(grpo_args_dict["output_dir"], "lora_adapter")
trainer.save_model(save_dir)
tokenizer.save_pretrained(save_dir)

print(f"\nLoRA adapter saved: {save_dir}")
print(
    f"\nNext steps:\n"
    f"  1. Merge: python /scratch/safety/safety/training/merge_lora.py \\\n"
    f"               --adapter {save_dir} \\\n"
    f"               --base {config.model_name} \\\n"
    f"               --output {grpo_args_dict['output_dir']}/merged\n"
    f"  2. Copy template + gen_config from base model\n"
    f"  3. Validate: PYTHONPATH=/scratch/safety python /scratch/safety/shared/test_vllm_base.py "
    f"--model {grpo_args_dict['output_dir']}/merged\n"
    f"  4. Push if pass@1 > 76%: hf upload cs-552-2026-ma-que/safety_model "
    f"{grpo_args_dict['output_dir']}/merged ."
)
