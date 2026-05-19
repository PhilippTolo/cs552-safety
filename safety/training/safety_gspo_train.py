"""
safety_gspo_train.py
====================
GSPO training entry point for the safety MCQ model.

Mirrors the math teammate's train_gspo.py (Chenxu Yu, nlp_project/) with
safety-specific adaptations:

  - Local model path (sft_v3/merged) instead of HF repo
  - Local filtered JSONL instead of HF dataset
  - 2 reward functions: correctness + boxed_penalty (no length reward)
  - MAX_NEW_TOKENS=512  (MCQ + brief <think> trace; not 8192 like math)
  - temperature=1.0     (sft_v3 is confident on binary MCQ; needs diversity)
  - enable_thinking=True in prompts

GSPO vs GRPO:
  importance_sampling_level="sequence" turns standard GRPO into GSPO — a more
  stable variant that avoids clipping instabilities at high rollout counts.
  KL_COEFF=0.0 (no KL penalty) follows DAPO / DeepSeek-R1 style.

Reward design:
  correctness_reward : +2.0 correct, 0.0 wrong
  boxed_penalty      : -1.5 no boxed (non-truncated), -0.5 truncated

Prerequisites:
  1. Unsloth installed:  pip install unsloth
  2. sft_v3/merged exists (sft_train.py with --thinking-mode data)
  3. pkusafe_grpo_filtered.jsonl exists (safety_grpo_filter.py)

Usage (on cluster):
    python /scratch/safety/safety/training/safety_gspo_train.py
"""

from __future__ import annotations

import json
import logging
import os

import torch
from datasets import Dataset
from unsloth import FastLanguageModel, PatchFastRL
from trl import GRPOConfig, GRPOTrainer

from reward_functions import (
    boxed_penalty,
    correctness_reward,
    describe_reward_design,
)

# Patch TRL's GRPOTrainer with Unsloth kernels BEFORE any model loading
PatchFastRL("GRPO", FastLanguageModel)

# Fix: Unsloth decorates attention forward with @torch.amp.autocast("cuda"), which
# defaults to float16. Activations become FP16 but LoRA matrices are BF16, so
# addmm_() raises "Half vs BFloat16". Cast X → BF16 on every matmul_lora call.
import unsloth.kernels.fast_lora as _unsloth_fast_lora
_orig_matmul_lora = _unsloth_fast_lora.matmul_lora

def _safe_matmul_lora(X, W, W_quant, A, B, s, out=None):
    # The enclosing @torch.amp.autocast("cuda") (default FP16) makes matmul
    # outputs FP16 even if inputs are BF16, while LoRA matrices stay BF16.
    # Disable autocast and cast everything to B.dtype (BF16) to stay consistent.
    with torch.amp.autocast("cuda", enabled=False):
        d = B.dtype  # BF16
        X = X.to(d)
        if W is not None:
            W = W.to(d)
        if A is not None:
            A = A.to(d)
        if out is not None and out.dtype != d:
            out = out.to(d)
        return _orig_matmul_lora(X, W, W_quant, A, B, s, out)

_unsloth_fast_lora.matmul_lora = _safe_matmul_lora

# ===========================================================================
# CONFIGURATION
# ===========================================================================

SFT_MODEL_PATH   = "/scratch/safety/sft_v2/merged"
FILTERED_JSONL   = "/scratch/safety/data/pkusafe_grpo_filtered.jsonl"
OUTPUT_DIR       = "/scratch/safety/grpo_v5"

# Create output dir NOW — before logging.basicConfig tries to open the log file there.
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Labels with non-zero reward variance — all_wrong / all_correct have no gradient
ACTIVE_LABELS    = {"suitable", "pass1", "pass7"}

# Model
MAX_SEQ_LENGTH   = 4096
LORA_RANK        = 64
LORA_ALPHA       = 64
LORA_DROPOUT     = 0.0
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

# Rollout — MAX_NEW_TOKENS must match reward_functions.py MAX_NEW_TOKENS
NUM_GENERATIONS  = 8
MAX_NEW_TOKENS   = 512    # room for brief <think> trace + \boxed{X}
ROLLOUT_TEMP     = 1.0    # high T needed — sft_v3 is confident on binary MCQ
ROLLOUT_TOP_P    = 0.95
ROLLOUT_TOP_K    = 20

# Optimisation
PER_DEVICE_BATCH  = 1
GRAD_ACCUM_STEPS  = 4
NUM_TRAIN_EPOCHS  = 3
LEARNING_RATE     = 5e-6
WARMUP_RATIO      = 0.05
CLIP_EPSILON      = 0.2
KL_COEFF          = 0.0   # GSPO / DAPO: no KL penalty
MAX_GRAD_NORM     = 0.5

LOGGING_STEPS     = 5
SAVE_STEPS        = 50

SYSTEM_PROMPT = (
    "You are a safety-aware AI assistant. Given a user prompt and two AI responses (A and B), "
    "reason through which response is safer, then state your final answer as \\boxed{A} or \\boxed{B}."
)

# ===========================================================================
# LOGGING
# ===========================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(OUTPUT_DIR, "train_gspo.log"), encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)

# ===========================================================================
# DATA
# ===========================================================================

def load_training_data() -> list[dict]:
    """Load filtered JSONL, keep only active-label examples."""
    log.info(f"Loading filtered dataset: {FILTERED_JSONL}")
    rows, label_counts = [], {}

    with open(FILTERED_JSONL, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            lbl  = item.get("label", "")
            if lbl not in ACTIVE_LABELS:
                continue
            rows.append({"problem": item["problem"], "answer": item["answer"]})
            label_counts[lbl] = label_counts.get(lbl, 0) + 1

    log.info(f"  total kept: {len(rows)}")
    for lbl, cnt in sorted(label_counts.items()):
        log.info(f"    {lbl}: {cnt}")
    return rows


def build_dataset(rows: list[dict], tokenizer) -> Dataset:
    """
    Build HF Dataset with a "prompt" column.
    GRPOTrainer passes extra columns (e.g. "answer") as kwargs to reward fns.
    """
    formatted = []
    for row in rows:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": row["problem"]},
        ]
        formatted.append({
            "prompt":  tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,
            ),
            "answer":  row["answer"],
            "problem": row["problem"],  # kept for debugging
        })
    return Dataset.from_list(formatted)

# ===========================================================================
# MODEL
# ===========================================================================

def load_model_and_tokenizer():
    """Load SFT v3 merged model and attach a fresh LoRA adapter."""
    log.info(f"Loading base model: {SFT_MODEL_PATH}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name             = SFT_MODEL_PATH,
        max_seq_length         = MAX_SEQ_LENGTH,
        load_in_4bit           = False,
        load_in_8bit           = False,
        fast_inference         = False,          # vLLM causes FP16/BF16 dtype mismatch in matmul_lora
        dtype                  = torch.bfloat16,
    )

    # With fast_inference=True, Unsloth may load the training-path model in FP16
    # even when dtype=bfloat16 is requested.  LoRA matrices initialise in BF16
    # on A100, so addmm_ in matmul_lora raises "Half vs BFloat16".
    # Fix: cast FP16 → BF16 BEFORE LoRA init (params) and AFTER (params + buffers).
    def _cast_fp16_to_bf16(m: torch.nn.Module, label: str) -> int:
        count = 0
        for p in m.parameters():
            if p.dtype == torch.float16:
                p.data = p.data.to(torch.bfloat16)
                count += 1
        for b in m.buffers():
            if b.dtype == torch.float16:
                b.data = b.data.to(torch.bfloat16)
                count += 1
        if count:
            log.info(f"  [{label}] Cast {count} FP16 tensors → BF16")
        return count

    _cast_fp16_to_bf16(model, "pre-LoRA")

    log.info("Attaching LoRA adapter …")
    model = FastLanguageModel.get_peft_model(
        model,
        r                          = LORA_RANK,
        lora_alpha                 = LORA_ALPHA,
        lora_dropout               = LORA_DROPOUT,
        target_modules             = LORA_TARGET_MODULES,
        use_gradient_checkpointing = "unsloth",
        random_state               = 42,
    )

    # Re-cast after PEFT wrapping — Unsloth may reinitialise weight views in FP16.
    _cast_fp16_to_bf16(model, "post-LoRA")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    log.info(f"  trainable: {trainable:,}  ({trainable/total*100:.1f}% of {total:,})")
    return model, tokenizer

# ===========================================================================
# TRAINING ARGS
# ===========================================================================

def build_training_args() -> GRPOConfig:
    return GRPOConfig(
        output_dir  = OUTPUT_DIR,
        run_name    = "safety_gspo_v5",

        # ── GSPO: sequence-level importance sampling ──────────────
        importance_sampling_level     = "sequence",

        # ── rollout ───────────────────────────────────────────────
        num_generations               = NUM_GENERATIONS,
        max_completion_length         = MAX_NEW_TOKENS,
        temperature                   = ROLLOUT_TEMP,
        top_p                         = ROLLOUT_TOP_P,
        top_k                         = ROLLOUT_TOP_K,

        # ── optimisation ──────────────────────────────────────────
        per_device_train_batch_size   = PER_DEVICE_BATCH,
        gradient_accumulation_steps   = GRAD_ACCUM_STEPS,
        num_train_epochs              = NUM_TRAIN_EPOCHS,
        learning_rate                 = LEARNING_RATE,
        lr_scheduler_type             = "cosine",
        warmup_ratio                  = WARMUP_RATIO,
        optim                         = "adamw_8bit",
        max_grad_norm                 = MAX_GRAD_NORM,

        # ── GSPO clipping ─────────────────────────────────────────
        epsilon                       = CLIP_EPSILON,
        beta                          = KL_COEFF,

        # ── precision ─────────────────────────────────────────────
        bf16                          = torch.cuda.is_bf16_supported(),
        fp16                          = not torch.cuda.is_bf16_supported(),

        # ── Unsloth long-context kernels ──────────────────────────
        unsloth_grpo_mini_batch        = 2,
        unsloth_logit_chunk_multiplier = 2,

        # ── checkpointing ─────────────────────────────────────────
        logging_steps                 = LOGGING_STEPS,
        save_steps                    = SAVE_STEPS,
        save_total_limit              = 3,
        report_to                     = "none",
        remove_unused_columns         = False,   # keep "answer" for reward fns
        seed                          = 42,
    )

# ===========================================================================
# MAIN
# ===========================================================================

def main():
    log.info(describe_reward_design())

    rows             = load_training_data()
    model, tokenizer = load_model_and_tokenizer()
    train_dataset    = build_dataset(rows, tokenizer)
    training_args    = build_training_args()

    log.info(f"Training on {len(train_dataset)} samples")
    log.info(
        f"Effective batch: {PER_DEVICE_BATCH} × {GRAD_ACCUM_STEPS} prompts, "
        f"{NUM_GENERATIONS} rollouts each"
    )

    trainer = GRPOTrainer(
        model            = model,
        processing_class = tokenizer,
        args             = training_args,
        train_dataset    = train_dataset,
        reward_funcs     = [
            correctness_reward,  # +2.0 correct, 0.0 wrong
            boxed_penalty,       # -1.5 missing boxed (non-trunc), -0.5 truncated
        ],
    )

    trainer.train()

    save_path = os.path.join(OUTPUT_DIR, "final")
    log.info(f"Saving LoRA adapter → {save_path}")
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    log.info("Done.")
    log.info(
        f"\nNext steps:\n"
        f"  1. Merge  : python merge_lora.py "
        f"--adapter {save_path} --base {SFT_MODEL_PATH} "
        f"--output /scratch/safety/grpo_v5/merged\n"
        f"  2. Validate: python /scratch/safety/shared/test_vllm_base.py "
        f"--model /scratch/safety/grpo_v5/merged "
        f"--validation-file /scratch/standard-project-m2-ma-que/validation_samples/safety.jsonl\n"
        f"  3. Push if better than sft_v3"
    )


if __name__ == "__main__":
    main()
