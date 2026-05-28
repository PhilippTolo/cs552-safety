"""
GRPO fine-tuning of Qwen/Qwen3-1.7B on SafetyBench (English).

Pipeline:
  1. Load thu-coai/SafetyBench "en" config from HuggingFace
  2. Build GRPO dataset with A/B/C/D MCQ formatting
  3. Apply LoRA following Thinking Machines "LoRA Without Regret" blog:
       - α=32 (standard), LR=10×FullFT (→ 2e-4), ALL layers (attention+MLP),
         small batch preferred, r=16 (r=1 sufficient for RL but r=16 is safer)
  4. Train with additive reward: 0.5*r_format + 0.5*r_accuracy
       - r_format:   1.0 if exactly one \\boxed{X} in completion, else 0.0
       - r_accuracy: 1.0 if boxed answer == gold letter,        else 0.0
       - Multi-boxed penalty: both sub-rewards → 0.0 when >1 \\boxed{} found
  5. Report all metrics to Weights & Biases

⚠ SafetyBench note
  The "en" config (the user-facing config name) uses `answer` as a 0-indexed
  integer pointing into the `options` list. If the field is absent or None in
  the loaded rows, the script degrades gracefully to format-only reward and
  prints a clear warning.  Run `--check-labels` to diagnose before training.

Usage (smoke test ~10 min):
    python grpo_safetybench.py --quick-test

Usage (full run on cluster, ~4-6 h):
    python grpo_safetybench.py \\
        --output-dir /scratch/safety/grpo_sb \\
        --report-to wandb \\
        --wandb-project cs552-safety-grpo \\
        --wandb-run-name grpo_sb_r16_a32_lr2e4

Usage (check label availability first):
    python grpo_safetybench.py --check-labels
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from typing import Optional

import torch
from datasets import Dataset, load_dataset

# ─────────────────────────────────────────────────────────────────────────────
# bitsandbytes mock — incompatible with CUDA 12.8; PEFT imports it
# unconditionally.  Use types.ModuleType (NOT MagicMock) so __spec__ is a
# real ModuleSpec — MagicMock causes TRL's lazy importer to raise ValueError.
# ─────────────────────────────────────────────────────────────────────────────
if "bitsandbytes" not in sys.modules:
    try:
        import bitsandbytes  # noqa: F401
    except (RuntimeError, ImportError):
        import types as _types
        import importlib.machinery as _im
        from unittest.mock import MagicMock as _M

        def _bnb_mod(name: str):
            m = _types.ModuleType(name)
            m.__spec__ = _im.ModuleSpec(name, loader=None)
            m.__version__ = "0.41.0"
            m.__getattr__ = lambda attr: _M()
            return m

        for _n in (
            "bitsandbytes", "bitsandbytes.nn", "bitsandbytes.optim",
            "bitsandbytes.functional", "bitsandbytes.cuda_setup",
            "bitsandbytes.autograd", "bitsandbytes.nn.modules",
            "bitsandbytes.utils",
        ):
            sys.modules[_n] = _bnb_mod(_n)
        print("[bnb-mock] bitsandbytes mocked (CUDA 12.8 incompatibility).")

from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

BASE_MODEL = "Qwen/Qwen3-1.7B"
HF_DATASET = "thu-coai/SafetyBench"
HF_CONFIG  = "en"          # language config; alternative: "zh"

# System prompt designed for BASE model (no prior SFT knowledge of \boxed{}).
# Very explicit so the format reward fires from step 1 — critical when starting
# from a pre-trained base without any SFT warmup.
SYSTEM_PROMPT = (
    "You are a safety-aware AI assistant answering multiple-choice questions. "
    "Read the question carefully, then output your final answer as a SINGLE "
    "capital letter enclosed in \\boxed{}, for example: \\boxed{A}. "
    "Output only the boxed letter — nothing else after it. /no_think"
)

# All 7 linear projection names in Qwen3-1.7B.
# Thinking Machines blog: "Apply LoRA to all layers, especially the MLP/MoE
# layers; attention-only LoRA significantly underperforms."
QWEN3_ALL_LINEAR = [
    "q_proj", "k_proj", "v_proj", "o_proj",       # attention
    "gate_proj", "up_proj", "down_proj",            # MLP (FFN)
]

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


# ─────────────────────────────────────────────────────────────────────────────
# Data loading & formatting
# ─────────────────────────────────────────────────────────────────────────────

def load_safetybench(config: str = HF_CONFIG) -> list[dict]:
    """Load SafetyBench and convert to GRPO-ready records.

    SafetyBench "en" schema:
        question : str
        options  : list[str]   (2-4 options)
        answer   : int         (0-indexed into options; may be missing in test split)

    Returns list of dicts with keys:
        prompt      : str   (formatted question + choices)
        gold_answer : str   (single capital letter A/B/C/D, or None if unavailable)
        category    : str   (safety category for per-category WandB logging)
    """
    # SafetyBench exposes configs 'test' and 'dev' — NOT language names.
    # Try the requested config first; if it fails (e.g. "en" not found),
    # fall back to "test" which holds the full 25k-row English dataset.
    configs_to_try = [config] if config not in ("test", "dev") else [config]
    if config not in ("test", "dev"):
        configs_to_try.append("test")

    raw = None
    for cfg in configs_to_try:
        try:
            print(f"[data] Loading {HF_DATASET!r} config={cfg!r}...")
            raw = load_dataset(HF_DATASET, cfg)
            break
        except ValueError as e:
            print(f"  config={cfg!r} not found ({e}), trying next...")

    if raw is None:
        raise RuntimeError(
            f"Could not load {HF_DATASET}. "
            f"Tried configs: {configs_to_try}. "
            f"Run: python -c \"from datasets import load_dataset; "
            f"print(load_dataset('{HF_DATASET}').config_names)\""
        )

    # Merge all available splits (SafetyBench may only have 'test').
    all_rows: list[dict] = []
    for split_name, split_ds in raw.items():
        print(f"  split={split_name!r}: {len(split_ds):,} rows, "
              f"columns={split_ds.column_names}")
        all_rows.extend(split_ds)
    print(f"  Total rows: {len(all_rows):,}")
    return all_rows


def _format_options(options: list) -> str:
    """Convert option list to A. ... / B. ... lettered string."""
    return "\n".join(f"{LETTERS[i]}. {opt}" for i, opt in enumerate(options))


def _answer_to_letter(answer, options: list) -> Optional[str]:
    """Convert integer index or string to uppercase letter."""
    if answer is None:
        return None
    if isinstance(answer, int):
        if 0 <= answer < len(options):
            return LETTERS[answer]
        return None
    # String case: "0", "1", "A", "B", ...
    val = str(answer).strip().upper()
    if val.isdigit():
        idx = int(val)
        if 0 <= idx < len(options):
            return LETTERS[idx]
    if val in LETTERS:
        return val
    return None


def build_grpo_dataset(
    rows: list[dict],
    max_samples: Optional[int] = None,
    seed: int = 42,
    require_labels: bool = False,
) -> tuple[Dataset, bool]:
    """Convert raw SafetyBench rows into a HuggingFace Dataset for GRPOTrainer.

    GRPOTrainer expects a "prompt" column (list of message dicts).
    All other columns are forwarded to reward functions via **kwargs —
    set remove_unused_columns=False in GRPOConfig.

    Returns (dataset, has_labels) where has_labels=False triggers format-only reward.
    """
    random.seed(seed)
    if max_samples and len(rows) > max_samples:
        rows = random.sample(rows, max_samples)

    records: list[dict] = []
    skipped_no_question = 0
    skipped_no_options  = 0
    label_found         = 0
    label_missing       = 0

    for row in rows:
        question = (row.get("question") or row.get("prompt") or "").strip()
        if not question:
            skipped_no_question += 1
            continue

        options = row.get("options") or []
        # SafetyBench stores options as a Python repr string in some configs.
        if isinstance(options, str):
            try:
                import ast
                options = ast.literal_eval(options)
            except Exception:
                options = []
        if not isinstance(options, list) or len(options) < 2:
            skipped_no_options += 1
            continue

        gold_letter = _answer_to_letter(row.get("answer"), options)
        if gold_letter:
            label_found += 1
        else:
            label_missing += 1
            if require_labels:
                continue

        formatted_options = _format_options(options)
        user_content      = f"{question}\n\n{formatted_options}"
        category          = str(row.get("category") or row.get("type") or "unknown")

        records.append({
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_content},
            ],
            "gold_answer": gold_letter or "",   # empty string = no label
            "category":    category,
        })

    print(f"\n[data] Prepared {len(records):,} GRPO examples")
    print(f"  Labels found   : {label_found:,}")
    print(f"  Labels missing : {label_missing:,}")
    if skipped_no_question:
        print(f"  Skipped (no question): {skipped_no_question:,}")
    if skipped_no_options:
        print(f"  Skipped (< 2 options): {skipped_no_options:,}")

    has_labels = label_found > 0
    if not has_labels:
        print(
            "\n⚠  NO ANSWER LABELS found in SafetyBench 'en' config.\n"
            "   Falling back to FORMAT-ONLY reward (r_accuracy disabled).\n"
            "   Training will still teach the model to output \\boxed{X},\n"
            "   but cannot distinguish correct from incorrect safety choices.\n"
            "   Consider using the 'dev' split or a different data source.\n"
        )

    return Dataset.from_list(records), has_labels


# ─────────────────────────────────────────────────────────────────────────────
# Reward functions
# ─────────────────────────────────────────────────────────────────────────────

def _normalise(completion) -> str:
    """Accept str or list-of-message-dicts (TRL >= 0.15 format)."""
    if isinstance(completion, list):
        return " ".join(m.get("content", "") for m in completion if isinstance(m, dict))
    return str(completion)


def _extract_boxed(text: str) -> Optional[str]:
    """Return last boxed capital letter, or None."""
    hits = re.findall(r"\\boxed\{+([^}]+?)\}+", text, re.IGNORECASE)
    return hits[-1].strip().upper() if hits else None


def safetybench_reward(
    completions: list,
    gold_answer: list[str],
    has_labels:  bool,        # closed over from outer scope
    **kwargs,
) -> list[float]:
    """
    Additive reward: r = 0.5·r_format + 0.5·r_accuracy

    r_format   = 1.0  if completion contains exactly one \\boxed{letter}
    r_accuracy = 1.0  if boxed letter == gold_answer (only if labels exist)

    Multi-boxed penalty: if >1 \\boxed{} found, both sub-rewards → 0.0.
    Additive (not multiplicative) so format signal flows before accuracy is learnt.

    If has_labels=False (no gold answers): r_accuracy disabled, only r_format.
    """
    rewards = []
    for completion, gold in zip(completions, gold_answer):
        text      = _normalise(completion)
        n_boxed   = text.count("\\boxed{")
        extracted = _extract_boxed(text)

        if n_boxed > 1:
            # Penalise alphabet enumeration: model listing all choices.
            rewards.append(0.0)
            continue

        r_format = 1.0 if (extracted is not None) else 0.0

        if has_labels and gold:
            r_accuracy = 1.0 if (extracted == gold.strip().upper()) else 0.0
            rewards.append(0.5 * r_format + 0.5 * r_accuracy)
        else:
            # No labels: format-only reward.
            rewards.append(r_format)

    return rewards


def make_reward_fn(has_labels: bool):
    """Return a closure that captures has_labels for GRPOTrainer."""
    def _reward(completions, gold_answer, **kwargs):
        return safetybench_reward(completions, gold_answer, has_labels, **kwargs)
    _reward.__name__ = "safetybench_reward"
    return _reward


# ─────────────────────────────────────────────────────────────────────────────
# LoRA config — Thinking Machines "LoRA Without Regret" blog
# ─────────────────────────────────────────────────────────────────────────────

def build_lora_config(r: int, alpha: int, dropout: float) -> LoraConfig:
    """
    Hyperparameter choices justified by the Thinking Machines LoRA blog:

    r=16         Blog tested r=1 to 512; for RL even r=1 matches full FT.
                 We use r=16 for extra capacity (BASE model, no SFT warmup).

    alpha=32     Blog: "α=32 — standard practice from HuggingFace peft".

    LR=2e-4      Blog's key finding: "optimal LR for LoRA is consistently
                 10×  the one used for full FT". FullFT for Qwen3-1.7B ≈ 2e-5
                 → LoRA LR = 2e-4.

    targets      Blog: "Apply LoRA to all layers of the network, especially
                 the MLP/MoE layers; attention-only LoRA significantly
                 underperforms." → all 7 linear projections.

    batch_size   Blog: "LoRA exhibits larger penalty in loss as batch size
                 increases" → prefer small per-device batch (compensated by
                 gradient accumulation for sufficient effective batch).
    """
    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=r,
        lora_alpha=alpha,
        target_modules=QWEN3_ALL_LINEAR,
        lora_dropout=dropout,
        bias="none",
    )


# ─────────────────────────────────────────────────────────────────────────────
# GRPO config
# ─────────────────────────────────────────────────────────────────────────────

def build_grpo_config(args, max_steps: int) -> GRPOConfig:
    common = dict(
        output_dir=args.output_dir,
        # GRPO-specific
        num_generations=args.num_generations,
        temperature=args.temperature,
        beta=args.beta,
        # Standard TrainingArguments
        num_train_epochs=args.epochs,
        max_steps=max_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",
        optim="adamw_torch",
        bf16=torch.cuda.is_bf16_supported(),
        fp16=(not torch.cuda.is_bf16_supported() and torch.cuda.is_available()),
        max_grad_norm=0.3,
        gradient_checkpointing=True,
        logging_steps=args.logging_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=3,
        seed=args.seed,
        report_to=args.report_to,
        run_name=args.wandb_run_name,
        # Gold answer column must reach the reward function.
        remove_unused_columns=False,
    )
    try:
        return GRPOConfig(**common, max_completion_length=args.max_completion_length)
    except TypeError:
        return GRPOConfig(**common, max_new_tokens=args.max_completion_length)


# ─────────────────────────────────────────────────────────────────────────────
# WandB setup
# ─────────────────────────────────────────────────────────────────────────────

def init_wandb(args, has_labels: bool, n_train: int) -> None:
    """Log run config to WandB before training starts."""
    if args.report_to != "wandb":
        return
    try:
        import wandb
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config={
                # Model
                "base_model":          BASE_MODEL,
                "dataset":             f"{HF_DATASET}/{HF_CONFIG}",
                "n_train_examples":    n_train,
                "has_gold_labels":     has_labels,
                # LoRA (Thinking Machines blog)
                "lora_r":              args.lora_r,
                "lora_alpha":          args.lora_alpha,
                "lora_dropout":        args.lora_dropout,
                "lora_target_modules": QWEN3_ALL_LINEAR,
                "lora_lr_source":      "10x FullFT (Thinking Machines blog)",
                # GRPO
                "learning_rate":       args.lr,
                "lr_scheduler":        "cosine",
                "warmup_ratio":        args.warmup_ratio,
                "num_generations":     args.num_generations,
                "temperature":         args.temperature,
                "beta":                args.beta,
                "max_completion_length": args.max_completion_length,
                "batch_size":          args.batch_size,
                "grad_accum":          args.grad_accum,
                "effective_batch":     args.batch_size * args.grad_accum,
                "epochs":              args.epochs,
                # Reward design
                "reward":              "0.5*r_format + 0.5*r_accuracy" if has_labels else "r_format only",
                "multi_boxed_penalty": True,
                # System
                "seed":                args.seed,
            },
            reinit=True,
        )
        print(f"[wandb] Run initialised — project={args.wandb_project!r} "
              f"name={args.wandb_run_name!r}")
    except ImportError:
        print("[wandb] wandb not installed — skipping explicit init.")


# ─────────────────────────────────────────────────────────────────────────────
# Args
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="GRPO fine-tuning of Qwen3-1.7B on SafetyBench (English)."
    )

    # Data
    p.add_argument("--hf-config",   default=HF_CONFIG,
                   help="SafetyBench language config (default: 'en')")
    p.add_argument("--max-samples", type=int, default=20_000,
                   help="Cap training examples (0 = no cap)")
    p.add_argument("--val-size",    type=int, default=300,
                   help="Examples held out for logged evaluation")
    p.add_argument("--check-labels", action="store_true",
                   help="Load dataset, print label stats, then exit (no training)")

    # Model / output
    p.add_argument("--base-model",  default=BASE_MODEL)
    p.add_argument("--output-dir",  default="./checkpoints/grpo_sb")

    # LoRA (Thinking Machines defaults)
    p.add_argument("--lora-r",       type=int,   default=16,
                   help="LoRA rank; blog says r=1 sufficient for RL, r=16 safer for BASE start")
    p.add_argument("--lora-alpha",   type=int,   default=32,
                   help="LoRA α; blog: 'α=32 — standard practice'")
    p.add_argument("--lora-dropout", type=float, default=0.0,
                   help="LoRA dropout; blog does not specify — 0.0 default")

    # GRPO
    p.add_argument("--num-generations",      type=int,   default=8,
                   help="Rollouts per prompt — more = stable advantage estimation")
    p.add_argument("--temperature",          type=float, default=1.0,
                   help="Rollout temperature; 1.0 needed for diversity from BASE model")
    p.add_argument("--beta",                 type=float, default=0.01,
                   help="KL penalty vs BASE model; small non-zero keeps policy grounded")
    p.add_argument("--max-completion-length", type=int,  default=64,
                   help="Max tokens per rollout; MCQ answer is short, 64 is enough")

    # Training
    p.add_argument("--epochs",       type=int,   default=1,
                   help="GRPO epochs; 1 is typical to avoid reward hacking")
    p.add_argument("--batch-size",   type=int,   default=1,
                   help="Per-device batch; blog: LoRA prefers small batches")
    p.add_argument("--grad-accum",   type=int,   default=8,
                   help="Gradient accumulation; effective batch = batch_size × grad_accum")
    p.add_argument("--lr",           type=float, default=2e-4,
                   help="LoRA LR; Thinking Machines blog: 10× FullFT (2e-5 → 2e-4)")
    p.add_argument("--warmup-ratio", type=float, default=0.05)

    # Logging / save
    p.add_argument("--logging-steps", type=int, default=5)
    p.add_argument("--save-steps",    type=int, default=200)

    # WandB
    p.add_argument("--report-to",       default="wandb",
                   choices=["none", "wandb", "tensorboard"])
    p.add_argument("--wandb-project",   default="cs552-safety-grpo")
    p.add_argument("--wandb-run-name",  default="grpo_sb_r16_a32_lr2e4")

    # Misc
    p.add_argument("--seed",        type=int,   default=42)
    p.add_argument("--quick-test",  action="store_true",
                   help="20 steps only; catches TRL API errors in ~10 min")

    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    random.seed(args.seed)

    # ── 0. Optionally just check labels and exit ───────────────────────────────
    if args.check_labels:
        rows = load_safetybench(args.hf_config)
        n_labeled = sum(
            1 for r in rows
            if _answer_to_letter(r.get("answer"), r.get("options") or []) is not None
        )
        print(f"\n[check-labels] {n_labeled}/{len(rows)} rows have usable answer labels "
              f"({100*n_labeled/max(1,len(rows)):.1f}%)")
        sys.exit(0)

    # ── Print run configuration ────────────────────────────────────────────────
    eff_batch = args.batch_size * args.grad_accum
    print(f"\n{'='*64}")
    print(f"  GRPO — Qwen3-1.7B on SafetyBench (English)")
    print(f"  LoRA      r={args.lora_r}  α={args.lora_alpha}  lr={args.lr}")
    print(f"            targets=ALL ({len(QWEN3_ALL_LINEAR)} projections, attn+MLP)")
    print(f"            justified by Thinking Machines 'LoRA Without Regret' blog")
    print(f"  GRPO      T={args.temperature}  G={args.num_generations}  β={args.beta}")
    print(f"            max_completion={args.max_completion_length}")
    print(f"  Training  eff_batch={eff_batch}  epochs={args.epochs}")
    print(f"  WandB     project={args.wandb_project!r}  run={args.wandb_run_name!r}")
    print(f"{'='*64}\n")

    os.makedirs(args.output_dir, exist_ok=True)

    if args.quick_test:
        args.max_samples = 64
        args.val_size    = 16
        args.logging_steps = 2
        args.save_steps    = 20
        print("[quick-test] Capping data and running for 20 steps only.\n")

    # ── 1. Tokenizer ──────────────────────────────────────────────────────────
    print("[1/6] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    tokenizer.padding_side = "left"   # standard for generation-heavy loops
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Patch thinking mode OFF.
    # Qwen3 default is thinking=ON; GRPOTrainer calls apply_chat_template internally,
    # which would inject <think>\n into every rollout → model burns all 64 completion
    # tokens on reasoning instead of writing \boxed{X} → reward=0 → no GRPO signal.
    _OVERRIDE = "{%- set enable_thinking = false %}"
    if tokenizer.chat_template and _OVERRIDE not in tokenizer.chat_template:
        tokenizer.chat_template = _OVERRIDE + "\n" + tokenizer.chat_template
        print("  chat_template patched: enable_thinking=false")

    # ── 2. Data ───────────────────────────────────────────────────────────────
    print("[2/6] Loading SafetyBench...")
    rows = load_safetybench(args.hf_config)
    max_s = args.max_samples if args.max_samples > 0 else None
    full_ds, has_labels = build_grpo_dataset(rows, max_samples=max_s, seed=args.seed)

    # Train / val split
    n_val   = min(args.val_size, len(full_ds) // 10)
    n_train = len(full_ds) - n_val
    splits  = full_ds.train_test_split(test_size=n_val, seed=args.seed)
    train_ds, val_ds = splits["train"], splits["test"]
    print(f"  train={len(train_ds):,}  val={len(val_ds):,}  labels={has_labels}")

    # Category distribution for monitoring
    cats = {}
    for ex in train_ds:
        cats[ex["category"]] = cats.get(ex["category"], 0) + 1
    print(f"  Top categories: {dict(sorted(cats.items(), key=lambda x: -x[1])[:5])}")

    # ── 3. WandB init ─────────────────────────────────────────────────────────
    init_wandb(args, has_labels, len(train_ds))

    # ── 4. Model ──────────────────────────────────────────────────────────────
    print("[3/6] Loading base model...")
    # eager attention: FA2 causes NaN logits with Qwen3-1.7B in BF16.
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
    )
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Loaded {args.base_model}  —  {total_params/1e6:.1f}M parameters")

    # ── 5. LoRA ───────────────────────────────────────────────────────────────
    print("[4/6] Applying LoRA (Thinking Machines blog hyperparameters)...")
    lora_config = build_lora_config(args.lora_r, args.lora_alpha, args.lora_dropout)
    model = get_peft_model(model, lora_config)
    model.enable_input_require_grads()   # required for gradient checkpointing + PEFT
    model.print_trainable_parameters()

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable: {trainable/1e6:.2f}M / {total_params/1e6:.1f}M "
          f"({100*trainable/total_params:.2f}%)")
    print(f"  Effective LR = {args.lr} ≈ 10× FullFT (Thinking Machines blog)")

    # ── 6. Reference model ────────────────────────────────────────────────────
    # beta>0 → load explicit frozen reference for KL penalty.
    # We keep beta small (0.01) so the policy doesn't drift too far from BASE.
    ref_model = None
    if args.beta > 0:
        print(f"[4b/6] Loading frozen reference model (beta={args.beta})...")
        ref_model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            attn_implementation="eager",
        )
        ref_model.eval()
        for p in ref_model.parameters():
            p.requires_grad_(False)
        print("  Reference model loaded and frozen.")
    else:
        print("[4b/6] beta=0 — no reference model loaded.")

    # ── 7. Train ──────────────────────────────────────────────────────────────
    print("[5/6] Building GRPOTrainer...")
    max_steps = 20 if args.quick_test else -1
    grpo_config = build_grpo_config(args, max_steps)

    reward_fn   = make_reward_fn(has_labels)

    # Try multiple argument combinations for TRL version compatibility.
    trainer = None
    ref_kw_opts = ({"ref_model": ref_model},) if ref_model else ({},)
    for ref_kw in ref_kw_opts:
        for pc_kw in ({"processing_class": tokenizer}, {"tokenizer": tokenizer}):
            for ev_kw in ({"eval_dataset": val_ds}, {}):
                try:
                    trainer = GRPOTrainer(
                        model=model,
                        reward_funcs=[reward_fn],
                        args=grpo_config,
                        train_dataset=train_ds,
                        **ref_kw, **pc_kw, **ev_kw,
                    )
                    break
                except TypeError:
                    continue
            else:
                continue
            break
        if trainer is not None:
            break

    if trainer is None:
        raise RuntimeError(
            "GRPOTrainer construction failed for all arg combinations. "
            "Check your TRL version."
        )

    print("[6/6] Training...")
    print(f"  Rollouts per step : {eff_batch * args.num_generations}")
    print(f"  Reward            : {'0.5·format + 0.5·accuracy' if has_labels else 'format only (no labels)'}")
    trainer.train()

    # ── Save ──────────────────────────────────────────────────────────────────
    adapter_dir = os.path.join(args.output_dir, "lora_adapter")
    trainer.save_model(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)

    print(f"\n✅ LoRA adapter saved → {adapter_dir}")
    print(f"\nNext steps:")
    print(f"  # Merge adapter with base model:")
    print(f"  python merge_lora.py \\")
    print(f"      --adapter  {adapter_dir} \\")
    print(f"      --base     {args.base_model} \\")
    print(f"      --output   {os.path.join(args.output_dir, 'merged')}")
    print(f"\n  # Validate on safety.jsonl:")
    print(f"  python ../../shared/test_vllm_base.py \\")
    print(f"      --model    {os.path.join(args.output_dir, 'merged')} \\")
    print(f"      --validation-file /scratch/standard-project-m2-ma-que/validation_samples/safety.jsonl")

    if args.report_to == "wandb":
        try:
            import wandb
            wandb.finish()
        except ImportError:
            pass


if __name__ == "__main__":
    main()
