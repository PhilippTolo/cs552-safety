"""
reward_functions.py
===================
Reward functions for GSPO safety MCQ training.

Design (two separate functions — GRPOTrainer sums them):
  1. correctness_reward — primary signal (+2.0 correct / 0.0 wrong)
  2. boxed_penalty      — penalise missing \\boxed{}, with truncation guard

No length reward: unlike math, we want concise MCQ answers, not longer chains.
The model should reason briefly in <think>...</think> then commit to \\boxed{X}.

References:
  GRPO-LEAD (Zhang & Zuo, EMNLP 2025) — explicit penalty design
  DRPO (arxiv 2510.04474)              — keep length separate from correctness
"""

from __future__ import annotations

import re
from typing import Optional

# ===========================================================================
# CONSTANTS
# ===========================================================================

REWARD_CORRECT    =  2.0   # correct \\boxed{X} answer
REWARD_WRONG      =  0.0   # wrong or missing answer
PENALTY_NO_BOXED  = -1.5   # non-truncated completion with no \\boxed{}
PENALTY_TRUNCATED = -0.5   # truncated (hit token limit before writing \\boxed{})

# Must match MAX_NEW_TOKENS in safety_gspo_train.py
MAX_NEW_TOKENS    = 512
TRUNCATION_RATIO  = 0.98
# Character-to-token proxy for Qwen3 safety completions (~3.5 chars/token)
CHAR_LIMIT_PROXY  = int(MAX_NEW_TOKENS * TRUNCATION_RATIO * 3.5)

# ===========================================================================
# HELPERS
# ===========================================================================

def _completion_text(completion) -> str:
    """Normalise completion to plain string.

    TRL < 0.15 passes completions as strings; TRL >= 0.15 passes them as
    messages dicts [{"role": "assistant", "content": "..."}].  Both forms
    are handled here so reward functions work across versions.
    """
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list):
        return " ".join(
            m.get("content", "") for m in completion if isinstance(m, dict)
        )
    return str(completion)


def _extract_answer(completion) -> Optional[str]:
    """Extract uppercase letter from last \\boxed{X} in completion."""
    text = _completion_text(completion)
    matches = re.findall(r"\\boxed\{+([^}]+?)\}+", text, re.IGNORECASE)
    return matches[-1].strip().upper() if matches else None

# ===========================================================================
# REWARD FUNCTION 1 — Correctness
# ===========================================================================

def correctness_reward(
    prompts:     list[str],
    completions: list[str],
    answer:      list[str],   # gold letters from dataset "answer" column
    **kwargs,
) -> list[float]:
    """
    Primary reward: +2.0 for correct \\boxed{X}, 0.0 otherwise.

    Kept pure — no length component — to avoid the DRPO problem where
    length penalties can push correct-but-long rollouts below zero advantage.
    """
    rewards = []
    for completion, ref in zip(completions, answer):
        pred = _extract_answer(completion)
        correct = pred is not None and pred == ref.strip().upper()
        rewards.append(REWARD_CORRECT if correct else REWARD_WRONG)
    return rewards

# ===========================================================================
# REWARD FUNCTION 2 — Boxed penalty (with truncation guard)
# ===========================================================================

def boxed_penalty(
    prompts:     list[str],
    completions: list[str],
    **kwargs,
) -> list[float]:
    """
    Penalise completions missing \\boxed{}.

    Two cases to avoid double-penalising:
      A — TRUNCATED (hit token limit before writing \\boxed{}):
          Soft penalty -0.5; correctness_reward already gives 0.
      B — NON-TRUNCATED but no \\boxed{}:
          Hard penalty -1.5; model had room to write it and chose not to.

    If \\boxed{} is present: 0.0 regardless.
    """
    rewards = []
    for completion in completions:
        text      = _completion_text(completion)
        has_boxed = _extract_answer(text) is not None
        is_trunc  = len(text) >= CHAR_LIMIT_PROXY
        if has_boxed:
            rewards.append(0.0)
        elif is_trunc:
            rewards.append(PENALTY_TRUNCATED)
        else:
            rewards.append(PENALTY_NO_BOXED)
    return rewards

# ===========================================================================
# SUMMARY (for startup logging)
# ===========================================================================

def describe_reward_design() -> str:
    return """
Reward Design — Safety MCQ
===========================
Function            | Correct+boxed | Truncated | Wrong+boxed | Wrong,no boxed
--------------------|---------------|-----------|-------------|----------------
correctness_reward  |    +2.0       |    0.0    |    0.0      |    0.0
boxed_penalty       |     0.0       |   -0.5    |    0.0      |   -1.5
--------------------|---------------|-----------|-------------|----------------
TOTAL (typical)     |    +2.0       |   -0.5    |    0.0      |   -1.5

No length reward — concise MCQ answers preferred over longer chains.
"""
