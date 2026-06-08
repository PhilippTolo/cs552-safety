"""reward_fns.py — GRPO rewards for safety multiple-choice.

Two signals only — **format** and **correctness**. No length / truncation
reward (by design: the safety MCQ task wants a short, decisive boxed letter,
matching the course CI which extracts a single letter from \\boxed{}).

TRL GRPOTrainer signature (v0.29+):  fn(completions, **kwargs) -> list[float]
Prompt-level dataset columns (e.g. `answer`) are forwarded as kwargs, repeated
once per generation so they align positionally with `completions`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from extract_answer import extract_boxed_answer, normalize_letter

if TYPE_CHECKING:
    from omegaconf import DictConfig


def _text(completion) -> str:
    """Normalise a completion to plain text regardless of TRL format."""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list) and completion:
        return completion[0].get("content", "")
    return ""


def make_reward_fns(cfg: DictConfig):
    """Return (format_reward, correctness_reward)."""
    r = cfg.reward

    def format_reward(completions: list, **kwargs) -> list[float]:
        """+r.format if the completion contains a well-formed \\boxed{<letter>},
        else r.format_penalty. Independent of whether the letter is correct."""
        out = []
        for c in completions:
            letter = normalize_letter(extract_boxed_answer(_text(c)))
            out.append(float(r.format) if letter is not None else float(r.format_penalty))
        return out

    def correctness_reward(completions: list, answer: list[str], **kwargs) -> list[float]:
        """+r.correct if the boxed letter matches the gold option letter, else 0.

        `answer` arrives already normalised to a bare gold letter by format_data."""
        out = []
        for c, gold in zip(completions, answer):
            pred = normalize_letter(extract_boxed_answer(_text(c)))
            gold_l = normalize_letter(gold)
            out.append(float(r.correct) if (pred is not None and pred == gold_l) else 0.0)
        return out

    return format_reward, correctness_reward
