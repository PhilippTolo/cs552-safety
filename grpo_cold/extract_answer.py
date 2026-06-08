"""extract_answer.py — boxed-answer extraction for safety multiple-choice GRPO.

Self-contained (ported from the multilingual pipeline — nested-brace safe).
Provides:
  * extract_boxed_answer  — pull the content of the LAST \\boxed{...}
  * normalize_letter      — reduce that content to a single uppercase option letter

The MC task only needs a letter (A..Z), so we deliberately avoid any heavy
math normalisation. Matches the course CI's letter extraction behaviour.
"""

from __future__ import annotations

import re


# ---------------------------------------------------------------------------
# Boxed extraction (nested-brace safe)
# ---------------------------------------------------------------------------

def last_boxed_only_string(string: str) -> str | None:
    """Return the last ``\\boxed{...}`` / ``\\fbox{...}`` substring (incl. the
    command and braces), handling nested braces. None if absent."""
    idx = string.rfind("\\boxed")
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        return None
    return string[idx : right_brace_idx + 1]


def remove_boxed(s: str) -> str | None:
    for left in ("\\boxed{", "\\fbox{"):
        if s.startswith(left) and s.endswith("}"):
            return s[len(left) : -1]
    return None


def extract_boxed_answer(pred_str: str) -> str | None:
    """Inner content of the last ``\\boxed{}`` in *pred_str*, or None."""
    boxed_str = last_boxed_only_string(pred_str)
    if boxed_str is None:
        return None
    return remove_boxed(boxed_str)


# ---------------------------------------------------------------------------
# Letter normalisation
# ---------------------------------------------------------------------------

_CMD_RE = re.compile(r"\\(?:text|mathrm|mathbf|mathsf|rm|bf|operatorname)\s*")


def normalize_letter(s: str | None) -> str | None:
    """Reduce boxed content (or a gold label) to a single uppercase letter.

    Handles ``C``, ``C)``, ``(C)``, ``\\text{C}``, ``\\boxed{C}`` leftovers, etc.
    Returns None if no letter can be found.
    """
    if s is None:
        return None
    s = _CMD_RE.sub("", s)            # drop \text{...} style wrappers' command name
    s = re.sub(r"[^A-Za-z]", "", s)   # keep letters only
    if not s:
        return None
    return s[0].upper()


def parse_gold(answer: str) -> str | None:
    """Parse a gold answer that may be a bare letter ("B") or a boxed string
    ("\\boxed{B}"). Returns a single uppercase letter or None.

    NOTE: we cannot feed "\\boxed{C}" straight to normalize_letter — it would
    strip to "boxedC" and return 'B' (first letter). So box-form is unwrapped
    first.
    """
    if answer is None:
        return None
    if "\\boxed" in answer or "\\fbox" in answer:
        return normalize_letter(extract_boxed_answer(answer))
    return normalize_letter(answer)
