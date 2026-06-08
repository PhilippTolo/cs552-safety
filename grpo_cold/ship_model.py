"""ship_model.py — merge the GRPO LoRA and bake a CI-ready safety_model.

The course CI calls ONLY ``tokenizer.apply_chat_template(messages,
add_generation_prompt=True)`` with the bare user question — it never passes our
system prompt and never passes ``enable_thinking``. So both must be baked into
the shipped ``chat_template.jinja``:

  1. auto-inject the golden-nugget SYSTEM_PROMPT when no system message is given;
  2. force NON-THINKING (``enable_thinking = false``) so the rendered generation
     prompt matches training exactly (ends with the empty ``<think>\\n\\n</think>``
     block — the context GRPO was trained from; mismatch here is Lessons 38/40).

Steps: merge LoRA → save merged model+tokenizer → write chat_template.jinja +
generation_config.json → verify the rendered prompt. Then push with `hf upload`.

Usage (on cluster):
    PYTHONPATH=/scratch/safety python grpo_cold/ship_model.py \
        --adapter /scratch/safety/grpo_safety_checkpoints/final \
        --base Qwen/Qwen3-1.7B \
        --output /scratch/safety/grpo_cold/merged
    # validate, then:
    hf upload cs-552-2026-ma-que/safety_model /scratch/safety/grpo_cold/merged .
"""

from __future__ import annotations

import argparse
import importlib.machinery
import json
import pathlib
import re
import sys
import types

# bitsandbytes mock (CUDA 12.8) — must precede peft import. See train_grpo.py.
if "bitsandbytes" not in sys.modules:
    _bnb = types.ModuleType("bitsandbytes")
    _bnb.__spec__ = importlib.machinery.ModuleSpec("bitsandbytes", None)
    for _s in ["nn", "optim", "functional", "cextension", "cuda_setup", "utils"]:
        _sub = types.ModuleType(f"bitsandbytes.{_s}")
        _sub.__spec__ = importlib.machinery.ModuleSpec(f"bitsandbytes.{_s}", None)
        sys.modules[f"bitsandbytes.{_s}"] = _sub
        setattr(_bnb, _s, _sub)
    _bnb.cextension.COMPILED_WITH_CUDA = False
    sys.modules["bitsandbytes"] = _bnb

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from format_data import SYSTEM_PROMPT

# Greedy, non-thinking decisive single letter — matches the grpo_v5 floor and the
# strongest thinking=OFF competitors (vibe-trainers, camykaz).
GEN_CONFIG = {
    "bos_token_id": 151643,
    "do_sample": False,
    "eos_token_id": [151645, 151643],
    "pad_token_id": 151643,
    "transformers_version": "4.51.0",
}


def merge(adapter: str, base: str, output: str) -> None:
    from peft import PeftModel

    print(f"[merge] base={base}  adapter={adapter}")
    model = AutoModelForCausalLM.from_pretrained(
        base, torch_dtype=torch.bfloat16, trust_remote_code=True)
    model = PeftModel.from_pretrained(model, adapter)
    model = model.merge_and_unload()
    pathlib.Path(output).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output, safe_serialization=True)
    AutoTokenizer.from_pretrained(base, trust_remote_code=True).save_pretrained(output)
    print(f"[merge] merged model saved -> {output}")


def bake_template(output: str) -> None:
    """Force non-thinking + auto-inject SYSTEM_PROMPT into chat_template.jinja."""
    out = pathlib.Path(output)
    jinja = out / "chat_template.jinja"
    if jinja.exists():
        t = jinja.read_text(encoding="utf-8")
    else:
        tc = json.load(open(out / "tokenizer_config.json", encoding="utf-8"))
        t = tc.get("chat_template", "")
    if not t:
        raise RuntimeError("no chat_template found in merged output")

    # Drop any existing enable_thinking set so our forced value is unambiguous.
    t = re.sub(r"\{%-?\s*set enable_thinking\s*=\s*(?:true|false)\s*-?%\}\s*\n?", "", t)

    # Inject SYSTEM_PROMPT as a DOUBLE-quoted Jinja literal: escape backslash,
    # double-quote, and newline. Apostrophes pass through untouched — Jinja's
    # lexer rejects \' inside a single-quoted literal (the golden nuggets contain
    # possessives like "friend's"), so single-quoting is not an option here.
    if '"' in SYSTEM_PROMPT:
        raise ValueError(
            "SYSTEM_PROMPT contains a double-quote; the double-quoted Jinja "
            "injection can't safely escape it. Remove the \" from format_data.SYSTEM_PROMPT."
        )
    e = SYSTEM_PROMPT.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
    preamble = (
        "{%- set enable_thinking = false %}\n"
        "{%- if not messages or messages[0]['role'] != 'system' %}\n"
        '{%- set messages = [{"role": "system", "content": "' + e + '"}] + messages %}\n'
        "{%- endif %}\n"
    )
    jinja.write_text(preamble + t, encoding="utf-8")
    print(f"[bake] chat_template.jinja written ({len(preamble) + len(t)} chars)")

    json.dump(GEN_CONFIG, open(out / "generation_config.json", "w"), indent=2)
    print("[bake] generation_config.json written (greedy, non-thinking)")


def verify(output: str) -> None:
    tok = AutoTokenizer.from_pretrained(output, trust_remote_code=True)
    rendered = tok.apply_chat_template(
        [{"role": "user", "content": "Is sharing your bank PIN with a stranger safe?\n"
                                     "A) Yes\nB) No"}],
        tokenize=False, add_generation_prompt=True,
    )
    print("\n[verify] --- rendered prompt tail (last 60 chars) ---")
    print(repr(rendered[-60:]))
    checks = {
        "golden nugget present": "Worked examples" in rendered,
        "user question present": "bank PIN" in rendered,
        "non-thinking empty block (ends </think>)": rendered.rstrip().endswith("</think>"),
        "single \\boxed in system": "\\boxed{B}" in rendered,
    }
    print("\n[verify] checks:")
    for k, v in checks.items():
        print(f"  [{'OK ' if v else 'XX '}] {k}")
    if not checks["non-thinking empty block (ends </think>)"]:
        print("  WARNING: prompt does not end with the empty </think> block. "
              "Non-thinking may not be active — training/inference mismatch risk "
              "(Lessons 38/40). Inspect chat_template.jinja before pushing.")
    if not checks["user question present"]:
        raise RuntimeError("user question missing from rendered prompt — template broken "
                           "(possible _fs_ns drop, Lesson 42).")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--adapter", required=True, help="LoRA adapter dir (…/final)")
    p.add_argument("--base", default="Qwen/Qwen3-1.7B")
    p.add_argument("--output", required=True)
    p.add_argument("--skip_merge", action="store_true",
                   help="adapter already merged into --output; only (re)bake template")
    args = p.parse_args()

    if not args.skip_merge:
        merge(args.adapter, args.base, args.output)
    bake_template(args.output)
    verify(args.output)
    print(f"\n[done] ready to push:\n  hf upload cs-552-2026-ma-que/safety_model {args.output} .")


if __name__ == "__main__":
    main()
