"""
Patch sft_cot/merged tokenizer to use 4neurons-style system prompt injection.
Run from /scratch/safety after merge_lora.py completes.

Usage:
    python /scratch/safety/safety/patch_cot_template.py
"""
import json, os

MERGED = "/scratch/safety/sft_cot/merged"

# System prompt matching REASONING_SYSTEM_PROMPT from generate_cot_traces.py
SYS = (
    "You are a helpful assistant. "
    "For multiple-choice questions, reason step by step, "
    "then provide your final answer as a single letter inside "
    r"\boxed{}, for example: \boxed{A}. "
    r"Do not include anything after the \boxed{}."
)

# Jinja snippet: inject system message if none provided (4neurons pattern)
INJECT = (
    "{%- if not messages or messages[0]['role'] != 'system' %}\n"
    "{%- set messages = [{\"role\": \"system\", \"content\": "
    + json.dumps(SYS)
    + "}] + messages %}\n"
    "{%- endif %}\n"
)

# Patch tokenizer_config.json
cfg_path = os.path.join(MERGED, "tokenizer_config.json")
with open(cfg_path, encoding="utf-8") as f:
    cfg = json.load(f)

template = cfg.get("chat_template", "")
if "You are a helpful assistant" in template:
    print("Already patched — skipping.")
else:
    cfg["chat_template"] = INJECT + template
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)
    print("Patched tokenizer_config.json")

# Patch chat_template.jinja if it exists (takes precedence over tokenizer_config.json)
jinja_path = os.path.join(MERGED, "chat_template.jinja")
if os.path.exists(jinja_path):
    with open(jinja_path, encoding="utf-8") as f:
        src = f.read()
    if "You are a helpful assistant" in src:
        print("chat_template.jinja already patched — skipping.")
    else:
        with open(jinja_path, "w", encoding="utf-8") as f:
            f.write(INJECT + src)
        print("Patched chat_template.jinja")
else:
    print("No chat_template.jinja found — only tokenizer_config.json patched.")

print("\nVerify with:")
print("  python3 -c \"from transformers import AutoTokenizer; t=AutoTokenizer.from_pretrained('/scratch/safety/sft_cot/merged'); print(t.apply_chat_template([{'role':'user','content':'Which is safer? A) X B) Y'}], tokenize=False, add_generation_prompt=True))\"")
