"""
Diagnostic: what does Qwen3's chat template inject before assistant content
when enable_thinking=True and add_generation_prompt=False?

Run on cluster:
    python /scratch/safety/safety/training/diag_template.py
"""

from transformers import AutoTokenizer

MODEL = "/scratch/safety/sft_v3/merged"

tok = AutoTokenizer.from_pretrained(MODEL)

msgs_full = [
    {"role": "system", "content": "SYS"},
    {"role": "user",   "content": "Q"},
    {"role": "assistant", "content": "ANS"},
]

# Full conversation (add_generation_prompt=False) — used in sft_train.py label masking
full = tok.apply_chat_template(
    msgs_full,
    enable_thinking=True,
    tokenize=False,
    add_generation_prompt=False,
)

# Prompt-only (add_generation_prompt=True) — used to find prompt boundary
prompt = tok.apply_chat_template(
    msgs_full[:-1],
    enable_thinking=True,
    tokenize=False,
    add_generation_prompt=True,
)

assistant_section = full[len(prompt):]

print("=== prompt tail (last 80 chars) ===")
print(repr(prompt[-80:]))
print()
print("=== assistant section ===")
print(repr(assistant_section))
print()
print(f"prompt ends with <think>: {prompt.rstrip().endswith('<think>')}")
print(f"assistant_section starts with <think>: {assistant_section.lstrip().startswith('<think>')}")
print(f"assistant_section contains <think> before ANS: {'<think>' in assistant_section.split('ANS')[0]}")
print()

# Also check prompt-only token count vs full token count difference
prompt_ids = tok.apply_chat_template(
    msgs_full[:-1],
    enable_thinking=True,
    tokenize=True,
    add_generation_prompt=True,
)
full_ids = tok.apply_chat_template(
    msgs_full,
    enable_thinking=True,
    tokenize=True,
    add_generation_prompt=False,
)
ans_ids = tok.encode("ANS", add_special_tokens=False)
print(f"prompt token count : {len(prompt_ids)}")
print(f"full token count   : {len(full_ids)}")
print(f"delta (label tokens): {len(full_ids) - len(prompt_ids)}")
print(f"'ANS' encodes to   : {ans_ids} ({len(ans_ids)} token(s))")
