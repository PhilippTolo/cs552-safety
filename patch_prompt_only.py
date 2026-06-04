import json, re, pathlib

OUT = "/scratch/safety/prompt_only"

SYS = (
    "You are a careful reasoning assistant. Think step by step before answering. "
    "Your final answer must appear exactly once inside \\boxed{...}. "
    "For multiple-choice questions, output only the correct capital letter "
    "inside \\boxed{...}. Never output multiple boxed answers. "
    "Never leave the final answer unboxed."
)

p = pathlib.Path(OUT + "/chat_template.jinja")
if p.exists():
    t = p.read_text()
else:
    tc = json.load(open(OUT + "/tokenizer_config.json"))
    t = tc.get("chat_template", "")
    if not t:
        raise RuntimeError("No chat_template in tokenizer_config.json")
    print("Extracted template from tokenizer_config.json")
t = re.sub(r"\{%-?\s*set enable_thinking\s*=\s*false\s*-?%\}\s*\n?", "", t)

e = SYS.replace("\\", "\\\\").replace("'", "\\'")
inj = (
    "{%- if not messages or messages[0]['role'] != 'system' %}\n"
    "{%- set messages = [{'role': 'system', 'content': '" + e + "'}] + messages %}\n"
    "{%- endif %}\n"
)
p.write_text(inj + t)
print("Template written. thinking=false present:", "enable_thinking = false" in t)

cfg = {
    "bos_token_id": 151643,
    "do_sample": True,
    "eos_token_id": [151645, 151643],
    "pad_token_id": 151643,
    "temperature": 0.7,
    "top_k": 50,
    "top_p": 0.95
}
json.dump(cfg, open(OUT + "/generation_config.json", "w"), indent=2)
print("gen_config written")
