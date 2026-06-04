import json, re, pathlib

OUT = "/scratch/safety/prompt_only"
SRC = ("/scratch/hf_cache/hub/models--Qwen--Qwen3-1.7B"
       "/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e")

SYS = (
    "You are a careful reasoning assistant. Think step by step before answering. "
    "Your final answer must appear exactly once inside \\boxed{...}. "
    "For multiple-choice questions, output only the correct capital letter "
    "inside \\boxed{...}. Never output multiple boxed answers. "
    "Never leave the final answer unboxed."
)


def get_template():
    for d in [OUT, SRC]:
        jinja = pathlib.Path(d + "/chat_template.jinja")
        if jinja.exists():
            print(f"  template source: {jinja}")
            return jinja.read_text()
        tc_path = pathlib.Path(d + "/tokenizer_config.json")
        if tc_path.exists():
            try:
                tc = json.load(open(tc_path))
                t = tc.get("chat_template", "")
                if t:
                    print(f"  template source: {tc_path} (embedded)")
                    return t
            except Exception as ex:
                print(f"  could not read {tc_path}: {ex}")
    raise RuntimeError("Cannot find chat_template in OUT or SRC")


t = get_template()
t = re.sub(r"\{%-?\s*set enable_thinking\s*=\s*false\s*-?%\}\s*\n?", "", t)

e = SYS.replace("\\", "\\\\").replace("'", "\\'")
inj = (
    "{%- if not messages or messages[0]['role'] != 'system' %}\n"
    "{%- set messages = [{'role': 'system', 'content': '" + e + "'}] + messages %}\n"
    "{%- endif %}\n"
)
pathlib.Path(OUT + "/chat_template.jinja").write_text(inj + t)
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
