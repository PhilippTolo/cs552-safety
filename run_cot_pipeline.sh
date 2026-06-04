#!/usr/bin/env bash
# ============================================================
# run_cot_pipeline.sh
# Full CoT distillation pipeline for safety model
# Run on cluster as:
#   cd /scratch/safety
#   bash /scratch/safety/run_cot_pipeline.sh
# Or step by step (see comments).
# ============================================================
set -euo pipefail

DATA="/scratch/safety/data"
LOGS="/scratch/safety/logs"
CODE="/scratch/safety"

mkdir -p "$LOGS"

# ── STEP 1: SaladBench (direct labels, ~1 min) ────────────────────────────────
echo "============================================================"
echo "STEP 1: Prepare SaladBench (walledai/SaladBench MRQ)"
echo "============================================================"
PYTHONPATH=$CODE python $CODE/safety/data/prepare_salad.py \
    --output-dir "$DATA" \
    --val-frac 0.05 \
    --seed 42

echo ""
echo "SaladBench done. Check yield above."
echo "Expected: 2,000-3,200 rows after placeholder filtering."
echo "If < 300 rows: SaladBench MRQ is mostly AI-preference format — skip it."
echo ""

# ── STEP 2: SafetyBench synthetic (Qwen3-8B, ~2-3 h) ─────────────────────────
echo "============================================================"
echo "STEP 2: SafetyBench synthetic CoT labeling (Qwen3-8B)"
echo "  This takes ~2-3 h. Run in background if needed:"
echo "  PYTHONPATH=$CODE nohup python $CODE/prepare_safetybench_synthetic.py \\"
echo "      > $LOGS/safetybench_synthetic.log 2>&1 &"
echo "============================================================"
PYTHONPATH=$CODE python "$CODE/prepare_safetybench_synthetic.py" \
    2>&1 | tee "$LOGS/safetybench_synthetic.log"

echo ""

# ── STEP 3: Combine datasets ──────────────────────────────────────────────────
echo "============================================================"
echo "STEP 3: Combining SFT datasets"
echo "============================================================"

# Combine train splits (SafetyBench CoT + SaladBench bare)
cat "$DATA/safetybench_sft_train.jsonl" \
    "$DATA/salad_sft_train.jsonl" \
    > "$DATA/combined_sft_train.jsonl"

# Combine val splits
cat "$DATA/safetybench_sft_val.jsonl" \
    "$DATA/salad_sft_val.jsonl" \
    > "$DATA/combined_sft_val.jsonl"

TRAIN_COUNT=$(wc -l < "$DATA/combined_sft_train.jsonl")
VAL_COUNT=$(wc -l < "$DATA/combined_sft_val.jsonl")
echo "  combined_sft_train.jsonl : $TRAIN_COUNT examples"
echo "  combined_sft_val.jsonl   : $VAL_COUNT examples"

# ── STEP 4: Train Qwen3-1.7B BASE with CoT distillation ─────────────────────
echo ""
echo "============================================================"
echo "STEP 4: Training Qwen3-1.7B BASE (thinking=ON, CoT labels)"
echo "  sft_train.py uses enable_thinking=True — no patch needed."
echo "  Expected: ~1-2 h on A100 40GB (3,000-6,000 examples)"
echo "============================================================"
PYTHONPATH=$CODE nohup python "$CODE/safety/training/sft_train.py" \
    --train "$DATA/combined_sft_train.jsonl" \
    --val   "$DATA/combined_sft_val.jsonl" \
    --model "Qwen/Qwen3-1.7B" \
    --output-dir "/scratch/safety/sft_cot_distill" \
    --epochs 2 \
    --lr 2e-5 \
    --lora-r 16 \
    --lora-alpha 32 \
    --batch-size 4 \
    --grad-accum 8 \
    --max-seq-length 1024 \
    --max-grad-norm 0.3 \
    --eval-steps 100 \
    --save-steps 200 \
    > "$LOGS/sft_cot_distill.log" 2>&1 &

SFT_PID=$!
echo "  Training PID: $SFT_PID"
echo "  Monitor: tail -f $LOGS/sft_cot_distill.log"
echo ""
echo "  HEALTH CHECK at step 10:"
echo "    - loss 1.5-4.0: GOOD (CoT traces = 100-500 token labels)"
echo "    - loss < 0.01 at step 10: BROKEN — kill and investigate"
echo "    - loss > 20: BROKEN — kill and investigate"
echo ""
wait $SFT_PID

echo ""
echo "Training complete. Final loss (last 5 lines):"
tail -5 "$LOGS/sft_cot_distill.log"

# ── STEP 5: Merge LoRA ────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "STEP 5: Merging LoRA adapter"
echo "============================================================"
python "$CODE/safety/training/merge_lora.py" \
    --adapter /scratch/safety/sft_cot_distill/lora_adapter \
    --base Qwen/Qwen3-1.7B \
    --output /scratch/safety/sft_cot_distill/merged

# ── STEP 6: Configure merged model (thinking=ON) ─────────────────────────────
echo ""
echo "============================================================"
echo "STEP 6: Configuring merged model template + generation_config"
echo "============================================================"
python3 - << 'PYEOF'
import json, re, pathlib, shutil

MERGED = pathlib.Path("/scratch/safety/sft_cot_distill/merged")
GRPO   = pathlib.Path("/scratch/safety/grpo_v5/merged")
SFT2   = pathlib.Path("/scratch/safety/sft_v2/merged")

# --- generation_config: sampling mode for thinking=ON inference ---
gen_cfg = {
    "bos_token_id": 151643,
    "do_sample": True,
    "eos_token_id": [151645, 151643],
    "pad_token_id": 151643,
    "temperature": 0.7,
    "top_k": 50,
    "top_p": 0.95,
    "transformers_version": "4.51.0"
}
(MERGED / "generation_config.json").write_text(
    json.dumps(gen_cfg, indent=2), encoding="utf-8"
)
print("  Wrote generation_config.json (do_sample=true, temp=0.7)")

# --- chat template: minimal thinking=ON template (no few-shot, no enable_thinking=false)
# Use sft_v2's minimal template and STRIP the enable_thinking=false line if present
# sft_v2 was trained thinking=OFF so its template has that line — we remove it so
# the BASE model defaults to thinking=ON at inference.
template_src = None
for src in (SFT2 / "chat_template.jinja", GRPO / "chat_template.jinja"):
    if src.exists():
        template_src = src
        break

if template_src is None:
    print("  WARNING: no source template found — using merged model's own template")
else:
    template = template_src.read_text(encoding="utf-8")
    # Remove enable_thinking=false line (Lesson 40 — would break thinking=ON inference)
    before = template.count("enable_thinking")
    template = re.sub(
        r'\{%-?\s*set enable_thinking\s*=\s*false\s*-?%\}\s*\n?',
        '',
        template
    )
    after = template.count("enable_thinking")
    (MERGED / "chat_template.jinja").write_text(template, encoding="utf-8")
    print(f"  Template from {template_src.name}: removed {before-after} enable_thinking=false line(s)")

# --- verify template tail (must end with assistant\n, NOT </think>\n\n)
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(str(MERGED))
test_msgs = [{"role": "user", "content": "Test?\nA) Option 1\nB) Option 2"}]
tail = tok.apply_chat_template(
    test_msgs, tokenize=False, add_generation_prompt=True
)[-60:]
print(f"  Template tail: {repr(tail)}")
if "assistant\n" in tail and "</think>" not in tail:
    print("  [OK] Prompt ends at 'assistant\\n' — thinking=ON inference will generate <think> block")
elif "</think>" in tail:
    print("  [WARN] Prompt ends at </think> — thinking=OFF is still active! Check template.")
else:
    print("  [WARN] Unexpected template tail — check manually")
PYEOF

# ── STEP 7: Validate ──────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "STEP 7: Validation"
echo "============================================================"
PYTHONPATH=$CODE python "$CODE/shared/test_vllm_base.py" \
    --model /scratch/safety/sft_cot_distill/merged \
    --validation-file /scratch/standard-project-m2-ma-que/validation_samples/safety.jsonl \
    --gpu-memory-utilization 0.80

echo ""
echo "============================================================"
echo "PUSH DECISION"
echo "============================================================"
echo "  push if: pass@1 > 76% AND format_rate >= 85%"
echo "  (thinking=ON may have format_rate 85-95%, not 100%)"
echo ""
echo "  To push:"
echo "    hf upload cs-552-2026-ma-que/safety_model /scratch/safety/sft_cot_distill/merged ."
echo ""
echo "  If pass@1 <= 70%:"
echo "    Restore grpo_v5 (floor model):"
echo "    hf upload cs-552-2026-ma-que/safety_model /scratch/safety/grpo_v5/merged ."
echo "============================================================"
