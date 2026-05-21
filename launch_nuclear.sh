#!/usr/bin/env bash
# Nuclear option — Path B: CoT trace generation → SFT → merge → validate
# Run from /scratch/safety/:
#   chmod +x launch_nuclear.sh && nohup ./launch_nuclear.sh > logs/nuclear.log 2>&1 &
#   then: tail -f logs/nuclear.log

set -euo pipefail

SCRATCH="/scratch/safety"
DATA="$SCRATCH/data"
CODE="$SCRATCH/safety"
SHARED="$SCRATCH/shared"
LOGS="$SCRATCH/logs"
VAL="/scratch/standard-project-m2-ma-que/validation_samples/safety.jsonl"

COT_DATA="$DATA/mmlu_safety_cot.jsonl"
SFT_OUT="$SCRATCH/sft_nuclear"
MERGED="$SFT_OUT/merged"

mkdir -p "$LOGS"

echo "======================================================"
echo "  Nuclear Pipeline — Path B"
echo "  $(date)"
echo "======================================================"

# ── Step 1: Generate CoT traces ────────────────────────────────────────────────
echo ""
echo "[1/4] Generating CoT traces from Qwen3-1.7B base..."
python "$CODE/data/generate_cot_traces.py" \
    --input  "$DATA/mmlu_safety_train.jsonl" \
    --output "$COT_DATA"

YIELD=$(wc -l < "$COT_DATA")
echo "  Yield: $YIELD examples"

if [ "$YIELD" -lt 200 ]; then
    echo "ERROR: CoT yield too low ($YIELD < 200). Check logs. Aborting."
    exit 1
fi

# Quick sanity check — print first assistant turn
echo "  Sample trace (first example assistant turn):"
python3 -c "
import json
with open('$COT_DATA') as f:
    d = json.loads(f.readline())
print(d['messages'][2]['content'][:400])
"
echo ""

# ── Step 2: Nuclear SFT ────────────────────────────────────────────────────────
echo "[2/4] Starting nuclear SFT (warm start from sft_mcq_v2/merged, r=64)..."
python "$CODE/training/sft_train.py" \
    --model       "$SCRATCH/sft_mcq_v2/merged" \
    --train       "$COT_DATA" \
    --val         "$DATA/pkusafe_sft_val.jsonl" \
    --output-dir  "$SFT_OUT" \
    --lora-r      64 \
    --lora-alpha  128 \
    --lr          1e-5 \
    --epochs      3 \
    --batch-size  4 \
    --grad-accum  8 \
    --max-seq-length 2048 \
    --max-grad-norm  0.3 \
    --eval-steps  50 \
    --save-steps  50 \
    --logging-steps 5

# ── Step 3: Merge ──────────────────────────────────────────────────────────────
echo ""
echo "[3/4] Merging LoRA adapter..."
python "$CODE/training/merge_lora.py" \
    --adapter "$SFT_OUT/lora_adapter" \
    --base    "$SCRATCH/sft_mcq_v2/merged" \
    --output  "$MERGED"

# Copy greedy generation config from sft_mcq_v2 (do_sample=false)
cp "$SCRATCH/sft_mcq_v2/merged/generation_config.json" "$MERGED/generation_config.json"
echo "  Copied greedy generation_config.json"

# ── Step 4: Validate ───────────────────────────────────────────────────────────
echo ""
echo "[4/4] Validating merged model..."
python "$SHARED/test_vllm_base.py" \
    --model "$MERGED" \
    --validation-file "$VAL" \
    --gpu-memory-utilization 0.80

echo ""
echo "======================================================"
echo "  Pipeline complete — $(date)"
echo "  Merged model: $MERGED"
echo ""
echo "  NEXT STEPS:"
echo "  1. Check pass@1 above — if > current HF model (80%), push."
echo "  2. Before pushing, patch chat_template.jinja:"
echo "     - Remove '/no_think' from system prompt"
echo "     - Replace with REASONING_SYSTEM_PROMPT from generate_cot_traces.py"
echo "     - (The model now generates <think> from weights — template must not suppress it)"
echo "  3. Push:"
echo "     hf upload cs-552-2026-ma-que/safety_model $MERGED ."
echo "======================================================"
