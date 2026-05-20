#!/bin/bash
set -e

echo "=== Merging LoRA adapter ==="
python /scratch/safety/safety/training/merge_lora.py \
    --adapter /scratch/safety/sft_mcq_v2/lora_adapter \
    --base /scratch/safety/grpo_v5/merged \
    --output /scratch/safety/sft_mcq_v2/merged

echo "=== Checking chat_template ==="
if grep -q "few_shot_prefix" /scratch/safety/sft_mcq_v2/merged/chat_template.jinja 2>/dev/null; then
    echo "  chat_template.jinja OK (few_shot_prefix present)"
else
    echo "  Copying chat_template.jinja from grpo_v5/merged"
    cp /scratch/safety/grpo_v5/merged/chat_template.jinja /scratch/safety/sft_mcq_v2/merged/chat_template.jinja
fi

echo "=== Validating ==="
python /scratch/safety/shared/test_vllm_base.py \
    --model /scratch/safety/sft_mcq_v2/merged \
    --validation-file /scratch/standard-project-m2-ma-que/validation_samples/safety.jsonl \
    --gpu-memory-utilization 0.80
