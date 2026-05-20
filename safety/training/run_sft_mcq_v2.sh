#!/bin/bash
mkdir -p /scratch/safety/sft_mcq_v2
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True nohup python \
    /scratch/safety/safety/training/sft_train.py \
    --train /scratch/safety/data/mcq_safety_filtered.jsonl \
    --val /scratch/safety/data/pkusafe_sft_val.jsonl \
    --base-checkpoint /scratch/safety/grpo_v5/merged \
    --output-dir /scratch/safety/sft_mcq_v2 \
    --epochs 2 \
    --lr 5e-6 \
    --lora-r 32 \
    --lora-alpha 64 \
    --batch-size 4 \
    --grad-accum 8 \
    --max-seq-length 1536 \
    > /scratch/safety/sft_mcq_v2/train.log 2>&1 &
echo "Training PID: $!"
echo "Watch: tail -f /scratch/safety/sft_mcq_v2/train.log"
