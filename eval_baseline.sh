#!/bin/bash
# 只生成baseline视频（不使用adapter）

CUDA_VISIBLE_DEVICES=4 python eval_adapter.py \
    --testset_path "datasets/test/demo.csv" \
    --output_dir "out/defense" \
    --adapter_map "sexual:checkpoints/sexual/safe_adapter_epoch55.pt,violent:checkpoints/violent/safe_adapter.pt,political:checkpoints/political/safe_adapter_epoch70.pt,disturbing:checkpoints/disturbing/safe_adapter.pt" \
    --cls_ckpt_path "checkpoints/prompt_classifier.pt" \
    --generate_baseline \
    --skip_existing \
    --num_frames 49 \
    --height 480 \
    --width 720 \
    --num_inference_steps 50 \
    --guidance_scale 7.5 \
    --fps 16 \
    --route_thresh 0.3
