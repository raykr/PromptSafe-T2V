#!/bin/bash
# 单adapter模式：只生成baseline视频（不使用adapter）

CUDA_VISIBLE_DEVICES=4 python eval_adapter.py \
    --mode single \
    --testset_path "datasets/test/toxic.csv" \
    --output_dir "out/single/toxic" \
    --adapter_path "checkpoints/sexual/safe_adapter_epoch55.pt" \
    --cls_ckpt_path "checkpoints/classifier/prompt_classifier.pt" \
    --generate_baseline \
    --skip_existing \
    --num_frames 49 \
    --height 480 \
    --width 720 \
    --num_inference_steps 50 \
    --guidance_scale 7.5 \
    --fps 16
