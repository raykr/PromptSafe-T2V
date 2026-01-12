#!/bin/bash
# 单adapter模式：只生成防御视频（使用adapter）

CUDA_VISIBLE_DEVICES=4 python infer_adapter.py \
    --mode single \
    --testset_path "datasets/test/toxic.csv" \
    --output_dir "out/single/toxic" \
    --adapter_path "checkpoints/cogvideox-2b/sexual/safe_adapter_epoch55.pt" \
    --cls_ckpt_path "checkpoints/cogvideox-2b/classifier/prompt_classifier.pt" \
    --generate_defense \
    --skip_existing \
    --num_frames 49 \
    --height 480 \
    --width 720 \
    --num_inference_steps 50 \
    --guidance_scale 7.5 \
    --fps 16
