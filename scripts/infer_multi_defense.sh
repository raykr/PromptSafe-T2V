#!/bin/bash
# 只生成防御视频（使用adapter）

python infer_adapter.py \
    --testset_path "datasets/test/tiny/sexual.csv" \
    --output_dir "out/tiny/sexual/multi_defense" \
    --adapter_map "sexual:checkpoints/sexual/safe_adapter.pt,violent:checkpoints/violent/safe_adapter.pt,political:checkpoints/political/safe_adapter.pt,disturbing:checkpoints/disturbing/safe_adapter.pt" \
    --cls_ckpt_path "checkpoints/classifier/prompt_classifier.pt" \
    --generate_defense \
    --skip_existing \
    --num_frames 49 \
    --height 480 \
    --width 720 \
    --num_inference_steps 50 \
    --guidance_scale 7.5 \
    --fps 16 \
    --seed 42 \
    --mode multi \
    --device "cuda:1" \
    --cls_device "cuda:0" \
    --route_thresh 0.3

python infer_adapter.py \
    --testset_path "datasets/test/tiny/violent.csv" \
    --output_dir "out/tiny/violent/multi_defense" \
    --adapter_map "sexual:checkpoints/sexual/safe_adapter.pt,violent:checkpoints/violent/safe_adapter.pt,political:checkpoints/political/safe_adapter.pt,disturbing:checkpoints/disturbing/safe_adapter.pt" \
    --cls_ckpt_path "checkpoints/classifier/prompt_classifier.pt" \
    --generate_defense \
    --skip_existing \
    --num_frames 49 \
    --height 480 \
    --width 720 \
    --num_inference_steps 50 \
    --guidance_scale 7.5 \
    --fps 16 \
    --seed 42 \
    --mode multi \
    --device "cuda:1" \
    --cls_device "cuda:0" \
    --route_thresh 0.3

python infer_adapter.py \
    --testset_path "datasets/test/tiny/political.csv" \
    --output_dir "out/tiny/political/multi_defense" \
    --adapter_map "sexual:checkpoints/sexual/safe_adapter.pt,violent:checkpoints/violent/safe_adapter.pt,political:checkpoints/political/safe_adapter.pt,disturbing:checkpoints/disturbing/safe_adapter.pt" \
    --cls_ckpt_path "checkpoints/classifier/prompt_classifier.pt" \
    --generate_defense \
    --skip_existing \
    --num_frames 49 \
    --height 480 \
    --width 720 \
    --num_inference_steps 50 \
    --guidance_scale 7.5 \
    --fps 16 \
    --seed 42 \
    --mode multi \
    --device "cuda:1" \
    --cls_device "cuda:0" \
    --route_thresh 0.3

python infer_adapter.py \
    --testset_path "datasets/test/tiny/disturbing.csv" \
    --output_dir "out/tiny/disturbing/multi_defense" \
    --adapter_map "sexual:checkpoints/sexual/safe_adapter.pt,violent:checkpoints/violent/safe_adapter.pt,political:checkpoints/political/safe_adapter.pt,disturbing:checkpoints/disturbing/safe_adapter.pt" \
    --cls_ckpt_path "checkpoints/classifier/prompt_classifier.pt" \
    --generate_defense \
    --skip_existing \
    --num_frames 49 \
    --height 480 \
    --width 720 \
    --num_inference_steps 50 \
    --guidance_scale 7.5 \
    --fps 16 \
    --seed 42 \
    --mode multi \
    --device "cuda:1" \
    --cls_device "cuda:0" \
    --route_thresh 0.3