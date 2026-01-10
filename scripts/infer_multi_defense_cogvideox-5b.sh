#!/bin/bash
# 只生成防御视频（使用adapter）

python infer_adapter.py \
    --model_path "/home/raykr/models/zai-org/CogVideoX-5b" \
    --torch_dtype "BF16" \
    --testset_path "datasets/test/tiny/sexual.csv" \
    --output_dir "out/CogVideoX-5b/tiny/sexual/multi_defense" \
    --adapter_map "sexual:checkpoints/cogvideox-2b/sexual/safe_adapter.pt,violent:checkpoints/cogvideox-2b/violent/safe_adapter.pt,political:checkpoints/cogvideox-2b/political/safe_adapter.pt,disturbing:checkpoints/cogvideox-2b/disturbing/safe_adapter.pt" \
    --cls_ckpt_path "checkpoints/cogvideox-2b/classifier/prompt_classifier.pt" \
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
    --model_path "/home/raykr/models/zai-org/CogVideoX-5b" \
    --torch_dtype "BF16" \
    --testset_path "datasets/test/tiny/violent.csv" \
    --output_dir "out/CogVideoX-5b/tiny/violent/multi_defense" \
    --adapter_map "sexual:checkpoints/cogvideox-2b/sexual/safe_adapter.pt,violent:checkpoints/cogvideox-2b/violent/safe_adapter.pt,political:checkpoints/cogvideox-2b/political/safe_adapter.pt,disturbing:checkpoints/cogvideox-2b/disturbing/safe_adapter.pt" \
    --cls_ckpt_path "checkpoints/cogvideox-2b/classifier/prompt_classifier.pt" \
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
    --model_path "/home/raykr/models/zai-org/CogVideoX-5b" \
    --torch_dtype "BF16" \
    --testset_path "datasets/test/tiny/political.csv" \
    --output_dir "out/CogVideoX-5b/tiny/political/multi_defense" \
    --adapter_map "sexual:checkpoints/cogvideox-2b/sexual/safe_adapter.pt,violent:checkpoints/cogvideox-2b/violent/safe_adapter.pt,political:checkpoints/cogvideox-2b/political/safe_adapter.pt,disturbing:checkpoints/cogvideox-2b/disturbing/safe_adapter.pt" \
    --cls_ckpt_path "checkpoints/cogvideox-2b/classifier/prompt_classifier.pt" \
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
    --model_path "/home/raykr/models/zai-org/CogVideoX-5b" \
    --torch_dtype "BF16" \
    --testset_path "datasets/test/tiny/disturbing.csv" \
    --output_dir "out/CogVideoX-5b/tiny/disturbing/multi_defense" \
    --adapter_map "sexual:checkpoints/cogvideox-2b/sexual/safe_adapter.pt,violent:checkpoints/cogvideox-2b/violent/safe_adapter.pt,political:checkpoints/cogvideox-2b/political/safe_adapter.pt,disturbing:checkpoints/cogvideox-2b/disturbing/safe_adapter.pt" \
    --cls_ckpt_path "checkpoints/cogvideox-2b/classifier/prompt_classifier.pt" \
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

# benign
python infer_adapter.py \
    --model_path "/home/raykr/models/zai-org/CogVideoX-5b" \
    --torch_dtype "BF16" \
    --testset_path "datasets/test/benign.csv" \
    --output_dir "out/CogVideoX-5b/benign/multi_defense" \
    --adapter_map "sexual:checkpoints/cogvideox-2b/sexual/safe_adapter.pt,violent:checkpoints/cogvideox-2b/violent/safe_adapter.pt,political:checkpoints/cogvideox-2b/political/safe_adapter.pt,disturbing:checkpoints/cogvideox-2b/disturbing/safe_adapter.pt" \
    --cls_ckpt_path "checkpoints/cogvideox-2b/classifier/prompt_classifier.pt" \
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