#!/bin/bash
# 只生成baseline视频（不使用adapter）

python infer_adapter.py \
    --model_path "/home/raykr/models/zai-org/CogVideoX-5b" \
    --testset_path "datasets/test/tiny/sexual.csv" \
    --output_dir "out/CogVideoX-5b/tiny/sexual/baseline" \
    --generate_baseline \
    --skip_existing \
    --num_frames 49 \
    --height 480 \
    --width 720 \
    --num_inference_steps 50 \
    --guidance_scale 7.5 \
    --fps 16 \
    --seed 42 \
    --mode single \
    --device "cuda:0" \
    --torch_dtype "BF16" &

python infer_adapter.py \
    --model_path "/home/raykr/models/zai-org/CogVideoX-5b" \
    --testset_path "datasets/test/tiny/violent.csv" \
    --output_dir "out/CogVideoX-5b/tiny/violent/baseline" \
    --generate_baseline \
    --skip_existing \
    --num_frames 49 \
    --height 480 \
    --width 720 \
    --num_inference_steps 50 \
    --guidance_scale 7.5 \
    --fps 16 \
    --seed 42 \
    --mode single \
    --device "cuda:1" \
    --torch_dtype "BF16" &

wait

python infer_adapter.py \
    --model_path "/home/raykr/models/zai-org/CogVideoX-5b" \
    --testset_path "datasets/test/tiny/political.csv" \
    --output_dir "out/CogVideoX-5b/tiny/political/baseline" \
    --generate_baseline \
    --skip_existing \
    --num_frames 49 \
    --height 480 \
    --width 720 \
    --num_inference_steps 50 \
    --guidance_scale 7.5 \
    --fps 16 \
    --seed 42 \
    --mode single \
    --device "cuda:0" \
    --torch_dtype "BF16" &

python infer_adapter.py \
    --model_path "/home/raykr/models/zai-org/CogVideoX-5b" \
    --testset_path "datasets/test/tiny/disturbing.csv" \
    --output_dir "out/CogVideoX-5b/tiny/disturbing/baseline" \
    --generate_baseline \
    --skip_existing \
    --num_frames 49 \
    --height 480 \
    --width 720 \
    --num_inference_steps 50 \
    --guidance_scale 7.5 \
    --fps 16 \
    --seed 42 \
    --mode single \
    --device "cuda:1" \
    --torch_dtype "BF16" &
wait

# benign
python infer_adapter.py \
    --model_path "/home/raykr/models/zai-org/CogVideoX-5b" \
    --testset_path "datasets/test/benign.csv" \
    --output_dir "out/CogvideoX-5b/benign/baseline" \
    --generate_baseline \
    --skip_existing \
    --num_frames 49 \
    --height 480 \
    --width 720 \
    --num_inference_steps 50 \
    --guidance_scale 7.5 \
    --fps 16 \
    --seed 42 \
    --mode single \
    --device "cuda:0" \
    --torch_dtype "BF16"