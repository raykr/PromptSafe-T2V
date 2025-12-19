#!/bin/bash
# 只生成baseline视频（不使用adapter）

python infer_adapter.py \
    --model_path "/home/raykr/models/zai-org/CogVideoX-2b" \
    --testset_path "datasets/test/tiny/sexual.csv" \
    --output_dir "out/tiny/sexual/baseline" \
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
    --device "cuda:0" &

python infer_adapter.py \
    --model_path "/home/raykr/models/zai-org/CogVideoX-2b" \
    --testset_path "datasets/test/tiny/violent.csv" \
    --output_dir "out/tiny/violent/baseline" \
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
    --device "cuda:1" &

wait

python infer_adapter.py \
    --model_path "/home/raykr/models/zai-org/CogVideoX-2b" \
    --testset_path "datasets/test/tiny/political.csv" \
    --output_dir "out/tiny/political/baseline" \
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
    --device "cuda:0" &

python infer_adapter.py \
    --model_path "/home/raykr/models/zai-org/CogVideoX-2b" \
    --testset_path "datasets/test/tiny/disturbing.csv" \
    --output_dir "out/tiny/disturbing/baseline" \
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
    --device "cuda:1" &
wait