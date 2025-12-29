#!/bin/bash
# 只生成baseline视频（不使用adapter）

# python infer_adapter.py \
#     --model_path "/home/raykr/models/zai-org/CogVideoX-2b" \
#     --testset_path "datasets/test/tiny/sexual.csv" \
#     --output_dir "out/cogvideox-2b/tiny/sexual/baseline" \
#     --generate_baseline \
#     --skip_existing \
#     --num_frames 49 \
#     --height 480 \
#     --width 720 \
#     --num_inference_steps 50 \
#     --guidance_scale 7.5 \
#     --fps 16 \
#     --seed 42 \
#     --mode single \
#     --device "cuda:0" &

# python infer_adapter.py \
#     --model_path "/home/raykr/models/zai-org/CogVideoX-2b" \
#     --testset_path "datasets/test/tiny/violent.csv" \
#     --output_dir "out/cogvideox-2b/tiny/violent/baseline" \
#     --generate_baseline \
#     --skip_existing \
#     --num_frames 49 \
#     --height 480 \
#     --width 720 \
#     --num_inference_steps 50 \
#     --guidance_scale 7.5 \
#     --fps 16 \
#     --seed 42 \
#     --mode single \
#     --device "cuda:1" &

# wait

# python infer_adapter.py \
#     --model_path "/home/raykr/models/zai-org/CogVideoX-2b" \
#     --testset_path "datasets/test/tiny/political.csv" \
#     --output_dir "out/cogvideox-2b/tiny/political/baseline" \
#     --generate_baseline \
#     --skip_existing \
#     --num_frames 49 \
#     --height 480 \
#     --width 720 \
#     --num_inference_steps 50 \
#     --guidance_scale 7.5 \
#     --fps 16 \
#     --seed 42 \
#     --mode single \
#     --device "cuda:0" &

# python infer_adapter.py \
#     --model_path "/home/raykr/models/zai-org/CogVideoX-2b" \
#     --testset_path "datasets/test/tiny/disturbing.csv" \
#     --output_dir "out/cogvideox-2b/tiny/disturbing/baseline" \
#     --generate_baseline \
#     --skip_existing \
#     --num_frames 49 \
#     --height 480 \
#     --width 720 \
#     --num_inference_steps 50 \
#     --guidance_scale 7.5 \
#     --fps 16 \
#     --seed 42 \
#     --mode single \
#     --device "cuda:1" &
# wait


python infer_adapter.py \
    --model_type wan \
    --model_path "/home/raykr/models/Wan-AI/Wan2.1-T2V-1.3B-Diffusers" \
    --testset_path "datasets/test/tiny/sexual.csv" \
    --output_dir "out/wan2.1-t2v-1.3b-diffusers/tiny/sexual/baseline" \
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
    --model_type wan \
    --model_path "/home/raykr/models/Wan-AI/Wan2.1-T2V-1.3B-Diffusers" \
    --testset_path "datasets/test/tiny/violent.csv" \
    --output_dir "out/wan2.1-t2v-1.3b-diffusers/tiny/violent/baseline" \
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
    --model_type wan \
    --model_path "/home/raykr/models/Wan-AI/Wan2.1-T2V-1.3B-Diffusers" \
    --testset_path "datasets/test/tiny/political.csv" \
    --output_dir "out/wan2.1-t2v-1.3b-diffusers/tiny/political/baseline" \
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
    --model_type wan \
    --model_path "/home/raykr/models/Wan-AI/Wan2.1-T2V-1.3B-Diffusers" \
    --testset_path "datasets/test/tiny/disturbing.csv" \
    --output_dir "out/wan2.1-t2v-1.3b-diffusers/tiny/disturbing/baseline" \
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


# ---------------------------- VEIL ----------------------------
python infer_adapter.py \
    --model_type wan \
    --model_path "/home/raykr/models/Wan-AI/Wan2.1-T2V-1.3B-Diffusers" \
    --testset_path "datasets/test/VEIL/sexual.csv" \
    --output_dir "out/wan2.1-t2v-1.3b-diffusers/VEIL/sexual/baseline" \
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
    --model_type wan \
    --model_path "/home/raykr/models/Wan-AI/Wan2.1-T2V-1.3B-Diffusers" \
    --testset_path "datasets/test/VEIL/violent.csv" \
    --output_dir "out/wan2.1-t2v-1.3b-diffusers/VEIL/violent/baseline" \
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
    --model_type wan \
    --model_path "/home/raykr/models/Wan-AI/Wan2.1-T2V-1.3B-Diffusers" \
    --testset_path "datasets/test/VEIL/political.csv" \
    --output_dir "out/wan2.1-t2v-1.3b-diffusers/VEIL/political/baseline" \
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
    --model_type wan \
    --model_path "/home/raykr/models/Wan-AI/Wan2.1-T2V-1.3B-Diffusers" \
    --testset_path "datasets/test/VEIL/disturbing.csv" \
    --output_dir "out/wan2.1-t2v-1.3b-diffusers/VEIL/disturbing/baseline" \
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