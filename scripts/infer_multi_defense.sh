#!/bin/bash
# 只生成防御视频（使用adapter）

# ------------------------ CogVideoX-2b ------------------------
# ---------------------------- tiny ----------------------------
# python infer_adapter.py \
#     --testset_path "datasets/test/tiny/sexual.csv" \
#     --output_dir "out/cogvideox-2b/tiny/sexual/multi_defense" \
#     --adapter_map "sexual:checkpoints/cogvideox-2b/sexual/safe_adapter.pt,violent:checkpoints/cogvideox-2b/violent/safe_adapter.pt,political:checkpoints/cogvideox-2b/political/safe_adapter.pt,disturbing:checkpoints/cogvideox-2b/disturbing/safe_adapter.pt" \
#     --cls_ckpt_path "checkpoints/cogvideox-2b/classifier/prompt_classifier.pt" \
#     --generate_defense \
#     --skip_existing \
#     --num_frames 49 \
#     --height 480 \
#     --width 720 \
#     --num_inference_steps 50 \
#     --guidance_scale 7.5 \
#     --fps 16 \
#     --seed 42 \
#     --mode multi \
#     --device "cuda:1" \
#     --cls_device "cuda:0" \
#     --route_thresh 0.3

# python infer_adapter.py \
#     --testset_path "datasets/test/tiny/violent.csv" \
#     --output_dir "out/cogvideox-2b/tiny/violent/multi_defense" \
#     --adapter_map "sexual:checkpoints/cogvideox-2b/sexual/safe_adapter.pt,violent:checkpoints/cogvideox-2b/violent/safe_adapter.pt,political:checkpoints/cogvideox-2b/political/safe_adapter.pt,disturbing:checkpoints/cogvideox-2b/disturbing/safe_adapter.pt" \
#     --cls_ckpt_path "checkpoints/cogvideox-2b/classifier/prompt_classifier.pt" \
#     --generate_defense \
#     --skip_existing \
#     --num_frames 49 \
#     --height 480 \
#     --width 720 \
#     --num_inference_steps 50 \
#     --guidance_scale 7.5 \
#     --fps 16 \
#     --seed 42 \
#     --mode multi \
#     --device "cuda:1" \
#     --cls_device "cuda:0" \
#     --route_thresh 0.3

# python infer_adapter.py \
#     --testset_path "datasets/test/tiny/political.csv" \
#     --output_dir "out/cogvideox-2b/tiny/political/multi_defense" \
#     --adapter_map "sexual:checkpoints/cogvideox-2b/sexual/safe_adapter.pt,violent:checkpoints/cogvideox-2b/violent/safe_adapter.pt,political:checkpoints/cogvideox-2b/political/safe_adapter.pt,disturbing:checkpoints/cogvideox-2b/disturbing/safe_adapter.pt" \
#     --cls_ckpt_path "checkpoints/cogvideox-2b/classifier/prompt_classifier.pt" \
#     --generate_defense \
#     --skip_existing \
#     --num_frames 49 \
#     --height 480 \
#     --width 720 \
#     --num_inference_steps 50 \
#     --guidance_scale 7.5 \
#     --fps 16 \
#     --seed 42 \
#     --mode multi \
#     --device "cuda:1" \
#     --cls_device "cuda:0" \
#     --route_thresh 0.3

# python infer_adapter.py \
#     --testset_path "datasets/test/tiny/disturbing.csv" \
#     --output_dir "out/cogvideox-2b/tiny/disturbing/multi_defense" \
#     --adapter_map "sexual:checkpoints/cogvideox-2b/sexual/safe_adapter.pt,violent:checkpoints/cogvideox-2b/violent/safe_adapter.pt,political:checkpoints/cogvideox-2b/political/safe_adapter.pt,disturbing:checkpoints/cogvideox-2b/disturbing/safe_adapter.pt" \
#     --cls_ckpt_path "checkpoints/cogvideox-2b/classifier/prompt_classifier.pt" \
#     --generate_defense \
#     --skip_existing \
#     --num_frames 49 \
#     --height 480 \
#     --width 720 \
#     --num_inference_steps 50 \
#     --guidance_scale 7.5 \
#     --fps 16 \
#     --seed 42 \
#     --mode multi \
#     --device "cuda:1" \
#     --cls_device "cuda:0" \
#     --route_thresh 0.3

# ---------------------------- VEIL ----------------------------
# python infer_adapter.py \
#     --testset_path "datasets/test/VEIL/sexual.csv" \
#     --output_dir "out/cogvideox-2b/VEIL/sexual/multi_defense" \
#     --adapter_map "sexual:checkpoints/cogvideox-2b/sexual/safe_adapter.pt,violent:checkpoints/cogvideox-2b/violent/safe_adapter.pt,political:checkpoints/cogvideox-2b/political/safe_adapter.pt,disturbing:checkpoints/cogvideox-2b/disturbing/safe_adapter.pt" \
#     --cls_ckpt_path "checkpoints/cogvideox-2b/classifier/prompt_classifier.pt" \
#     --generate_defense \
#     --skip_existing \
#     --num_frames 49 \
#     --height 480 \
#     --width 720 \
#     --num_inference_steps 50 \
#     --guidance_scale 7.5 \
#     --fps 16 \
#     --seed 42 \
#     --mode multi \
#     --device "cuda:1" \
#     --cls_device "cuda:0" \
#     --route_thresh 0.3

# python infer_adapter.py \
#     --testset_path "datasets/test/VEIL/violent.csv" \
#     --output_dir "out/cogvideox-2b/VEIL/violent/multi_defense" \
#     --adapter_map "sexual:checkpoints/cogvideox-2b/sexual/safe_adapter.pt,violent:checkpoints/cogvideox-2b/violent/safe_adapter.pt,political:checkpoints/cogvideox-2b/political/safe_adapter.pt,disturbing:checkpoints/cogvideox-2b/disturbing/safe_adapter.pt" \
#     --cls_ckpt_path "checkpoints/cogvideox-2b/classifier/prompt_classifier.pt" \
#     --generate_defense \
#     --skip_existing \
#     --num_frames 49 \
#     --height 480 \
#     --width 720 \
#     --num_inference_steps 50 \
#     --guidance_scale 7.5 \
#     --fps 16 \
#     --seed 42 \
#     --mode multi \
#     --device "cuda:1" \
#     --cls_device "cuda:0" \
#     --route_thresh 0.3

# python infer_adapter.py \
#     --testset_path "datasets/test/VEIL/political.csv" \
#     --output_dir "out/cogvideox-2b/VEIL/political/multi_defense" \
#     --adapter_map "sexual:checkpoints/cogvideox-2b/sexual/safe_adapter.pt,violent:checkpoints/cogvideox-2b/violent/safe_adapter.pt,political:checkpoints/cogvideox-2b/political/safe_adapter.pt,disturbing:checkpoints/cogvideox-2b/disturbing/safe_adapter.pt" \
#     --cls_ckpt_path "checkpoints/cogvideox-2b/classifier/prompt_classifier.pt" \
#     --generate_defense \
#     --skip_existing \
#     --num_frames 49 \
#     --height 480 \
#     --width 720 \
#     --num_inference_steps 50 \
#     --guidance_scale 7.5 \
#     --fps 16 \
#     --seed 42 \
#     --mode multi \
#     --device "cuda:1" \
#     --cls_device "cuda:0" \
#     --route_thresh 0.3

# python infer_adapter.py \
#     --testset_path "datasets/test/VEIL/disturbing.csv" \
#     --output_dir "out/cogvideox-2b/VEIL/disturbing/multi_defense" \
#     --adapter_map "sexual:checkpoints/cogvideox-2b/sexual/safe_adapter.pt,violent:checkpoints/cogvideox-2b/violent/safe_adapter.pt,political:checkpoints/cogvideox-2b/political/safe_adapter.pt,disturbing:checkpoints/cogvideox-2b/disturbing/safe_adapter.pt" \
#     --cls_ckpt_path "checkpoints/cogvideox-2b/classifier/prompt_classifier.pt" \
#     --generate_defense \
#     --skip_existing \
#     --num_frames 49 \
#     --height 480 \
#     --width 720 \
#     --num_inference_steps 50 \
#     --guidance_scale 7.5 \
#     --fps 16 \
#     --seed 42 \
#     --mode multi \
#     --device "cuda:1" \
#     --cls_device "cuda:0" \
#     --route_thresh 0.3

# ------------------------ Wan2.1-T2V-1.3B-Diffusers ------------------------
# ---------------------------- tiny ----------------------------
python infer_adapter.py \
    --model_type "wan" \
    --model_path "/home/raykr/models/Wan-AI/Wan2.1-T2V-1.3B-Diffusers" \
    --testset_path "datasets/test/tiny/sexual.csv" \
    --output_dir "out/wan2.1-t2v-1.3b-diffusers/tiny/sexual/multi_defense" \
    --adapter_map "sexual:checkpoints/wan2.1-t2v-1.3b-diffusers/sexual/safe_adapter.pt,violent:checkpoints/wan2.1-t2v-1.3b-diffusers/violent/safe_adapter.pt,political:checkpoints/wan2.1-t2v-1.3b-diffusers/political/safe_adapter.pt,disturbing:checkpoints/wan2.1-t2v-1.3b-diffusers/disturbing/safe_adapter.pt" \
    --cls_ckpt_path "checkpoints/wan2.1-t2v-1.3b-diffusers/classifier/prompt_classifier.pt" \
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
    --model_type "wan" \
    --model_path "/home/raykr/models/Wan-AI/Wan2.1-T2V-1.3B-Diffusers" \
    --testset_path "datasets/test/tiny/violent.csv" \
    --output_dir "out/wan2.1-t2v-1.3b-diffusers/tiny/violent/multi_defense" \
    --adapter_map "sexual:checkpoints/wan2.1-t2v-1.3b-diffusers/sexual/safe_adapter_epoch50.pt,violent:checkpoints/wan2.1-t2v-1.3b-diffusers/violent/safe_adapter.pt,political:checkpoints/wan2.1-t2v-1.3b-diffusers/political/safe_adapter.pt,disturbing:checkpoints/wan2.1-t2v-1.3b-diffusers/disturbing/safe_adapter.pt" \
    --cls_ckpt_path "checkpoints/wan2.1-t2v-1.3b-diffusers/classifier/prompt_classifier.pt" \
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
    --model_type "wan" \
    --model_path "/home/raykr/models/Wan-AI/Wan2.1-T2V-1.3B-Diffusers" \
    --testset_path "datasets/test/tiny/political.csv" \
    --output_dir "out/wan2.1-t2v-1.3b-diffusers/tiny/political/multi_defense" \
    --adapter_map "sexual:checkpoints/wan2.1-t2v-1.3b-diffusers/sexual/safe_adapter_epoch50.pt,violent:checkpoints/wan2.1-t2v-1.3b-diffusers/violent/safe_adapter.pt,political:checkpoints/wan2.1-t2v-1.3b-diffusers/political/safe_adapter.pt,disturbing:checkpoints/wan2.1-t2v-1.3b-diffusers/disturbing/safe_adapter.pt" \
    --cls_ckpt_path "checkpoints/wan2.1-t2v-1.3b-diffusers/classifier/prompt_classifier.pt" \
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
    --model_type "wan" \
    --model_path "/home/raykr/models/Wan-AI/Wan2.1-T2V-1.3B-Diffusers" \
    --testset_path "datasets/test/tiny/disturbing.csv" \
    --output_dir "out/wan2.1-t2v-1.3b-diffusers/tiny/disturbing/multi_defense" \
    --adapter_map "sexual:checkpoints/wan2.1-t2v-1.3b-diffusers/sexual/safe_adapter_epoch50.pt,violent:checkpoints/wan2.1-t2v-1.3b-diffusers/violent/safe_adapter.pt,political:checkpoints/wan2.1-t2v-1.3b-diffusers/political/safe_adapter.pt,disturbing:checkpoints/wan2.1-t2v-1.3b-diffusers/disturbing/safe_adapter.pt" \
    --cls_ckpt_path "checkpoints/wan2.1-t2v-1.3b-diffusers/classifier/prompt_classifier.pt" \
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



# ---------------------------- VEIL ----------------------------
python infer_adapter.py \
    --model_type "wan" \
    --model_path "/home/raykr/models/Wan-AI/Wan2.1-T2V-1.3B-Diffusers" \
    --testset_path "datasets/test/VEIL/sexual.csv" \
    --output_dir "out/wan2.1-t2v-1.3b-diffusers/VEIL/sexual/multi_defense" \
    --adapter_map "sexual:checkpoints/wan2.1-t2v-1.3b-diffusers/sexual/safe_adapter_epoch50.pt,violent:checkpoints/wan2.1-t2v-1.3b-diffusers/violent/safe_adapter.pt,political:checkpoints/wan2.1-t2v-1.3b-diffusers/political/safe_adapter.pt,disturbing:checkpoints/wan2.1-t2v-1.3b-diffusers/disturbing/safe_adapter.pt" \
    --cls_ckpt_path "checkpoints/wan2.1-t2v-1.3b-diffusers/classifier/prompt_classifier.pt" \
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
    --model_type "wan" \
    --model_path "/home/raykr/models/Wan-AI/Wan2.1-T2V-1.3B-Diffusers" \
    --testset_path "datasets/test/VEIL/violent.csv" \
    --output_dir "out/wan2.1-t2v-1.3b-diffusers/VEIL/violent/multi_defense" \
    --adapter_map "sexual:checkpoints/wan2.1-t2v-1.3b-diffusers/sexual/safe_adapter_epoch50.pt,violent:checkpoints/wan2.1-t2v-1.3b-diffusers/violent/safe_adapter.pt,political:checkpoints/wan2.1-t2v-1.3b-diffusers/political/safe_adapter.pt,disturbing:checkpoints/wan2.1-t2v-1.3b-diffusers/disturbing/safe_adapter.pt" \
    --cls_ckpt_path "checkpoints/wan2.1-t2v-1.3b-diffusers/classifier/prompt_classifier.pt" \
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
    --model_type "wan" \
    --model_path "/home/raykr/models/Wan-AI/Wan2.1-T2V-1.3B-Diffusers" \
    --testset_path "datasets/test/VEIL/political.csv" \
    --output_dir "out/wan2.1-t2v-1.3b-diffusers/VEIL/political/multi_defense" \
    --adapter_map "sexual:checkpoints/wan2.1-t2v-1.3b-diffusers/sexual/safe_adapter_epoch50.pt,violent:checkpoints/wan2.1-t2v-1.3b-diffusers/violent/safe_adapter.pt,political:checkpoints/wan2.1-t2v-1.3b-diffusers/political/safe_adapter.pt,disturbing:checkpoints/wan2.1-t2v-1.3b-diffusers/disturbing/safe_adapter.pt" \
    --cls_ckpt_path "checkpoints/wan2.1-t2v-1.3b-diffusers/classifier/prompt_classifier.pt" \
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
    --model_type "wan" \
    --model_path "/home/raykr/models/Wan-AI/Wan2.1-T2V-1.3B-Diffusers" \
    --testset_path "datasets/test/VEIL/disturbing.csv" \
    --output_dir "out/wan2.1-t2v-1.3b-diffusers/VEIL/disturbing/multi_defense" \
    --adapter_map "sexual:checkpoints/wan2.1-t2v-1.3b-diffusers/sexual/safe_adapter_epoch50.pt,violent:checkpoints/wan2.1-t2v-1.3b-diffusers/violent/safe_adapter.pt,political:checkpoints/wan2.1-t2v-1.3b-diffusers/political/safe_adapter.pt,disturbing:checkpoints/wan2.1-t2v-1.3b-diffusers/disturbing/safe_adapter.pt" \
    --cls_ckpt_path "checkpoints/wan2.1-t2v-1.3b-diffusers/classifier/prompt_classifier.pt" \
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