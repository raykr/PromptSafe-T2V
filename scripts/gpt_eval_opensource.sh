#!/bin/bash
# 使用开源模型进行评估（无需 OpenAI API）

# 推荐的开源模型：
# 1. LLaVA-1.5 (7B): llava-hf/llava-1.5-7b-hf
# 2. LLaVA-1.5 (13B): llava-hf/llava-1.5-13b-hf  
# 3. Qwen-VL: Qwen/Qwen-VL-Chat
# 4. Qwen-VL2: Qwen/Qwen2-VL-2B-Instruct (更小更快)

python gpt_t2v_eval.py \
    --baseline_dir "out/toxic/baseline" \
    --defense_dir "out/toxic/multi_defense" \
    --out_csv "results_opensource.csv" \
    --model_type opensource \
    --model "llava-hf/llava-1.5-7b-hf" \
    --device "cuda:0" \
    --fps 1 \
    --max_frames 8 \
    --resize_max_side 512 \
    --max_retries 3 \
    --aspects "sexual,violent,political,disturbing"

