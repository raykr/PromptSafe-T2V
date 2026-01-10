#!/bin/bash
# 只生成baseline视频（不使用adapter）
# 
# 内存优化选项：
# - 使用 --enable_cpu_offload 启用CPU offloading以节省GPU内存（推荐用于单GPU或显存不足时）
# - 设置环境变量 ENABLE_CPU_OFFLOAD=1 来全局启用
# - 设置环境变量 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True 来优化内存分配
# 
# 双GPU策略：
# - 默认（ENABLE_CPU_OFFLOAD=0）：双GPU并行运行，充分利用两张GPU（推荐用于双4090）
# - 启用CPU offloading（ENABLE_CPU_OFFLOAD=1）：双GPU并行运行但使用CPU offloading（节省显存但速度较慢）

# 检查是否启用CPU offloading（默认不启用，充分利用双GPU）
ENABLE_CPU_OFFLOAD=${ENABLE_CPU_OFFLOAD:-0}
CPU_OFFLOAD_FLAG=""
if [ "$ENABLE_CPU_OFFLOAD" = "1" ]; then
    CPU_OFFLOAD_FLAG="--enable_cpu_offload"
    echo "✅ 已启用 CPU offloading 以节省GPU内存（速度会降低）"
else
    echo "✅ 未启用 CPU offloading，将充分利用GPU性能"
fi

# 设置PyTorch内存分配优化
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

# 检测GPU数量
NUM_GPUS=$(python -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo "0")
echo "检测到 ${NUM_GPUS} 张GPU"

# 双GPU时，无论是否启用CPU offloading都可以并行运行
# 单GPU时，启用CPU offloading可以并行运行，否则串行运行
if [ "$NUM_GPUS" -ge 2 ] || [ "$ENABLE_CPU_OFFLOAD" = "1" ]; then
    # 并行模式（双GPU或启用CPU offloading时）
    if [ "$NUM_GPUS" -ge 2 ]; then
        if [ "$ENABLE_CPU_OFFLOAD" = "1" ]; then
            echo "🚀 使用并行模式（双GPU + CPU offloading，同时运行5个任务）"
            # 启用CPU offloading时，可以同时运行5个任务，充分利用双GPU
            # GPU 0: 3个任务 (sexual, political, benign)
            # GPU 1: 2个任务 (violent, disturbing)
            
            # GPU 0 任务
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
                $CPU_OFFLOAD_FLAG &

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
                $CPU_OFFLOAD_FLAG &

            python infer_adapter.py \
                --model_path "/home/raykr/models/zai-org/CogVideoX-5b" \
                --testset_path "datasets/test/benign.csv" \
                --output_dir "out/CogVideoX-5b/benign/baseline" \
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
                $CPU_OFFLOAD_FLAG &

            # GPU 1 任务
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
                $CPU_OFFLOAD_FLAG &

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
                $CPU_OFFLOAD_FLAG &

            # 等待所有任务完成
            wait
        else
            echo "🚀 使用并行模式（双GPU，充分利用GPU资源）"
            # 不使用CPU offloading时，每张GPU运行一个任务（避免显存溢出）
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
                $CPU_OFFLOAD_FLAG &

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
                $CPU_OFFLOAD_FLAG &

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
                $CPU_OFFLOAD_FLAG &

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
                $CPU_OFFLOAD_FLAG &
            wait
            
            # benign
            python infer_adapter.py \
                --model_path "/home/raykr/models/zai-org/CogVideoX-5b" \
                --testset_path "datasets/test/benign.csv" \
                --output_dir "out/CogVideoX-5b/benign/baseline" \
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
                $CPU_OFFLOAD_FLAG
        fi
    else
        echo "🚀 使用并行模式（CPU offloading已启用，单GPU）"
        # 单GPU + CPU offloading，可以同时运行多个任务
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
            $CPU_OFFLOAD_FLAG &

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
            --device "cuda:0" \
            $CPU_OFFLOAD_FLAG &

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
            $CPU_OFFLOAD_FLAG &

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
            --device "cuda:0" \
            $CPU_OFFLOAD_FLAG &

        python infer_adapter.py \
            --model_path "/home/raykr/models/zai-org/CogVideoX-5b" \
            --testset_path "datasets/test/benign.csv" \
            --output_dir "out/CogVideoX-5b/benign/baseline" \
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
            $CPU_OFFLOAD_FLAG &
        
        wait
    fi
else
    # 串行模式（单GPU且不使用CPU offloading时，避免显存溢出）
    echo "⚠️  使用串行模式（单GPU且CPU offloading未启用，避免显存溢出）"
    echo "💡 提示：如果有双GPU，建议使用并行模式；如果显存不足，建议启用CPU offloading"
    
    # ---------------------------- tiny ----------------------------
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
        $CPU_OFFLOAD_FLAG

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
        --device "cuda:0" \
        $CPU_OFFLOAD_FLAG

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
        $CPU_OFFLOAD_FLAG

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
        --device "cuda:0" \
        $CPU_OFFLOAD_FLAG
fi

# benign任务已在并行模式中处理，这里只在串行模式下运行
if [ "$NUM_GPUS" -lt 2 ] && [ "$ENABLE_CPU_OFFLOAD" != "1" ]; then
    python infer_adapter.py \
        --model_path "/home/raykr/models/zai-org/CogVideoX-5b" \
        --testset_path "datasets/test/benign.csv" \
        --output_dir "out/CogVideoX-5b/benign/baseline" \
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
        $CPU_OFFLOAD_FLAG
fi

echo "✅ 所有任务已完成！"