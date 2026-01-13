#!/bin/bash
# 快速测试 Open-Sora v2 Safe Adapter 推理

# 修复 PyTorch 与 Intel MKL 的符号冲突问题
export MKL_SERVICE_FORCE_INTEL=1
export KMP_DUPLICATE_LIB_OK=TRUE
export MKL_INTERFACE_LAYER=LP64,GNU

# 尝试预加载 Intel OpenMP 库（如果存在）
if [ -n "$CONDA_PREFIX" ] && [ -f "$CONDA_PREFIX/lib/libiomp5.so" ]; then
  export LD_PRELOAD="$CONDA_PREFIX/lib/libiomp5.so:$LD_PRELOAD"
fi

# PyTorch 内存优化
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 设置测试参数
export PROMPT="A beautiful sunset over the ocean with birds flying"
export OUTPUT_PATH="outputs/test_safe_video.mp4"
export DEFENSE_SCALE=1.0
export NUM_FRAMES=25  # 减少帧数以加快测试
export HEIGHT=480
export WIDTH=480
export NUM_STEPS=20  # 减少步数以加快测试
export GUIDANCE=7.5
export SEED=42
export CUDA_VISIBLE_DEVICES=0
# 如果仍然 OOM，可以启用 CPU offloading（会更慢但节省 GPU 内存）
# export OFFLOAD_MODEL=1

# 运行推理脚本
bash scripts/infer/infer_opensora_v2_safe_adapter.sh
