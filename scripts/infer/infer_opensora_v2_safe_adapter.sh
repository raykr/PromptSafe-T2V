#!/bin/bash
# Open-Sora v2 Safe Adapter 推理脚本
# 使用训练好的LoRA适配器进行安全视频生成

set -e

############## 配置参数 ##############

# Open-Sora v2 相关路径
# OPENSORA_REPO: Open-Sora代码仓库路径（必须指向代码仓库，不是模型目录）
OPENSORA_REPO="${OPENSORA_REPO:-/home/raykr/projects/Open-Sora}"

# T5 文本编码器权重路径
OPENSORA_CKPT="${OPENSORA_CKPT:-/home/raykr/models/hpcai-tech/Open-Sora-v2/google/t5-v1_1-xxl}"

# Adapter保存目录（训练时使用的路径）
ADAPTER_ROOT="${ADAPTER_ROOT:-checkpoints/opensora_v2_adapters}"

# 推理参数
PROMPT="${PROMPT:-A beautiful sunset over the ocean}"  # 要生成的视频提示词
OUTPUT_PATH="${OUTPUT_PATH:-outputs/safe_video.mp4}"  # 输出视频路径
DEFENSE_SCALE="${DEFENSE_SCALE:-1.0}"  # 防御强度（LoRA scale，越大防御越强）
FORCE_CATEGORY="${FORCE_CATEGORY:-}"  # 强制使用某个类别的adapter（可选：sexual, violent, political, disturbing）

# 视频生成参数
NUM_FRAMES="${NUM_FRAMES:-49}"  # 视频帧数
HEIGHT="${HEIGHT:-480}"  # 视频高度
WIDTH="${WIDTH:-848}"  # 视频宽度
NUM_STEPS="${NUM_STEPS:-30}"  # 扩散步数
GUIDANCE="${GUIDANCE:-7.5}"  # 引导强度
SEED="${SEED:-42}"  # 随机种子

# GPU设置
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# 内存优化选项
OFFLOAD_MODEL="${OFFLOAD_MODEL:-0}"  # 1=启用CPU offloading（节省GPU内存但更慢）
USE_MULTI_GPU="${USE_MULTI_GPU:-0}"  # 1=使用多GPU（需要至少2个GPU）

# PyTorch内存优化（可选）
# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

############## 打印配置 ##############

echo "=========================================="
echo " Open-Sora v2 Safe Adapter 推理"
echo "=========================================="
echo "OPENSORA_REPO   : $OPENSORA_REPO"
echo "OPENSORA_CKPT   : $OPENSORA_CKPT"
echo "ADAPTER_ROOT    : $ADAPTER_ROOT"
echo "PROMPT          : $PROMPT"
echo "OUTPUT_PATH     : $OUTPUT_PATH"
echo "DEFENSE_SCALE   : $DEFENSE_SCALE"
if [ -n "$FORCE_CATEGORY" ]; then
  echo "FORCE_CATEGORY  : $FORCE_CATEGORY"
else
  echo "FORCE_CATEGORY  : (自动检测)"
fi
echo "NUM_FRAMES      : $NUM_FRAMES"
echo "HEIGHT          : $HEIGHT"
echo "WIDTH           : $WIDTH"
echo "NUM_STEPS       : $NUM_STEPS"
echo "GUIDANCE        : $GUIDANCE"
echo "SEED            : $SEED"
echo "CUDA_VISIBLE_DEVICES : $CUDA_VISIBLE_DEVICES"
echo "OFFLOAD_MODEL   : $OFFLOAD_MODEL (1=启用CPU offloading)"
echo "USE_MULTI_GPU   : $USE_MULTI_GPU (1=使用多GPU)"
echo "=========================================="

# 检查adapter目录
if [ ! -d "$ADAPTER_ROOT" ]; then
  echo "❌ 错误: Adapter目录不存在: $ADAPTER_ROOT"
  echo "请先训练adapters或检查ADAPTER_ROOT路径"
  exit 1
fi

# 创建输出目录
mkdir -p "$(dirname "$OUTPUT_PATH")"

############## 执行推理 ##############

cd "$(dirname "$0")/.."

# 构建命令参数
CMD_ARGS=(
  --mode infer
  --opensora_repo "$OPENSORA_REPO"
  --opensora_ckpt "$OPENSORA_CKPT"
  --adapter_root "$ADAPTER_ROOT"
  --prompt "$PROMPT"
  --out "$OUTPUT_PATH"
  --defense_scale "$DEFENSE_SCALE"
  --num_frames "$NUM_FRAMES"
  --height "$HEIGHT"
  --width "$WIDTH"
  --num_steps "$NUM_STEPS"
  --guidance "$GUIDANCE"
  --seed "$SEED"
)

# 如果指定了强制类别，添加参数
if [ -n "$FORCE_CATEGORY" ]; then
  CMD_ARGS+=(--force_category "$FORCE_CATEGORY")
fi

# 添加内存优化选项
if [ "$OFFLOAD_MODEL" = "1" ]; then
  CMD_ARGS+=(--offload_model)
  echo "⚠️  启用CPU offloading（模型将加载到CPU，推理时移到GPU）"
fi

if [ "$USE_MULTI_GPU" = "1" ]; then
  CMD_ARGS+=(--use_multi_gpu)
  echo "⚠️  启用多GPU推理（使用DataParallel）"
fi

CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python opensora_safe_adapter.py "${CMD_ARGS[@]}"

echo ""
echo "=========================================="
echo " ✅ 推理完成"
echo " 视频保存于: $OUTPUT_PATH"
echo "=========================================="
