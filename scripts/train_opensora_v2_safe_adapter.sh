#!/bin/bash
# Open-Sora v2 Safe Adapter 训练脚本（在 T5 文本编码器上挂 LoRA）
# 使用前请确认：
#   1）已下载 HuggingFace 上的 Open-Sora-v2 到 /home/raykr/models/hpcai-tech/Open-Sora-v2
#   2）其中包含 T5-XXL 权重：google/t5-v1_1-xxl/

set -e

# 清理GPU内存（可选，如果有其他进程占用GPU）
echo "检查GPU使用情况..."
nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv,noheader,nounits | while IFS=',' read -r idx used free; do
    echo "GPU $idx: Used=${used}MB, Free=${free}MB"
done

# 提示：如果有其他进程占用GPU，可以先kill掉或使用其他GPU
if [ -n "$KILL_OTHER_GPU_PROCESSES" ] && [ "$KILL_OTHER_GPU_PROCESSES" = "1" ]; then
    echo "⚠️  警告：将清理GPU上的其他进程..."
    # 这里不自动kill，需要用户手动确认
    # fuser -v /dev/nvidia* 2>/dev/null || true
fi

############## 可按需修改的配置 ##############

# Open-Sora v2 本地目录（当前主要用来占位传给 --opensora_repo）
OPENSORA_REPO="${OPENSORA_REPO:-/home/raykr/models/hpcai-tech/Open-Sora-v2}"

# T5 文本编码器权重路径（Open-Sora v2 里面的 T5-XXL）
OPENSORA_CKPT="${OPENSORA_CKPT:-/home/raykr/models/hpcai-tech/Open-Sora-v2/google/t5-v1_1-xxl}"

# 训练数据（JSONL，一行一个：{"prompt": "...", "label": "sexual"}）
# 你已经有 sexual.jsonl，可以先用它单类试跑
TRAIN_JSONL="${TRAIN_JSONL:-datasets/train/sexual.jsonl}"

# Adapter 保存目录
ADAPTER_ROOT="${ADAPTER_ROOT:-checkpoints/opensora_v2_adapters}"

# 训练超参数
EPOCHS="${EPOCHS:-10}"
BATCH_SIZE="${BATCH_SIZE:-1}"  # 最小batch size避免OOM（T5-XXL很大，建议用1）
LR="${LR:-1e-4}"
LAMBDA_TOX="${LAMBDA_TOX:-1.0}"
LAMBDA_PRESERVE="${LAMBDA_PRESERVE:-0.1}"

# GPU设置（双卡4090）
# 单卡训练：CUDA_VISIBLE_DEVICES=0
# 双卡训练：CUDA_VISIBLE_DEVICES=0,1 并设置 USE_MULTI_GPU=1
# 如果GPU 0被占用，可以尝试：CUDA_VISIBLE_DEVICES=1
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"  # 默认用GPU 1，避免与占用GPU 0的进程冲突
USE_MULTI_GPU="${USE_MULTI_GPU:-0}"  # 设置为1启用多GPU
# 强制禁用gradient checkpointing（与LoRA训练不兼容，会导致梯度丢失）
# 注意：gradient checkpointing与LoRA训练不兼容，请不要启用
USE_GRADIENT_CHECKPOINTING=0  # 强制禁用

# 内存优化选项
PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"  # 避免内存碎片

############## 打印配置 ##############

echo "=========================================="
echo " Open-Sora v2 Safe Adapter 训练"
echo "=========================================="
echo "OPENSORA_REPO   : $OPENSORA_REPO"
echo "OPENSORA_CKPT   : $OPENSORA_CKPT"
echo "TRAIN_JSONL     : $TRAIN_JSONL"
echo "ADAPTER_ROOT    : $ADAPTER_ROOT"
echo "EPOCHS          : $EPOCHS"
echo "BATCH_SIZE      : $BATCH_SIZE"
echo "LR              : $LR"
echo "LAMBDA_TOX      : $LAMBDA_TOX"
echo "LAMBDA_PRESERVE : $LAMBDA_PRESERVE"
echo "CUDA_VISIBLE_DEVICES : $CUDA_VISIBLE_DEVICES"
echo "=========================================="

if [ ! -f "$TRAIN_JSONL" ]; then
  echo "❌ 找不到训练数据: $TRAIN_JSONL"
  exit 1
fi

mkdir -p "$(dirname "$ADAPTER_ROOT")"

############## 启动训练 ##############

cd "$(dirname "$0")/.."

# 构建命令参数
CMD_ARGS=(
  --mode train
  --opensora_repo "$OPENSORA_REPO"
  --opensora_ckpt "$OPENSORA_CKPT"
  --train_jsonl "$TRAIN_JSONL"
  --adapter_root "$ADAPTER_ROOT"
  --epochs "$EPOCHS"
  --batch_size "$BATCH_SIZE"
  --lr "$LR"
  --lambda_tox "$LAMBDA_TOX"
  --lambda_preserve "$LAMBDA_PRESERVE"
)

# Gradient checkpointing与LoRA训练不兼容，强制禁用
# 如果需要节省内存，请使用其他方法（减小batch_size、使用FP16等）
echo "✅ Gradient checkpointing已禁用（与LoRA训练不兼容）"
# 不添加 --use_gradient_checkpointing 参数

# 添加多GPU选项
if [ "$USE_MULTI_GPU" = "1" ]; then
  CMD_ARGS+=(--multi_gpu)
fi

# 设置PyTorch内存分配配置
export PYTORCH_CUDA_ALLOC_CONF

# 运行训练
CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python opensora_safe_adapter.py "${CMD_ARGS[@]}"

echo "=========================================="
echo " ✅ Open-Sora v2 Safe Adapter 训练完成"
echo " Adapters 保存于: $ADAPTER_ROOT/<category>/"
echo "=========================================="

