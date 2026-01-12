#!/bin/bash
# VideoCrafter2 Safe Adapter 训练脚本
# 训练所有有害类别的LoRA适配器

# 设置错误时退出
set -e

# ==================== 配置参数 ====================
# VideoCrafter2 相关路径
VIDEOCRAFTER_REPO="${VIDEOCRAFTER_REPO:-/path/to/VideoCrafter2}"  # VideoCrafter2仓库路径，请修改为实际路径
CONFIG_PATH="${CONFIG_PATH:-configs/inference_t2v_512_v2.0.yaml}"  # 配置文件路径，请修改为实际路径
CKPT_PATH="${CKPT_PATH:-/home/raykr/models/VideoCrafter/VideoCrafter2}"  # Checkpoint路径（目录或文件）

# 训练数据路径（JSONL格式，每行: {"prompt": "...", "label": "sexual"}）
TRAIN_JSONL="${TRAIN_JSONL:-datasets/train/train.jsonl}"

# Adapter保存路径
ADAPTER_ROOT="${ADAPTER_ROOT:-checkpoints/videocrafter2_adapters}"

# 训练超参数
EPOCHS="${EPOCHS:-10}"
BATCH_SIZE="${BATCH_SIZE:-4}"
LR="${LR:-1e-4}"
LAMBDA_TOX="${LAMBDA_TOX:-1.0}"
LAMBDA_PRESERVE="${LAMBDA_PRESERVE:-0.1}"

# 文本编码器输出维度（根据VideoCrafter2的cond_stage_model调整，常见值：768, 1024, 4096）
EMBED_DIM="${EMBED_DIM:-768}"

# GPU设置
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# ==================== 参数检查 ====================
echo "=========================================="
echo "VideoCrafter2 Safe Adapter 训练"
echo "=========================================="
echo "VideoCrafter2 仓库: $VIDEOCRAFTER_REPO"
echo "配置文件: $CONFIG_PATH"
echo "Checkpoint: $CKPT_PATH"
echo "训练数据: $TRAIN_JSONL"
echo "Adapter保存路径: $ADAPTER_ROOT"
echo "训练轮数: $EPOCHS"
echo "批次大小: $BATCH_SIZE"
echo "学习率: $LR"
echo "Lambda Tox: $LAMBDA_TOX"
echo "Lambda Preserve: $LAMBDA_PRESERVE"
echo "Embed Dim: $EMBED_DIM"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "=========================================="

# 检查必要文件
if [ ! -d "$VIDEOCRAFTER_REPO" ]; then
    echo "❌ 错误: VideoCrafter2仓库不存在: $VIDEOCRAFTER_REPO"
    echo "请设置 VIDEOCRAFTER_REPO 环境变量或修改脚本中的路径"
    exit 1
fi

if [ ! -f "$CONFIG_PATH" ]; then
    echo "❌ 错误: 配置文件不存在: $CONFIG_PATH"
    echo "请设置 CONFIG_PATH 环境变量或修改脚本中的路径"
    exit 1
fi

if [ ! -f "$TRAIN_JSONL" ] && [ ! -d "$CKPT_PATH" ] && [ ! -f "$CKPT_PATH" ]; then
    echo "⚠️  警告: 训练数据文件不存在: $TRAIN_JSONL"
    echo "请确保数据文件存在或修改 TRAIN_JSONL 路径"
fi

# ==================== 开始训练 ====================
echo ""
echo "开始训练所有类别的适配器..."
echo ""

# 运行训练
CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python videocrafter2_safe_adapter.py \
    --mode train \
    --videocrafter_repo "$VIDEOCRAFTER_REPO" \
    --config "$CONFIG_PATH" \
    --ckpt "$CKPT_PATH" \
    --train_jsonl "$TRAIN_JSONL" \
    --adapter_root "$ADAPTER_ROOT" \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --lr "$LR" \
    --lambda_tox "$LAMBDA_TOX" \
    --lambda_preserve "$LAMBDA_PRESERVE" \
    --embed_dim "$EMBED_DIM"

echo ""
echo "=========================================="
echo "✅ 所有类别训练完成！"
echo "Adapter保存在: $ADAPTER_ROOT"
echo "=========================================="
