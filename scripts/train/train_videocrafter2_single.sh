#!/bin/bash
# VideoCrafter2 Safe Adapter 单类别训练脚本
# 使用方法: ./train_videocrafter2_single.sh sexual

# 设置错误时退出
set -e

# ==================== 配置参数 ====================
# 类别参数（从命令行获取或使用默认值）
CATEGORY="${1:-sexual}"  # 默认使用sexual，也可以通过参数传入: sexual, violent, political, disturbing

# VideoCrafter2 相关路径
VIDEOCRAFTER_REPO="${VIDEOCRAFTER_REPO:-/path/to/VideoCrafter2}"  # 请修改为实际路径
CONFIG_PATH="${CONFIG_PATH:-configs/inference_t2v_512_v2.0.yaml}"  # 请修改为实际路径
CKPT_PATH="${CKPT_PATH:-/home/raykr/models/VideoCrafter/VideoCrafter2}"

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

# 文本编码器输出维度
EMBED_DIM="${EMBED_DIM:-768}"

# GPU设置
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# ==================== 参数检查 ====================
echo "=========================================="
echo "训练分类: $CATEGORY"
echo "=========================================="
echo "VideoCrafter2 仓库: $VIDEOCRAFTER_REPO"
echo "配置文件: $CONFIG_PATH"
echo "Checkpoint: $CKPT_PATH"
echo "训练数据: $TRAIN_JSONL"
echo "Adapter保存路径: $ADAPTER_ROOT"
echo "训练轮数: $EPOCHS"
echo "批次大小: $BATCH_SIZE"
echo "学习率: $LR"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "=========================================="

# 检查类别是否有效
VALID_CATEGORIES=("sexual" "violent" "political" "disturbing")
if [[ ! " ${VALID_CATEGORIES[@]} " =~ " ${CATEGORY} " ]]; then
    echo "❌ 错误: 无效的类别 '$CATEGORY'"
    echo "有效类别: ${VALID_CATEGORIES[@]}"
    exit 1
fi

# 检查必要文件
if [ ! -d "$VIDEOCRAFTER_REPO" ]; then
    echo "❌ 错误: VideoCrafter2仓库不存在: $VIDEOCRAFTER_REPO"
    exit 1
fi

if [ ! -f "$CONFIG_PATH" ]; then
    echo "❌ 错误: 配置文件不存在: $CONFIG_PATH"
    exit 1
fi

# ==================== 开始训练 ====================
echo ""
echo "开始训练类别: $CATEGORY"
echo ""

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
echo "✅ 类别 $CATEGORY 训练完成！"
echo "Adapter保存在: $ADAPTER_ROOT/$CATEGORY"
echo "=========================================="
