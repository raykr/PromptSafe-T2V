#!/bin/bash
# VideoCrafter2 Safe Adapter 并行训练脚本
# 使用多个GPU并行训练不同类别的适配器

# 设置错误时退出
set -e

# ==================== 配置参数 ====================
# VideoCrafter2 相关路径
VIDEOCRAFTER_REPO="${VIDEOCRAFTER_REPO:-/path/to/VideoCrafter2}"  # 请修改为实际路径
CONFIG_PATH="${CONFIG_PATH:-configs/inference_t2v_512_v2.0.yaml}"  # 请修改为实际路径
CKPT_PATH="${CKPT_PATH:-/home/raykr/models/VideoCrafter/VideoCrafter2}"

# 训练数据路径
TRAIN_JSONL="${TRAIN_JSONL:-datasets/train/train.jsonl}"

# Adapter保存路径
ADAPTER_ROOT="${ADAPTER_ROOT:-checkpoints/videocrafter2_adapters}"

# 训练超参数
EPOCHS="${EPOCHS:-10}"
BATCH_SIZE="${BATCH_SIZE:-4}"
LR="${LR:-1e-4}"
LAMBDA_TOX="${LAMBDA_TOX:-1.0}"
LAMBDA_PRESERVE="${LAMBDA_PRESERVE:-0.1}"
EMBED_DIM="${EMBED_DIM:-768}"

# GPU分配（根据你的GPU数量调整）
# 示例：2个GPU，每个训练2个类别
GPU_SEXUAL="${GPU_SEXUAL:-0}"
GPU_VIOLENT="${GPU_VIOLENT:-0}"
GPU_POLITICAL="${GPU_POLITICAL:-1}"
GPU_DISTURBING="${GPU_DISTURBING:-1}"

# ==================== 训练函数 ====================
train_category() {
    local category=$1
    local gpu=$2
    
    echo "=========================================="
    echo "开始训练类别: $category (GPU: $gpu)"
    echo "=========================================="
    
    CUDA_VISIBLE_DEVICES=$gpu python videocrafter2_safe_adapter.py \
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
        --embed_dim "$EMBED_DIM" \
        > "logs/train_${category}.log" 2>&1 &
    
    echo "✅ 类别 $category 训练任务已启动 (PID: $!)"
}

# ==================== 创建日志目录 ====================
mkdir -p logs

# ==================== 开始并行训练 ====================
echo "=========================================="
echo "VideoCrafter2 Safe Adapter 并行训练"
echo "=========================================="
echo "训练数据: $TRAIN_JSONL"
echo "Adapter保存路径: $ADAPTER_ROOT"
echo "训练轮数: $EPOCHS"
echo "批次大小: $BATCH_SIZE"
echo "学习率: $LR"
echo "=========================================="
echo ""

# 启动第一批训练任务（可以并行）
echo "启动第一批训练任务..."
train_category "sexual" "$GPU_SEXUAL"
train_category "violent" "$GPU_VIOLENT"

# 等待第一批完成
echo ""
echo "等待第一批训练任务完成..."
wait

echo "✅ 第一批训练完成 (sexual, violent)"
echo ""

# 启动第二批训练任务
echo "启动第二批训练任务..."
train_category "political" "$GPU_POLITICAL"
train_category "disturbing" "$GPU_DISTURBING"

# 等待第二批完成
echo ""
echo "等待第二批训练任务完成..."
wait

echo ""
echo "=========================================="
echo "✅ 所有类别训练完成！"
echo "Adapter保存在: $ADAPTER_ROOT"
echo "训练日志保存在: logs/"
echo "=========================================="
