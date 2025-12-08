#!/bin/bash
# 训练单个分类的示例脚本
# 使用方法: ./train_single.sh sexual

CATEGORY=${1:-sexual}  # 默认使用sexual，也可以通过参数传入

echo "训练分类: $CATEGORY"

python train_adapter.py \
    --category "$CATEGORY" \
    --num_epochs 100 \
    --batch_size 8 \
    --save_every 5
