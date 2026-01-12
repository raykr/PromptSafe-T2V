#!/bin/bash
# 批量训练所有分类的SafeAdapter模型

# 设置错误时退出
set -e

# 分类列表
categories=("sexual" "violence" "political" "disturbing")

# 遍历每个分类进行训练
for category in "${categories[@]}"; do
    echo "=========================================="
    echo "开始训练分类: $category"
    echo "=========================================="
    
    python train_adapter.py \
        --category "$category" \
        --model_path "/home/raykr/models/zai-org/CogVideoX-2b" \
        --hidden_size 4096 \
        --rank 256 \
        --lr 5e-4 \
        --num_epochs 100 \
        --batch_size 8 \
        --margin 0.1 \
        --lam_benign 0.1 \
        --device cuda \
        --save_every 5
    
    echo "✅ 分类 $category 训练完成"
    echo ""
done

echo "=========================================="
echo "所有分类训练完成！"
echo "=========================================="
