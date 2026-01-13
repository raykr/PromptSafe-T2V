#!/bin/bash
# 创建Open-Sora推理专用conda环境
# 避免与CogVideo环境冲突

set -e

ENV_NAME="opensora_inference"
PYTHON_VERSION="3.10"

echo "=========================================="
echo " 创建Open-Sora推理环境: $ENV_NAME"
echo "=========================================="

# 检查conda是否可用
if ! command -v conda &> /dev/null; then
    echo "❌ 错误: conda未安装或不在PATH中"
    exit 1
fi

# 检查环境是否已存在
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "⚠️  环境 $ENV_NAME 已存在"
    read -p "是否删除并重新创建? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "删除现有环境..."
        conda env remove -n $ENV_NAME -y
    else
        echo "使用现有环境"
        echo "激活环境: conda activate $ENV_NAME"
        exit 0
    fi
fi

# 创建新环境
echo "创建conda环境: $ENV_NAME (Python $PYTHON_VERSION)..."
conda create -n $ENV_NAME python=$PYTHON_VERSION -y

# 激活环境并安装依赖
echo ""
echo "安装依赖包..."
eval "$(conda shell.bash hook)"
conda activate $ENV_NAME

# 安装PyTorch和相关包（使用conda，更稳定）
echo "安装PyTorch和torchvision（使用conda）..."
conda install pytorch==2.4.0 torchvision==0.19.0 pytorch-cuda=12.1 -c pytorch -c nvidia -y

# 安装Open-Sora核心依赖
echo "安装Open-Sora依赖..."
pip install mmengine>=0.10.3
pip install colossalai>=0.4.4
pip install ftfy>=6.2.0
pip install accelerate>=0.29.2

# 安装其他必要依赖
echo "安装其他依赖..."
pip install transformers peft imageio numpy pandas av==13.1.0
pip install omegaconf>=2.3.0

# 安装项目特定依赖（如果有requirements.txt）
if [ -f "requirements.txt" ]; then
    echo "安装项目依赖..."
    pip install -r requirements.txt
fi

echo ""
echo "=========================================="
echo " ✅ 环境创建完成！"
echo "=========================================="
echo ""
echo "激活环境:"
echo "  conda activate $ENV_NAME"
echo ""
echo "验证安装:"
echo "  python -c 'import torch; import mmengine; import colossalai; print(\"✅ 所有依赖已安装\")'"
echo ""
echo "运行推理:"
echo "  conda activate $ENV_NAME"
echo "  bash scripts/infer_opensora_v2_safe_adapter.sh"
echo ""
