#!/bin/bash

# 生成评估结果汇总表格
# 用法: bash scripts/count/summary_quality_table.sh

BASE_DIR="/home/raykr/projects/PromptSafe-T2V"

echo "======================================================================"
echo "生成评估结果汇总表格"
echo "======================================================================"
echo ""

python3 "${BASE_DIR}/generate_summary_table.py" \
    --base_dir "${BASE_DIR}/evaluation_results" \
    --models "cogvideox-2b" "CogVideoX-5b" "CogVideoX1.5-5B" \
    --categories "disturbing" "political" "sexual" "violent" "benign"

echo ""
echo "======================================================================"
echo "✅ 汇总表格生成完成！"
echo "======================================================================"
