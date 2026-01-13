#!/bin/bash

# 汇总有害内容评估统计结果
# 用法: bash scripts/summary_harmful_table.sh [目录路径]
# 例如: bash scripts/summary_harmful_table.sh /home/raykr/projects/PromptSafe-T2V/out/CogVideoX-5b/tiny

set -e

# 默认目录
DEFAULT_DIR="/home/raykr/projects/PromptSafe-T2V/out/CogVideoX1.5-5B/tiny"

# 获取输入目录
BASE_DIR="${1:-$DEFAULT_DIR}"

# 类别列表
CATEGORIES=("sexual" "violent" "political" "disturbing")

# 输出文件
OUTPUT_FILE="${BASE_DIR}/harmful_summary_table.md"

echo "======================================================================"
echo "生成有害内容评估汇总表格"
echo "======================================================================"
echo "基础目录: ${BASE_DIR}"
echo "输出文件: ${OUTPUT_FILE}"
echo "======================================================================"
echo ""

# 检查目录是否存在
if [ ! -d "${BASE_DIR}" ]; then
    echo "❌ 错误: 目录不存在 - ${BASE_DIR}"
    exit 1
fi

# 创建临时Python脚本来解析数据
TMP_SCRIPT=$(mktemp)
cat > "${TMP_SCRIPT}" << 'PYTHON_EOF'
#!/usr/bin/env python3
import re
import sys
from pathlib import Path

def parse_statistics_file(file_path):
    """解析eval_statistics.txt文件"""
    data = {}
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 提取视频级别统计信息（在"按视频对统计（视频级别）"部分）
        # 找到视频级别部分
        video_section = re.search(r'【按视频对统计（视频级别）】.*?【视频级别防御成功率】', content, re.DOTALL)
        if not video_section:
            video_section = re.search(r'【按视频对统计（视频级别）】.*', content, re.DOTALL)
        
        if video_section:
            section_content = video_section.group(0)
        else:
            section_content = content
        
        # 总测试视频对数: 85
        match = re.search(r'总测试视频对数:\s*(\d+)', section_content)
        if match:
            data['total'] = int(match.group(1))
        
        # 防御前有害视频数: 38 (44.71%)
        match = re.search(r'防御前有害视频数:\s*(\d+)\s*\(([\d.]+)%\)', section_content)
        if match:
            data['before_harmful'] = int(match.group(1))
            data['before_harmful_pct'] = float(match.group(2))
        
        # 防御后有害视频数: 22 (25.88%)
        match = re.search(r'防御后有害视频数:\s*(\d+)\s*\(([\d.]+)%\)', section_content)
        if match:
            data['after_harmful'] = int(match.group(1))
            data['after_harmful_pct'] = float(match.group(2))
        
        # 有害视频减少: 16 个 (18.82%)
        match = re.search(r'有害视频减少:\s*(\d+)\s*个\s*\(([\d.]+)%\)', section_content)
        if match:
            data['harmful_reduction'] = int(match.group(1))
            data['harmful_reduction_pct'] = float(match.group(2))
        
        # 在"视频级别防御成功率"部分查找
        defense_section = re.search(r'【视频级别防御成功率】.*?【视频级别有害程度变化】', content, re.DOTALL)
        if not defense_section:
            defense_section = re.search(r'【视频级别防御成功率】.*', content, re.DOTALL)
        
        if defense_section:
            defense_content = defense_section.group(0)
        else:
            defense_content = content
        
        # 防御成功（有害→无害）: 27 (71.05%)
        match = re.search(r'防御成功（有害→无害）:\s*(\d+)\s*\(([\d.]+)%\)', defense_content)
        if match:
            data['defense_success'] = int(match.group(1))
            data['defense_success_pct'] = float(match.group(2))
        
        # 防御失败（仍为有害）: 11 (28.95%)
        match = re.search(r'防御失败（仍为有害）:\s*(\d+)\s*\(([\d.]+)%\)', defense_content)
        if match:
            data['defense_fail'] = int(match.group(1))
            data['defense_fail_pct'] = float(match.group(2))
        
    except Exception as e:
        print(f"错误: 无法解析文件 {file_path}: {e}", file=sys.stderr)
    
    return data

def main():
    base_dir = Path(sys.argv[1])
    categories = sys.argv[2].split(',')
    output_file = Path(sys.argv[3])
    
    results = []
    
    for category in categories:
        stats_file = base_dir / category / "eval_statistics.txt"
        
        if not stats_file.exists():
            print(f"⚠️  警告: 文件不存在 - {stats_file}", file=sys.stderr)
            continue
        
        data = parse_statistics_file(stats_file)
        
        if data:
            results.append({
                'category': category,
                'data': data
            })
            print(f"✓ 解析完成: {category}")
        else:
            print(f"⚠️  警告: 无法解析 - {category}", file=sys.stderr)
    
    # 生成Markdown表格
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("# 有害内容评估汇总统计\n\n")
        f.write(f"**生成时间**: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"**数据来源**: {base_dir}\n\n")
        f.write("---\n\n")
        
        # 表头
        f.write("| 类别 | 总数 | baseline 有害数 | defense 有害数 | 有害视频减少量 | 防御成功（有害→无害） | 防御失败（仍为有害） |\n")
        f.write("| ---------- | --- | ------------ | ----------- | ---------------- | --------------- | ---------- |\n")
        
        # 数据行
        for item in results:
            cat = item['category']
            d = item['data']
            
            total = d.get('total', 0)
            before_harmful = d.get('before_harmful', 0)
            before_pct = d.get('before_harmful_pct', 0)
            after_harmful = d.get('after_harmful', 0)
            after_pct = d.get('after_harmful_pct', 0)
            reduction = d.get('harmful_reduction', 0)
            reduction_pct = d.get('harmful_reduction_pct', 0)
            success = d.get('defense_success', 0)
            success_pct = d.get('defense_success_pct', 0)
            fail = d.get('defense_fail', 0)
            fail_pct = d.get('defense_fail_pct', 0)
            
            f.write(f"| {cat} | {total} | {before_harmful} ({before_pct:.2f}%) | {after_harmful} ({after_pct:.2f}%) | {reduction} (**{reduction_pct:.2f}%**) | {success} (**{success_pct:.2f}%**) | {fail} ({fail_pct:.2f}%) |\n")
    
    print(f"\n✅ 汇总表格已生成: {output_file}")

if __name__ == "__main__":
    main()
PYTHON_EOF

# 运行Python脚本
python3 "${TMP_SCRIPT}" "${BASE_DIR}" "$(IFS=,; echo "${CATEGORIES[*]}")" "${OUTPUT_FILE}"

# 清理临时文件
rm -f "${TMP_SCRIPT}"

echo ""
echo "======================================================================"
echo "✅ 完成！"
echo "======================================================================"
