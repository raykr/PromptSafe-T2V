#!/usr/bin/env python3
"""
生成评估结果汇总表格
显示每个模型、每个类别上每个指标的统计值（均值、标准差、中位数）
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse
from typing import Dict, List, Tuple

# 指标列表
METRICS = [
    'CLIP_Original', 'CLIP_Defended', 'CLIP_Diff',
    'Flow_Original', 'Flow_Defended', 'Flow_Diff',
    'LPIPS_Original', 'LPIPS_Defended', 'LPIPS_Diff'
]

# 统计函数
STATS = ['mean', 'std', 'median']


def load_all_results(base_dir: Path, models: List[str], categories: List[str]) -> pd.DataFrame:
    """加载所有模型的评估结果"""
    all_data = []
    
    for model in models:
        for category in categories:
            csv_path = base_dir / model / category / "evaluation_summary.csv"
            
            if not csv_path.exists():
                print(f"⚠️  跳过: {csv_path} (文件不存在)")
                continue
            
            try:
                df = pd.read_csv(csv_path)
                df['Model'] = model
                df['Category'] = category
                all_data.append(df)
                print(f"✓ 加载: {model}/{category} ({len(df)} 条记录)")
            except Exception as e:
                print(f"❌ 错误: 无法读取 {csv_path}: {e}")
    
    if not all_data:
        raise ValueError("没有找到任何评估结果文件")
    
    return pd.concat(all_data, ignore_index=True)


def calculate_statistics(df: pd.DataFrame, models: List[str], categories: List[str]) -> pd.DataFrame:
    """计算每个模型+类别组合的统计值"""
    results = []
    
    for model in models:
        for category in categories:
            subset = df[(df['Model'] == model) & (df['Category'] == category)]
            
            if len(subset) == 0:
                continue
            
            row = {
                'Model': model,
                'Category': category,
                'N': len(subset)
            }
            
            # 计算每个指标的统计值
            for metric in METRICS:
                if metric not in subset.columns:
                    continue
                
                values = pd.to_numeric(subset[metric], errors='coerce').dropna()
                
                if len(values) > 0:
                    row[f'{metric}_mean'] = values.mean()
                    row[f'{metric}_std'] = values.std()
                    row[f'{metric}_median'] = values.median()
                else:
                    row[f'{metric}_mean'] = np.nan
                    row[f'{metric}_std'] = np.nan
                    row[f'{metric}_median'] = np.nan
            
            results.append(row)
    
    return pd.DataFrame(results)


def format_table_for_display(df: pd.DataFrame) -> str:
    """格式化表格用于显示"""
    lines = []
    
    # 按模型分组
    for model in df['Model'].unique():
        model_df = df[df['Model'] == model].copy()
        lines.append(f"\n{'='*120}")
        lines.append(f"模型: {model}")
        lines.append(f"{'='*120}")
        
        # 表头
        header = f"{'Category':<15} {'N':<5}"
        for metric in METRICS:
            header += f" {metric[:15]:<15}"
        lines.append(header)
        lines.append("-" * 120)
        
        # 数据行（只显示均值）
        for _, row in model_df.iterrows():
            line = f"{row['Category']:<15} {int(row['N']):<5}"
            for metric in METRICS:
                mean_col = f'{metric}_mean'
                if mean_col in row and pd.notna(row[mean_col]):
                    line += f" {row[mean_col]:>15.4f}"
                else:
                    line += f" {'N/A':>15}"
            lines.append(line)
    
    return "\n".join(lines)


def create_detailed_table(df: pd.DataFrame) -> pd.DataFrame:
    """创建详细表格（包含所有统计值）"""
    detailed_rows = []
    
    for _, row in df.iterrows():
        model = row['Model']
        category = row['Category']
        n = row['N']
        
        for metric in METRICS:
            mean_col = f'{metric}_mean'
            std_col = f'{metric}_std'
            median_col = f'{metric}_median'
            
            if mean_col in row:
                detailed_rows.append({
                    'Model': model,
                    'Category': category,
                    'Metric': metric,
                    'N': n,
                    'Mean': row[mean_col] if pd.notna(row[mean_col]) else np.nan,
                    'Std': row[std_col] if pd.notna(row[std_col]) else np.nan,
                    'Median': row[median_col] if pd.notna(row[median_col]) else np.nan,
                })
    
    return pd.DataFrame(detailed_rows)


def main():
    parser = argparse.ArgumentParser(description="生成评估结果汇总表格")
    parser.add_argument(
        "--base_dir",
        type=str,
        default="/home/raykr/projects/PromptSafe-T2V/evaluation_results",
        help="评估结果基础目录"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="输出目录（默认：base_dir/merged）"
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=["cogvideox-2b", "CogVideoX-5b", "CogVideoX1.5-5B"],
        help="模型列表"
    )
    parser.add_argument(
        "--categories",
        type=str,
        nargs="+",
        default=["disturbing", "political", "sexual", "violent", "benign"],
        help="类别列表"
    )
    
    args = parser.parse_args()
    
    base_dir = Path(args.base_dir)
    output_dir = Path(args.output_dir) if args.output_dir else base_dir / "merged"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*120)
    print("生成评估结果汇总表格")
    print("="*120)
    print(f"基础目录: {base_dir}")
    print(f"输出目录: {output_dir}")
    print(f"模型: {args.models}")
    print(f"类别: {args.categories}")
    print("="*120)
    print()
    
    # 1. 加载所有数据
    print("📖 加载评估结果...")
    all_data = load_all_results(base_dir, args.models, args.categories)
    print(f"✓ 共加载 {len(all_data)} 条记录\n")
    
    # 2. 计算统计值
    print("📊 计算统计值...")
    stats_df = calculate_statistics(all_data, args.models, args.categories)
    print(f"✓ 计算完成 ({len(stats_df)} 个模型+类别组合)\n")
    
    # 3. 保存汇总表格（宽格式：每个指标一列）
    summary_csv = output_dir / "summary_table_wide.csv"
    stats_df.to_csv(summary_csv, index=False, float_format='%.4f')
    print(f"✓ 汇总表格已保存: {summary_csv}")
    
    # 4. 创建详细表格（长格式：每个指标一行）
    detailed_df = create_detailed_table(stats_df)
    detailed_csv = output_dir / "summary_table_detailed.csv"
    detailed_df.to_csv(detailed_csv, index=False, float_format='%.4f')
    print(f"✓ 详细表格已保存: {detailed_csv}")
    
    # 5. 生成可读的文本报告
    report_txt = output_dir / "summary_table_report.txt"
    with open(report_txt, 'w', encoding='utf-8') as f:
        f.write("="*120 + "\n")
        f.write("T2V安全Adapter防御效果评估 - 汇总统计表格\n")
        f.write("="*120 + "\n")
        f.write(f"\n生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"总记录数: {len(all_data)}\n")
        f.write(f"模型数: {len(args.models)}\n")
        f.write(f"类别数: {len(args.categories)}\n")
        f.write("\n" + "="*120 + "\n")
        f.write("说明:\n")
        f.write("  - N: 样本数量\n")
        f.write("  - 每个指标显示均值（Mean）、标准差（Std）、中位数（Median）\n")
        f.write("  - CLIP Score: 范围 [0, 100]，越高越好\n")
        f.write("  - Flow Consistency: 范围 [0, 1]，越高越好\n")
        f.write("  - LPIPS: 范围 [0, 1]，越低越好\n")
        f.write("="*120 + "\n\n")
        
        # 按模型分组显示
        for model in args.models:
            model_df = stats_df[stats_df['Model'] == model].copy()
            if len(model_df) == 0:
                continue
            
            f.write(f"\n{'='*120}\n")
            f.write(f"模型: {model}\n")
            f.write(f"{'='*120}\n\n")
            
            # 为每个类别创建表格
            for category in args.categories:
                cat_df = model_df[model_df['Category'] == category]
                if len(cat_df) == 0:
                    continue
                
                row = cat_df.iloc[0]
                f.write(f"类别: {category} (N={int(row['N'])})\n")
                f.write("-" * 120 + "\n")
                
                # 按指标分组显示
                for metric in METRICS:
                    mean_col = f'{metric}_mean'
                    std_col = f'{metric}_std'
                    median_col = f'{metric}_median'
                    
                    if mean_col in row and pd.notna(row[mean_col]):
                        f.write(f"  {metric:20s}: Mean={row[mean_col]:>10.4f}, Std={row[std_col]:>10.4f}, Median={row[median_col]:>10.4f}\n")
                
                f.write("\n")
    
    print(f"✓ 文本报告已保存: {report_txt}")
    
    # 6. 生成LaTeX表格（可选）
    latex_file = output_dir / "summary_table_latex.tex"
    with open(latex_file, 'w', encoding='utf-8') as f:
        f.write("\\begin{table*}[htbp]\n")
        f.write("\\centering\n")
        f.write("\\small\n")
        f.write("\\caption{T2V安全Adapter防御效果评估汇总统计}\n")
        f.write("\\label{tab:summary}\n")
        f.write("\\begin{tabular}{l" + "c" * (len(METRICS) + 2) + "}\n")
        f.write("\\toprule\n")
        f.write("Model & Category & N")
        for metric in METRICS:
            f.write(f" & {metric.replace('_', ' ')}")
        f.write(" \\\\\n")
        f.write("\\midrule\n")
        
        for model in args.models:
            model_df = stats_df[stats_df['Model'] == model].copy()
            for idx, (_, row) in enumerate(model_df.iterrows()):
                if idx > 0:
                    f.write(" & ")  # 继续同一模型
                else:
                    f.write(f"{model} & ")
                f.write(f"{row['Category']} & {int(row['N'])}")
                for metric in METRICS:
                    mean_col = f'{metric}_mean'
                    if mean_col in row and pd.notna(row[mean_col]):
                        f.write(f" & {row[mean_col]:.4f}")
                    else:
                        f.write(" & N/A")
                f.write(" \\\\\n")
        
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table*}\n")
    
    print(f"✓ LaTeX表格已保存: {latex_file}")
    
    # 7. 生成Markdown表格
    md_file = output_dir / "summary_table.md"
    with open(md_file, 'w', encoding='utf-8') as f:
        f.write("# T2V安全Adapter防御效果评估汇总统计\n\n")
        f.write(f"**生成时间**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"**总记录数**: {len(all_data)}\n\n")
        f.write("## 说明\n\n")
        f.write("- **N**: 样本数量\n")
        f.write("- 所有数值均为**均值**（Mean）\n")
        f.write("- **CLIP Score**: 范围 [0, 100]，越高越好\n")
        f.write("- **Flow Consistency**: 范围 [0, 1]，越高越好\n")
        f.write("- **LPIPS**: 范围 [0, 1]，越低越好\n")
        f.write("- **Diff列**: 显示变化值和百分比变化（↑表示增加，↓表示减少）\n\n")
        f.write("---\n\n")
        
        # 为每个模型创建表格
        for model in args.models:
            model_df = stats_df[stats_df['Model'] == model].copy()
            if len(model_df) == 0:
                continue
            
            f.write(f"## {model}\n\n")
            
            # 表头（为diff列添加百分比列）
            header = "| Category | N"
            separator = "|---------|----:"
            
            for metric in METRICS:
                metric_name = metric.replace('_', ' ')
                header += f" | {metric_name}"
                separator += " | ----:"
                
                # 如果是diff列，添加百分比列
                if metric.endswith('_Diff'):
                    base_metric = metric.replace('_Diff', '')
                    header += f" | {metric_name} %"
                    separator += " | ----:"
            
            header += " |\n"
            separator += " |\n"
            
            f.write(header)
            f.write(separator)
            
            # 数据行
            for category in args.categories:
                cat_df = model_df[model_df['Category'] == category]
                if len(cat_df) == 0:
                    continue
                
                row = cat_df.iloc[0]
                line = f"| {category} | {int(row['N'])}"
                
                for metric in METRICS:
                    mean_col = f'{metric}_mean'
                    if mean_col in row and pd.notna(row[mean_col]):
                        value = row[mean_col]
                        line += f" | {value:.4f}"
                        
                        # 如果是diff列，计算并添加百分比变化
                        if metric.endswith('_Diff'):
                            # 找到对应的Original列
                            if metric == 'CLIP_Diff':
                                original_col = 'CLIP_Original_mean'
                            elif metric == 'Flow_Diff':
                                original_col = 'Flow_Original_mean'
                            elif metric == 'LPIPS_Diff':
                                original_col = 'LPIPS_Original_mean'
                            else:
                                original_col = None
                            
                            if original_col and original_col in row and pd.notna(row[original_col]):
                                original_value = row[original_col]
                                if abs(original_value) > 1e-6:  # 避免除以0
                                    # 百分比变化 = (diff / original) * 100
                                    pct_change = (value / original_value) * 100
                                    arrow = "↑" if value > 0 else "↓" if value < 0 else ""
                                    line += f" | {abs(pct_change):.2f}% {arrow}"
                                else:
                                    line += " | N/A"
                            else:
                                line += " | N/A"
                    else:
                        line += " | N/A"
                        # 如果是diff列，也要添加N/A
                        if metric.endswith('_Diff'):
                            line += " | N/A"
                
                line += " |\n"
                f.write(line)
            
            f.write("\n")
        
        # 添加完整统计表格（包含均值、标准差、中位数）
        f.write("---\n\n")
        f.write("## 完整统计表格（包含均值、标准差、中位数）\n\n")
        f.write("> 格式: Mean (Std) [Median]\n\n")
        
        for model in args.models:
            model_df = stats_df[stats_df['Model'] == model].copy()
            if len(model_df) == 0:
                continue
            
            f.write(f"### {model}\n\n")
            
            # 表头（为diff列添加百分比列）
            header = "| Category | N"
            separator = "|---------|----:"
            
            for metric in METRICS:
                metric_name = metric.replace('_', ' ')
                header += f" | {metric_name}"
                separator += " | :----:"
                
                # 如果是diff列，添加百分比列
                if metric.endswith('_Diff'):
                    base_metric = metric.replace('_Diff', '')
                    header += f" | {metric_name} %"
                    separator += " | :----:"
            
            header += " |\n"
            separator += " |\n"
            
            f.write(header)
            f.write(separator)
            
            # 数据行（显示均值、标准差、中位数）
            for category in args.categories:
                cat_df = model_df[model_df['Category'] == category]
                if len(cat_df) == 0:
                    continue
                
                row = cat_df.iloc[0]
                line = f"| {category} | {int(row['N'])}"
                
                for metric in METRICS:
                    mean_col = f'{metric}_mean'
                    std_col = f'{metric}_std'
                    median_col = f'{metric}_median'
                    
                    if mean_col in row and pd.notna(row[mean_col]):
                        mean_val = row[mean_col]
                        std_val = row[std_col] if pd.notna(row[std_col]) else 0
                        median_val = row[median_col] if pd.notna(row[median_col]) else 0
                        line += f" | {mean_val:.4f} ({std_val:.4f}) [{median_val:.4f}]"
                        
                        # 如果是diff列，计算并添加百分比变化
                        if metric.endswith('_Diff'):
                            # 找到对应的Original列
                            if metric == 'CLIP_Diff':
                                original_col = 'CLIP_Original_mean'
                            elif metric == 'Flow_Diff':
                                original_col = 'Flow_Original_mean'
                            elif metric == 'LPIPS_Diff':
                                original_col = 'LPIPS_Original_mean'
                            else:
                                original_col = None
                            
                            if original_col and original_col in row and pd.notna(row[original_col]):
                                original_value = row[original_col]
                                if abs(original_value) > 1e-6:  # 避免除以0
                                    # 百分比变化 = (diff / original) * 100
                                    pct_change = (mean_val / original_value) * 100
                                    arrow = "↑" if mean_val > 0 else "↓" if mean_val < 0 else ""
                                    line += f" | {pct_change:+.2f}% {arrow}"
                                else:
                                    line += " | N/A"
                            else:
                                line += " | N/A"
                    else:
                        line += " | N/A"
                        # 如果是diff列，也要添加N/A
                        if metric.endswith('_Diff'):
                            line += " | N/A"
                
                line += " |\n"
                f.write(line)
            
            f.write("\n")
    
    print(f"✓ Markdown表格已保存: {md_file}")
    
    # 8. 打印简要汇总
    print("\n" + "="*120)
    print("📊 简要汇总（均值）")
    print("="*120)
    print(format_table_for_display(stats_df))
    print("="*120)
    
    print(f"\n✅ 所有表格已生成完成！")
    print(f"   输出目录: {output_dir}")


if __name__ == "__main__":
    main()
