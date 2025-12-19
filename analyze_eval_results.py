#!/usr/bin/env python3
"""
分析GPT评估结果CSV文件，统计防御前后有害内容的变化情况
"""
import pandas as pd
import argparse
from pathlib import Path


def analyze_eval_results(csv_path: str, output_path: str = None):
    """
    分析评估结果CSV文件
    
    Args:
        csv_path: 输入的CSV文件路径
        output_path: 可选，输出统计报告的路径
    """
    df = pd.read_csv(csv_path)
    
    print("=" * 80)
    print(f"📊 评估结果分析: {csv_path}")
    print("=" * 80)
    
    # 1. 总体统计
    print("\n【总体统计】")
    print("-" * 80)
    total_rows = len(df)
    before_harmful = df['before_nsfw'].sum()
    after_harmful = df['after_nsfw'].sum()
    before_harmful_ratio = before_harmful / total_rows * 100
    after_harmful_ratio = after_harmful / total_rows * 100
    
    print(f"总评估条目数: {total_rows}")
    print(f"\n防御前:")
    print(f"  - 有害数量: {before_harmful} ({before_harmful_ratio:.2f}%)")
    print(f"  - 无害数量: {total_rows - before_harmful} ({(100 - before_harmful_ratio):.2f}%)")
    print(f"\n防御后:")
    print(f"  - 有害数量: {after_harmful} ({after_harmful_ratio:.2f}%)")
    print(f"  - 无害数量: {total_rows - after_harmful} ({(100 - after_harmful_ratio):.2f}%)")
    
    # 防御效果
    harmful_reduction = before_harmful - after_harmful
    harmful_reduction_ratio = harmful_reduction / total_rows * 100 if total_rows > 0 else 0
    print(f"\n防御效果:")
    print(f"  - 有害内容减少: {harmful_reduction} 条 ({harmful_reduction_ratio:.2f}%)")
    print(f"  - 有害比例下降: {before_harmful_ratio - after_harmful_ratio:.2f} 个百分点")
    
    # 2. 防御成功率（从有害变为无害）
    print("\n【防御成功率】")
    print("-" * 80)
    # 防御成功：before_nsfw=True 且 after_nsfw=False
    defense_success = ((df['before_nsfw'] == True) & (df['after_nsfw'] == False)).sum()
    defense_fail = ((df['before_nsfw'] == True) & (df['after_nsfw'] == True)).sum()
    defense_success_ratio = defense_success / before_harmful * 100 if before_harmful > 0 else 0
    
    print(f"防御前有害的条目数: {before_harmful}")
    print(f"  - 防御成功（有害→无害）: {defense_success} ({defense_success_ratio:.2f}%)")
    print(f"  - 防御失败（仍为有害）: {defense_fail} ({100 - defense_success_ratio:.2f}%)")
    
    # 3. 有害程度变化（Yes百分比）
    print("\n【有害程度变化（Yes百分比）】")
    print("-" * 80)
    mean_before_yes = df['before_yes'].mean()
    mean_after_yes = df['after_yes'].mean()
    mean_delta_yes = df['delta_yes'].mean()
    
    print(f"平均Yes百分比:")
    print(f"  - 防御前: {mean_before_yes:.2f}%")
    print(f"  - 防御后: {mean_after_yes:.2f}%")
    print(f"  - 平均下降: {mean_delta_yes:.2f} 个百分点")
    
    # 只统计防御前有害的条目
    harmful_before = df[df['before_nsfw'] == True]
    if len(harmful_before) > 0:
        mean_before_yes_harmful = harmful_before['before_yes'].mean()
        mean_after_yes_harmful = harmful_before['after_yes'].mean()
        mean_delta_yes_harmful = harmful_before['delta_yes'].mean()
        print(f"\n仅统计防御前有害的条目 ({len(harmful_before)} 条):")
        print(f"  - 防御前平均Yes: {mean_before_yes_harmful:.2f}%")
        print(f"  - 防御后平均Yes: {mean_after_yes_harmful:.2f}%")
        print(f"  - 平均下降: {mean_delta_yes_harmful:.2f} 个百分点")
    
    # 4. 按aspect分类统计
    print("\n【按评估方面（Aspect）分类统计】")
    print("-" * 80)
    
    aspect_stats = []
    for aspect in df['aspect'].unique():
        aspect_df = df[df['aspect'] == aspect]
        aspect_total = len(aspect_df)
        aspect_before_harmful = aspect_df['before_nsfw'].sum()
        aspect_after_harmful = aspect_df['after_nsfw'].sum()
        aspect_defense_success = ((aspect_df['before_nsfw'] == True) & 
                                  (aspect_df['after_nsfw'] == False)).sum()
        aspect_mean_delta = aspect_df['delta_yes'].mean()
        
        aspect_stats.append({
            'aspect': aspect,
            'total': aspect_total,
            'before_harmful': aspect_before_harmful,
            'before_harmful_ratio': aspect_before_harmful / aspect_total * 100,
            'after_harmful': aspect_after_harmful,
            'after_harmful_ratio': aspect_after_harmful / aspect_total * 100,
            'defense_success': aspect_defense_success,
            'defense_success_ratio': aspect_defense_success / aspect_before_harmful * 100 if aspect_before_harmful > 0 else 0,
            'mean_delta_yes': aspect_mean_delta,
        })
    
    aspect_df_stats = pd.DataFrame(aspect_stats)
    print(aspect_df_stats.to_string(index=False))
    
    # 5. 按视频对统计（视频级别）
    print("\n【按视频对统计（视频级别）】")
    print("-" * 80)
    
    # 按idx分组，判断每个视频对是否在任何一个aspect上被判定为有害
    video_stats = df.groupby('idx').agg({
        'before_nsfw': 'any',  # 如果任何一个aspect被判定为有害，则视频有害
        'after_nsfw': 'any',
        'before_yes': 'mean',  # 该视频在所有aspect上的平均Yes百分比
        'after_yes': 'mean',
        'delta_yes': 'mean',
    }).reset_index()
    
    total_videos = len(video_stats)
    videos_before_harmful = video_stats['before_nsfw'].sum()
    videos_after_harmful = video_stats['after_nsfw'].sum()
    videos_before_harmful_ratio = videos_before_harmful / total_videos * 100
    videos_after_harmful_ratio = videos_after_harmful / total_videos * 100
    
    print(f"总测试视频对数: {total_videos}")
    print(f"\n防御前:")
    print(f"  - 有害视频数: {videos_before_harmful} ({videos_before_harmful_ratio:.2f}%)")
    print(f"  - 无害视频数: {total_videos - videos_before_harmful} ({(100 - videos_before_harmful_ratio):.2f}%)")
    print(f"\n防御后:")
    print(f"  - 有害视频数: {videos_after_harmful} ({videos_after_harmful_ratio:.2f}%)")
    print(f"  - 无害视频数: {total_videos - videos_after_harmful} ({(100 - videos_after_harmful_ratio):.2f}%)")
    
    # 视频级别的防御效果
    videos_harmful_reduction = videos_before_harmful - videos_after_harmful
    videos_harmful_reduction_ratio = videos_harmful_reduction / total_videos * 100 if total_videos > 0 else 0
    print(f"\n防御效果:")
    print(f"  - 有害视频减少: {videos_harmful_reduction} 个 ({videos_harmful_reduction_ratio:.2f}%)")
    print(f"  - 有害比例下降: {videos_before_harmful_ratio - videos_after_harmful_ratio:.2f} 个百分点")
    
    # 视频级别的防御成功率
    videos_defense_success = ((video_stats['before_nsfw'] == True) & 
                              (video_stats['after_nsfw'] == False)).sum()
    videos_defense_fail = ((video_stats['before_nsfw'] == True) & 
                           (video_stats['after_nsfw'] == True)).sum()
    videos_defense_success_ratio = videos_defense_success / videos_before_harmful * 100 if videos_before_harmful > 0 else 0
    
    print(f"\n视频级别防御成功率:")
    print(f"  防御前有害的视频数: {videos_before_harmful}")
    print(f"  - 防御成功（有害→无害）: {videos_defense_success} ({videos_defense_success_ratio:.2f}%)")
    print(f"  - 防御失败（仍为有害）: {videos_defense_fail} ({100 - videos_defense_success_ratio:.2f}%)")
    
    # 视频级别的有害程度变化
    videos_mean_before_yes = video_stats['before_yes'].mean()
    videos_mean_after_yes = video_stats['after_yes'].mean()
    videos_mean_delta_yes = video_stats['delta_yes'].mean()
    
    print(f"\n视频级别平均Yes百分比:")
    print(f"  - 防御前: {videos_mean_before_yes:.2f}%")
    print(f"  - 防御后: {videos_mean_after_yes:.2f}%")
    print(f"  - 平均下降: {videos_mean_delta_yes:.2f} 个百分点")
    
    # 只统计防御前有害的视频
    videos_harmful_before = video_stats[video_stats['before_nsfw'] == True]
    if len(videos_harmful_before) > 0:
        videos_mean_before_yes_harmful = videos_harmful_before['before_yes'].mean()
        videos_mean_after_yes_harmful = videos_harmful_before['after_yes'].mean()
        videos_mean_delta_yes_harmful = videos_harmful_before['delta_yes'].mean()
        print(f"\n仅统计防御前有害的视频 ({len(videos_harmful_before)} 个):")
        print(f"  - 防御前平均Yes: {videos_mean_before_yes_harmful:.2f}%")
        print(f"  - 防御后平均Yes: {videos_mean_after_yes_harmful:.2f}%")
        print(f"  - 平均下降: {videos_mean_delta_yes_harmful:.2f} 个百分点")
    
    # 6. 详细变化情况
    print("\n【详细变化情况（条目级别）】")
    print("-" * 80)
    change_types = {
        '无害→无害': ((df['before_nsfw'] == False) & (df['after_nsfw'] == False)).sum(),
        '无害→有害': ((df['before_nsfw'] == False) & (df['after_nsfw'] == True)).sum(),
        '有害→无害': ((df['before_nsfw'] == True) & (df['after_nsfw'] == False)).sum(),
        '有害→有害': ((df['before_nsfw'] == True) & (df['after_nsfw'] == True)).sum(),
    }
    
    for change_type, count in change_types.items():
        ratio = count / total_rows * 100
        print(f"{change_type}: {count} ({ratio:.2f}%)")
    
    # 视频级别的详细变化情况
    print("\n【详细变化情况（视频级别）】")
    print("-" * 80)
    video_change_types = {
        '无害→无害': ((video_stats['before_nsfw'] == False) & (video_stats['after_nsfw'] == False)).sum(),
        '无害→有害': ((video_stats['before_nsfw'] == False) & (video_stats['after_nsfw'] == True)).sum(),
        '有害→无害': ((video_stats['before_nsfw'] == True) & (video_stats['after_nsfw'] == False)).sum(),
        '有害→有害': ((video_stats['before_nsfw'] == True) & (video_stats['after_nsfw'] == True)).sum(),
    }
    
    for change_type, count in video_change_types.items():
        ratio = count / total_videos * 100
        print(f"{change_type}: {count} ({ratio:.2f}%)")
    
    # 7. 保存统计报告
    if output_path:
        report_lines = [
            "=" * 80,
            f"评估结果统计报告: {csv_path}",
            "=" * 80,
            "",
            "【总体统计（条目级别）】",
            f"总评估条目数: {total_rows}",
            f"防御前有害数量: {before_harmful} ({before_harmful_ratio:.2f}%)",
            f"防御后有害数量: {after_harmful} ({after_harmful_ratio:.2f}%)",
            f"有害内容减少: {harmful_reduction} 条 ({harmful_reduction_ratio:.2f}%)",
            f"有害比例下降: {before_harmful_ratio - after_harmful_ratio:.2f} 个百分点",
            "",
            "【防御成功率（条目级别）】",
            f"防御成功（有害→无害）: {defense_success} ({defense_success_ratio:.2f}%)",
            f"防御失败（仍为有害）: {defense_fail} ({100 - defense_success_ratio:.2f}%)",
            "",
            "【有害程度变化（条目级别）】",
            f"平均Yes百分比 - 防御前: {mean_before_yes:.2f}%",
            f"平均Yes百分比 - 防御后: {mean_after_yes:.2f}%",
            f"平均下降: {mean_delta_yes:.2f} 个百分点",
            "",
            "【按视频对统计（视频级别）】",
            f"总测试视频对数: {total_videos}",
            f"防御前有害视频数: {videos_before_harmful} ({videos_before_harmful_ratio:.2f}%)",
            f"防御后有害视频数: {videos_after_harmful} ({videos_after_harmful_ratio:.2f}%)",
            f"有害视频减少: {videos_harmful_reduction} 个 ({videos_harmful_reduction_ratio:.2f}%)",
            f"有害比例下降: {videos_before_harmful_ratio - videos_after_harmful_ratio:.2f} 个百分点",
            "",
            "【视频级别防御成功率】",
            f"防御前有害的视频数: {videos_before_harmful}",
            f"防御成功（有害→无害）: {videos_defense_success} ({videos_defense_success_ratio:.2f}%)",
            f"防御失败（仍为有害）: {videos_defense_fail} ({100 - videos_defense_success_ratio:.2f}%)",
            "",
            "【视频级别有害程度变化】",
            f"平均Yes百分比 - 防御前: {videos_mean_before_yes:.2f}%",
            f"平均Yes百分比 - 防御后: {videos_mean_after_yes:.2f}%",
            f"平均下降: {videos_mean_delta_yes:.2f} 个百分点",
            "",
            "【按Aspect分类统计】",
            aspect_df_stats.to_string(index=False),
            "",
            "【详细变化情况（条目级别）】",
        ]
        for change_type, count in change_types.items():
            ratio = count / total_rows * 100
            report_lines.append(f"{change_type}: {count} ({ratio:.2f}%)")
        
        report_lines.extend([
            "",
            "【详细变化情况（视频级别）】",
        ])
        for change_type, count in video_change_types.items():
            ratio = count / total_videos * 100
            report_lines.append(f"{change_type}: {count} ({ratio:.2f}%)")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        print(f"\n✅ 统计报告已保存到: {output_path}")
    
    print("\n" + "=" * 80)


def main():
    parser = argparse.ArgumentParser(description='分析GPT评估结果CSV文件')
    parser.add_argument('--csv', type=str, required=True, help='输入的CSV文件路径')
    parser.add_argument('--output', type=str, default=None, help='输出统计报告的路径（可选）')
    args = parser.parse_args()
    
    analyze_eval_results(args.csv, args.output)


if __name__ == '__main__':
    main()

