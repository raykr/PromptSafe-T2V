"""
批量生成视频脚本
读取 prompt 文件中的每条 prompt，调用 cli_demo.py 的推理方法生成视频
"""

import os
import sys
from pathlib import Path
from cli_demo import generate_video
import torch


def main():
    # 配置参数
    prompt_file = "/home/beihang/jzl/projects/PromptSafe-T2V/datasets/T2VSafetyBench/14.txt"
    model_path = "/home/beihang/jzl/models/zai-org/CogVideoX-5b"
    output_dir = "out/temporal_baseline_cogvideox-5b"
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 读取 prompt 文件
    with open(prompt_file, 'r', encoding='utf-8') as f:
        prompts = [line.strip() for line in f if line.strip()]
    
    print(f"共找到 {len(prompts)} 条 prompt")
    print(f"输出目录: {output_dir}")
    print(f"模型路径: {model_path}")
    print("-" * 80)
    
    # 对每条 prompt 生成视频
    for idx, prompt in enumerate(prompts, start=1):
        print(f"\n[{idx}/{len(prompts)}] 正在生成视频...")
        print(f"Prompt: {prompt[:100]}..." if len(prompt) > 100 else f"Prompt: {prompt}")
        
        # 生成输出文件名（使用索引编号）
        output_filename = f"video_{idx:03d}.mp4"
        output_path = os.path.join(output_dir, output_filename)
        
        try:
            # 调用 generate_video 函数
            generate_video(
                prompt=prompt,
                model_path=model_path,
                output_path=output_path,
                generate_type="t2v",
                dtype=torch.bfloat16,
                num_frames=81,
                num_inference_steps=50,
                guidance_scale=6.0,
                seed=42,
                fps=16,
            )
            print(f"✓ 成功生成: {output_path}")
        except Exception as e:
            print(f"✗ 生成失败: {e}")
            print(f"  继续处理下一条 prompt...")
            continue
    
    print("\n" + "=" * 80)
    print(f"批量生成完成！共处理 {len(prompts)} 条 prompt")
    print(f"输出目录: {os.path.abspath(output_dir)}")


if __name__ == "__main__":
    main()

