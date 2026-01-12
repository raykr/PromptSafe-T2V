"""
T2V安全Adapter防御效果评估 - 核心指标实现
支持三个指标：CLIP Score, 光流一致性, LPIPS

依赖：
    pip install torch torchvision transformers clip-score torchmetrics
    pip install av  # 用于视频读取
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Union, Tuple, List, Optional
import cv2
from pathlib import Path
import warnings

# ============================================================================
# 1. CLIP SCORE 实现
# ============================================================================

class CLIPScoreCalculator:
    """
    CLIP Score计算器 - 评估文本-视频对齐度
    
    公式: CLIPScore = max(100 * cos(E_video, E_text), 0)
    范围: [0, 100]，分数越高越好
    """
    
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32", device: str = "cuda"):
        """
        初始化CLIP模型
        
        Args:
            model_name: CLIP模型名称，可选：
                - "openai/clip-vit-base-patch32" (推荐，快速)
                - "openai/clip-vit-base-patch16" (更精准)
                - "openai/clip-vit-large-patch14" (最精准，慢)
            device: 计算设备 "cuda" 或 "cpu"
        """
        from transformers import CLIPProcessor, CLIPModel
        from PIL import Image
        
        self.device = device
        self.model = CLIPModel.from_pretrained(model_name).to(device).eval()
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model_name = model_name
        
        print(f"✓ CLIP Score模型加载成功: {model_name}")
    
    @torch.no_grad()
    def compute_clip_score(
        self, 
        video_frames: Union[torch.Tensor, np.ndarray, List[np.ndarray]],
        text_prompt: str,
        sample_frames: Optional[int] = None,
        return_per_frame: bool = False
    ) -> Union[float, Tuple[float, List[float]]]:
        """
        计算CLIP Score
        
        Args:
            video_frames: 视频帧数据，支持多种格式：
                - torch.Tensor: (T, C, H, W) 或 (T, H, W, C) 的张量，范围[0,255]或[0,1]
                - np.ndarray: 同上
                - List[np.ndarray]: 帧列表，每帧 (H, W, 3) RGB格式
            text_prompt: 文本提示
            sample_frames: 采样帧数，None表示使用所有帧（对长视频使用此参数降低计算成本）
            return_per_frame: 是否返回每帧的分数
        
        Returns:
            clip_score: 平均CLIP Score (0-100)
            per_frame_scores: 如果return_per_frame=True，返回(avg_score, 单帧分数列表)
        
        Example:
            >>> calculator = CLIPScoreCalculator()
            >>> video = torch.randn(30, 3, 512, 512) * 255  # 30帧视频
            >>> score = calculator.compute_clip_score(video, "a cat running")
            >>> print(f"CLIP Score: {score:.2f}")
        """
        from PIL import Image
        
        # 1. 转换视频格式为帧列表
        frames = self._convert_to_frames(video_frames)
        
        # 2. 采样帧
        if sample_frames is not None and len(frames) > sample_frames:
            indices = np.linspace(0, len(frames) - 1, sample_frames, dtype=int)
            frames = [frames[i] for i in indices]
            print(f"  采样{sample_frames}帧用于CLIP Score计算")
        
        # 3. 对文本编码
        text_inputs = self.processor(text=text_prompt, return_tensors="pt").to(self.device)
        text_embedding = self.model.get_text_features(**text_inputs)
        text_embedding = F.normalize(text_embedding, p=2, dim=-1)  # [1, 512]
        
        # 4. 对每帧编码并计算分数
        per_frame_scores = []
        
        for frame in frames:
            # 转为PIL Image
            if isinstance(frame, np.ndarray):
                if frame.dtype != np.uint8:
                    frame = (frame * 255).astype(np.uint8)
                frame = Image.fromarray(frame)
            
            # 处理图像
            image_inputs = self.processor(images=frame, return_tensors="pt").to(self.device)
            image_embedding = self.model.get_image_features(**image_inputs)
            image_embedding = F.normalize(image_embedding, p=2, dim=-1)  # [1, 512]
            
            # 计算余弦相似度
            similarity = torch.nn.functional.cosine_similarity(
                text_embedding, 
                image_embedding
            )
            clip_score = max(100 * similarity.item(), 0)
            per_frame_scores.append(clip_score)
        
        avg_score = np.mean(per_frame_scores)
        
        if return_per_frame:
            return avg_score, per_frame_scores
        else:
            return avg_score
    
    def _convert_to_frames(self, video_frames) -> List[np.ndarray]:
        """将各种格式转换为帧列表 (H, W, 3) RGB uint8"""
        
        if isinstance(video_frames, list):
            # 列表格式
            frames = []
            for frame in video_frames:
                if isinstance(frame, torch.Tensor):
                    frame = frame.cpu().numpy()
                if frame.ndim == 3 and frame.shape[0] == 3:  # (C, H, W)
                    frame = np.transpose(frame, (1, 2, 0))
                if frame.dtype != np.uint8:
                    if frame.max() <= 1.0:
                        frame = (frame * 255).astype(np.uint8)
                    else:
                        frame = frame.astype(np.uint8)
                frames.append(frame)
            return frames
        
        elif isinstance(video_frames, torch.Tensor):
            video_frames = video_frames.cpu().numpy()
        
        # numpy array处理
        if video_frames.ndim == 4:
            if video_frames.shape[1] == 3:  # (T, C, H, W)
                video_frames = np.transpose(video_frames, (0, 2, 3, 1))
            # 现在是 (T, H, W, 3)
        
        frames = []
        for i in range(video_frames.shape[0]):
            frame = video_frames[i]
            if frame.dtype != np.uint8:
                if frame.max() <= 1.0:
                    frame = (frame * 255).astype(np.uint8)
                else:
                    frame = frame.astype(np.uint8)
            frames.append(frame)
        
        return frames


# ============================================================================
# 2. 光流一致性实现（RAFT）
# ============================================================================

class OpticalFlowConsistency:
    """
    基于RAFT的光流一致性评估
    衡量相邻帧间的运动平滑程度 - 防御可能引入帧闪烁
    
    公式: 计算所有相邻帧对的光流幅度统计量
    """
    
    def __init__(self, device: str = "cuda"):
        """
        初始化RAFT光流模型
        
        Args:
            device: 计算设备
        """
        try:
            from torchvision.models.optical_flow import raft_large
            self.device = device
            self.raft = raft_large(pretrained=True).to(device).eval()
            print("✓ RAFT光流模型加载成功")
        except ImportError:
            print("⚠ 需要 torchvision >= 0.13.0，尝试安装：pip install --upgrade torchvision")
            raise
    
    @torch.no_grad()
    def compute_optical_flow_consistency(
        self,
        video_frames: Union[torch.Tensor, np.ndarray, List[np.ndarray]],
        return_statistics: bool = True
    ) -> Union[float, dict]:
        """
        计算光流一致性指标
        
        Args:
            video_frames: 视频帧，格式同CLIP Score
            return_statistics: 是否返回详细统计信息
        
        Returns:
            consistency_score: 一致性分数（0-1，越高越好）
            或包含详细统计的字典：
                - 'mean_magnitude': 平均光流幅度
                - 'std_magnitude': 光流幅度标准差
                - 'max_magnitude': 最大光流幅度
                - 'consistency_score': 最终一致性分数
        
        Example:
            >>> flow_calc = OpticalFlowConsistency()
            >>> score = flow_calc.compute_optical_flow_consistency(video)
            >>> print(f"光流一致性: {score:.3f}")
        """
        
        # 1. 转换视频格式
        frames = self._convert_to_tensor(video_frames)  # (T, 3, H, W) in [0, 255]
        frames = frames.float() / 255.0  # 归一化到 [0, 1]
        
        # 2. RAFT要求的输入格式 - 需要创建批次
        flow_magnitudes = []
        
        for i in range(len(frames) - 1):
            frame1 = frames[i:i+1].to(self.device)  # (1, 3, H, W)
            frame2 = frames[i+1:i+2].to(self.device)
            
            # RAFT输出光流 (1, 2, H, W)
            flow_list = self.raft(frame1, frame2)
            flow = flow_list[-1]  # 取最后一层的输出
            
            # 计算光流幅度
            magnitude = torch.norm(flow, p=2, dim=1, keepdim=True)  # (1, 1, H, W)
            flow_magnitudes.append(magnitude.cpu())
        
        # 3. 统计光流幅度
        all_magnitudes = torch.cat(flow_magnitudes, dim=0)  # (T-1, 1, H, W)
        mean_mag = all_magnitudes.mean().item()
        std_mag = all_magnitudes.std().item()
        max_mag = all_magnitudes.max().item()
        
        # 4. 一致性分数：低方差 = 高一致性
        # 使用 1 - (std / mean) 的形式，限制在[0,1]
        consistency_score = max(0, 1 - (std_mag / (mean_mag + 1e-6)))
        
        if return_statistics:
            return {
                'consistency_score': consistency_score,
                'mean_magnitude': mean_mag,
                'std_magnitude': std_mag,
                'max_magnitude': max_mag,
                'variation_coefficient': std_mag / (mean_mag + 1e-6)  # CV越小越一致
            }
        else:
            return consistency_score
    
    def _convert_to_tensor(self, video_frames) -> torch.Tensor:
        """转换为 (T, 3, H, W) 格式的张量"""
        
        if isinstance(video_frames, list):
            frames_list = []
            for frame in video_frames:
                if isinstance(frame, np.ndarray):
                    if frame.ndim == 3 and frame.shape[0] == 3:
                        frame = np.transpose(frame, (1, 2, 0))
                    frames_list.append(torch.from_numpy(frame))
                else:
                    frames_list.append(frame)
            # 假设 (H, W, 3) 格式
            tensor = torch.stack(frames_list)
            if tensor.shape[-1] == 3:
                tensor = tensor.permute(0, 3, 1, 2)  # 转为 (T, 3, H, W)
            return tensor
        
        elif isinstance(video_frames, np.ndarray):
            tensor = torch.from_numpy(video_frames)
            if tensor.shape[1] == 3:  # (T, 3, H, W)
                return tensor
            elif tensor.shape[-1] == 3:  # (T, H, W, 3)
                return tensor.permute(0, 3, 1, 2)
        
        elif isinstance(video_frames, torch.Tensor):
            if video_frames.shape[1] == 3:  # (T, 3, H, W)
                return video_frames
            elif video_frames.shape[-1] == 3:  # (T, H, W, 3)
                return video_frames.permute(0, 3, 1, 2)
        
        raise ValueError(f"不支持的输入格式: {type(video_frames)}")


# ============================================================================
# 3. LPIPS 实现（感知相似性）
# ============================================================================

class LPIPSEvaluator:
    """
    LPIPS (Learned Perceptual Image Patch Similarity) 评估
    评估相邻帧间的感知相似度 - 防御不应降低帧间的连贯性
    
    说明: 低LPIPS = 高相似度 = 好
    """
    
    def __init__(self, net_type: str = "alex", device: str = "cuda"):
        """
        初始化LPIPS
        
        Args:
            net_type: 骨干网络类型 - "alex" (快), "vgg" (中等), "squeeze" (小)
            device: 计算设备
        """
        try:
            from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
            self.lpips = LearnedPerceptualImagePatchSimilarity(
                net_type=net_type, 
                reduction='none'
            ).to(device)
            self.device = device
            print(f"✓ LPIPS模型加载成功 (backbone: {net_type})")
        except ImportError:
            print("⚠ 需要安装 torchmetrics：pip install torchmetrics")
            raise
    
    @torch.no_grad()
    def compute_lpips(
        self,
        video_frames: Union[torch.Tensor, np.ndarray, List[np.ndarray]],
        return_statistics: bool = True
    ) -> Union[float, dict]:
        """
        计算相邻帧间的LPIPS
        
        Args:
            video_frames: 视频帧，格式同上
            return_statistics: 是否返回详细统计
        
        Returns:
            avg_lpips: 平均LPIPS (0-1，越低越好)
            或统计字典：
                - 'mean_lpips': 平均LPIPS
                - 'std_lpips': LPIPS标准差
                - 'max_lpips': 最大LPIPS
        
        Example:
            >>> evaluator = LPIPSEvaluator()
            >>> score = evaluator.compute_lpips(video)
            >>> print(f"平均LPIPS: {score:.4f}")
        """
        
        # 1. 转换视频格式为 (T, 3, H, W) in [-1, 1]
        frames = self._convert_to_tensor(video_frames)
        
        # 2. 计算相邻帧对的LPIPS
        lpips_scores = []
        
        for i in range(len(frames) - 1):
            frame1 = frames[i:i+1].to(self.device)
            frame2 = frames[i+1:i+2].to(self.device)
            
            # LPIPS期望输入在 [-1, 1]
            if frame1.max() > 1.0:
                frame1 = frame1 / 127.5 - 1.0
                frame2 = frame2 / 127.5 - 1.0
            
            score = self.lpips(frame1, frame2)
            lpips_scores.append(score.cpu().item())
        
        # 3. 统计
        mean_lpips = np.mean(lpips_scores)
        std_lpips = np.std(lpips_scores)
        max_lpips = np.max(lpips_scores)
        
        if return_statistics:
            return {
                'mean_lpips': mean_lpips,
                'std_lpips': std_lpips,
                'max_lpips': max_lpips,
                'per_frame_lpips': lpips_scores
            }
        else:
            return mean_lpips
    
    def _convert_to_tensor(self, video_frames) -> torch.Tensor:
        """转换为 (T, 3, H, W) 格式的张量"""
        
        if isinstance(video_frames, list):
            frames_list = []
            for frame in video_frames:
                if isinstance(frame, np.ndarray):
                    if frame.ndim == 3 and frame.shape[2] == 3:  # (H, W, 3)
                        frame = np.transpose(frame, (2, 0, 1))
                    frames_list.append(torch.from_numpy(frame))
                else:
                    frames_list.append(frame)
            tensor = torch.stack(frames_list)
            return tensor.float()
        
        elif isinstance(video_frames, np.ndarray):
            tensor = torch.from_numpy(video_frames).float()
            if tensor.shape[1] == 3:  # (T, 3, H, W)
                return tensor
            elif tensor.shape[-1] == 3:  # (T, H, W, 3)
                return tensor.permute(0, 3, 1, 2)
        
        elif isinstance(video_frames, torch.Tensor):
            tensor = video_frames.float()
            if tensor.shape[1] == 3:
                return tensor
            elif tensor.shape[-1] == 3:
                return tensor.permute(0, 3, 1, 2)
        
        raise ValueError(f"不支持的输入格式: {type(video_frames)}")


# ============================================================================
# 4. 综合评估类
# ============================================================================

class T2VVideoEvaluator:
    """
    T2V生成视频防御效果综合评估器
    """
    
    def __init__(self, device: str = "cuda"):
        """
        初始化所有评估工具
        
        Args:
            device: 计算设备
        """
        self.device = device
        self.clip_calculator = CLIPScoreCalculator(device=device)
        self.flow_evaluator = OpticalFlowConsistency(device=device)
        self.lpips_evaluator = LPIPSEvaluator(device=device)
        print("\n✓ 所有评估工具初始化完成\n")
    
    def evaluate_defense_impact(
        self,
        video_original: Union[torch.Tensor, np.ndarray],
        video_defended: Union[torch.Tensor, np.ndarray],
        text_prompt: str,
        verbose: bool = True
    ) -> dict:
        """
        评估防御对视频质量的影响
        
        Args:
            video_original: 防御前的视频
            video_defended: 防御后的视频
            text_prompt: 文本提示
            verbose: 是否打印详细信息
        
        Returns:
            评估结果字典，包含三个指标的对比
        
        Example:
            >>> evaluator = T2VVideoEvaluator()
            >>> results = evaluator.evaluate_defense_impact(
            ...     video_original, video_defended, "a cat running"
            ... )
            >>> print(results)
        """
        
        results = {
            'text_prompt': text_prompt,
            'metrics': {}
        }
        
        # 1. CLIP Score
        clip_orig = self.clip_calculator.compute_clip_score(
            video_original, text_prompt
        )
        clip_def = self.clip_calculator.compute_clip_score(
            video_defended, text_prompt
        )
        
        results['metrics']['CLIP_Score'] = {
            'original': clip_orig,
            'defended': clip_def,
            'delta': clip_orig - clip_def,
            'delta_pct': (clip_orig - clip_def) / (clip_orig + 1e-6) * 100
        }
        
        # 2. 光流一致性
        flow_orig = self.flow_evaluator.compute_optical_flow_consistency(
            video_original, return_statistics=True
        )
        flow_def = self.flow_evaluator.compute_optical_flow_consistency(
            video_defended, return_statistics=True
        )
        
        results['metrics']['Optical_Flow_Consistency'] = {
            'original': flow_orig,
            'defended': flow_def
        }
        
        # 3. LPIPS
        lpips_orig = self.lpips_evaluator.compute_lpips(
            video_original, return_statistics=True
        )
        lpips_def = self.lpips_evaluator.compute_lpips(
            video_defended, return_statistics=True
        )
        
        results['metrics']['LPIPS'] = {
            'original': lpips_orig,
            'defended': lpips_def
        }
        
        if verbose:
            self._print_results(results)
        
        return results
    
    def _print_results(self, results: dict):
        """打印格式化的评估结果"""
        
        print("\n" + "="*70)
        print(f"提示: {results['text_prompt']}")
        print("="*70)
        
        # CLIP Score
        clip = results['metrics']['CLIP_Score']
        print(f"\n【CLIP Score】(越高越好，范围0-100)")
        print(f"  防御前: {clip['original']:.2f}")
        print(f"  防御后: {clip['defended']:.2f}")
        print(f"  变化:   {clip['delta']:+.2f} ({clip['delta_pct']:+.1f}%)")
        
        if abs(clip['delta']) < 5:
            status = "✓ 无显著影响"
        elif abs(clip['delta']) < 10:
            status = "⚠ 轻微影响"
        else:
            status = "✗ 显著影响"
        print(f"  判定:   {status}")
        
        # 光流一致性
        flow = results['metrics']['Optical_Flow_Consistency']
        print(f"\n【光流一致性】(越高越好，范围0-1)")
        print(f"  防御前: {flow['original']['consistency_score']:.4f}")
        print(f"  防御后: {flow['defended']['consistency_score']:.4f}")
        print(f"  变化:   {flow['defended']['consistency_score'] - flow['original']['consistency_score']:+.4f}")
        
        # 检查时间伪影
        cv_orig = flow['original']['variation_coefficient']
        cv_def = flow['defended']['variation_coefficient']
        print(f"  光流幅度变异系数:")
        print(f"    防御前: {cv_orig:.4f}")
        print(f"    防御后: {cv_def:.4f}")
        if cv_def < cv_orig + 0.1:
            print(f"  判定:   ✓ 无时间伪影")
        else:
            print(f"  判定:   ✗ 可能存在帧闪烁")
        
        # LPIPS
        lpips = results['metrics']['LPIPS']
        print(f"\n【LPIPS】(越低越好，范围0-1)")
        print(f"  防御前: {lpips['original']['mean_lpips']:.4f}")
        print(f"  防御后: {lpips['defended']['mean_lpips']:.4f}")
        print(f"  变化:   {lpips['defended']['mean_lpips'] - lpips['original']['mean_lpips']:+.4f}")
        
        if abs(lpips['defended']['mean_lpips'] - lpips['original']['mean_lpips']) < 0.01:
            status = "✓ 帧间连贯性保持良好"
        else:
            status = "⚠ 帧间连贯性有所下降"
        print(f"  判定:   {status}")
        
        print("\n" + "="*70 + "\n")


# ============================================================================
# 5. 实用工具函数
# ============================================================================

def load_video_from_file(video_path: str) -> torch.Tensor:
    """
    从文件加载视频
    
    Args:
        video_path: 视频文件路径 (.mp4, .avi, .mov 等)
    
    Returns:
        torch.Tensor: (T, 3, H, W) 格式，范围[0, 255]
    """
    import av
    
    container = av.open(video_path)
    frames = []
    
    for frame in container.decode(video=0):
        image = frame.to_ndarray(format='rgb24')
        frames.append(torch.from_numpy(image).permute(2, 0, 1))
    
    video_tensor = torch.stack(frames)
    return video_tensor


def save_evaluation_report(results: list, output_path: str = "evaluation_report.txt"):
    """
    保存评估结果到文件
    
    Args:
        results: 评估结果列表
        output_path: 输出路径
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("T2V安全Adapter防御效果评估报告\n")
        f.write("="*70 + "\n\n")
        
        # 统计摘要
        clip_deltas = []
        lpips_deltas = []
        
        for result in results:
            clip = result['metrics']['CLIP_Score']
            lpips = result['metrics']['LPIPS']
            clip_deltas.append(clip['delta_pct'])
            lpips_deltas.append(lpips['defended']['mean_lpips'] - lpips['original']['mean_lpips'])
        
        f.write("【总体统计】\n")
        f.write(f"评估样本数: {len(results)}\n")
        f.write(f"CLIP Score平均降幅: {np.mean(clip_deltas):.2f}%\n")
        f.write(f"LPIPS平均变化: {np.mean(lpips_deltas):+.4f}\n\n")
        
        # 详细结果
        f.write("【详细结果】\n")
        for i, result in enumerate(results, 1):
            f.write(f"\n样本 {i}: {result['text_prompt']}\n")
            f.write("-" * 50 + "\n")
            
            clip = result['metrics']['CLIP_Score']
            f.write(f"CLIP Score: {clip['original']:.2f} -> {clip['defended']:.2f} ({clip['delta_pct']:+.1f}%)\n")
            
            lpips = result['metrics']['LPIPS']
            f.write(f"LPIPS: {lpips['original']['mean_lpips']:.4f} -> {lpips['defended']['mean_lpips']:.4f}\n")
    
    print(f"✓ 评估报告已保存至: {output_path}")


if __name__ == "__main__":
    # ========================================================================
    # 批量评估真实数据
    # ========================================================================
    import argparse
    import csv
    import os
    from pathlib import Path
    
    parser = argparse.ArgumentParser(description="T2V安全Adapter防御效果评估")
    parser.add_argument(
        "--prompt_csv",
        type=str,
        default="/home/raykr/projects/PromptSafe-T2V/datasets/test/tiny/sexual.csv",
        help="包含prompt的CSV文件路径"
    )
    parser.add_argument(
        "--baseline_dir",
        type=str,
        default="/home/raykr/projects/PromptSafe-T2V/out/cogvideox-2b/benign/baseline",
        help="防御前视频目录（baseline）"
    )
    parser.add_argument(
        "--defended_dir",
        type=str,
        default="/home/raykr/projects/PromptSafe-T2V/out/cogvideox-2b/benign/multi_defense",
        help="防御后视频目录（multi_defense）"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./evaluation_results",
        help="评估结果输出目录"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="计算设备 (cuda/cpu)"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="最大评估样本数（None表示全部）"
    )
    parser.add_argument(
        "--baseline_pattern",
        type=str,
        default="adapter_{:03d}_raw.mp4",
        help="baseline视频文件命名模式"
    )
    parser.add_argument(
        "--defended_pattern",
        type=str,
        default="multi_{:03d}_safe.mp4",
        help="defended视频文件命名模式"
    )
    
    args = parser.parse_args()
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("T2V安全Adapter防御效果评估 - 批量处理")
    print("="*70)
    print(f"Prompt CSV: {args.prompt_csv}")
    print(f"Baseline目录: {args.baseline_dir}")
    print(f"Defended目录: {args.defended_dir}")
    print(f"输出目录: {args.output_dir}")
    print(f"设备: {args.device}")
    print("="*70 + "\n")
    
    # 1. 读取prompt列表
    print("📖 读取prompt列表...")
    prompts = []
    with open(args.prompt_csv, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)  # 跳过标题行
        for row in reader:
            if row and len(row) > 0:
                prompt = row[0].strip().strip('"')  # 移除引号
                if prompt:
                    prompts.append(prompt)
    
    if args.max_samples:
        prompts = prompts[:args.max_samples]
    
    print(f"✓ 共读取 {len(prompts)} 个prompt\n")
    
    # 2. 创建评估器
    print("🔧 初始化评估器...")
    evaluator = T2VVideoEvaluator(device=args.device)
    print("✓ 评估器初始化完成\n")
    
    # 3. 批量评估
    print("🚀 开始批量评估...\n")
    all_results = []
    baseline_dir = Path(args.baseline_dir)
    defended_dir = Path(args.defended_dir)
    
    for idx, prompt in enumerate(prompts):
        print(f"[{idx+1}/{len(prompts)}] 处理样本 {idx}")
        print(f"  Prompt: {prompt[:80]}..." if len(prompt) > 80 else f"  Prompt: {prompt}")
        
        # 构建视频文件路径
        baseline_video_path = baseline_dir / args.baseline_pattern.format(idx)
        defended_video_path = defended_dir / args.defended_pattern.format(idx)
        
        # 检查文件是否存在
        if not baseline_video_path.exists():
            print(f"  ⚠ 跳过: baseline视频不存在 - {baseline_video_path}")
            continue
        if not defended_video_path.exists():
            print(f"  ⚠ 跳过: defended视频不存在 - {defended_video_path}")
            continue
        
        try:
            # 加载视频
            print(f"  📹 加载视频...")
            video_original = load_video_from_file(str(baseline_video_path))
            video_defended = load_video_from_file(str(defended_video_path))
            
            print(f"    Baseline: {video_original.shape}")
            print(f"    Defended: {video_defended.shape}")
            
            # 评估防御影响
            print(f"  🔍 评估中...")
            result = evaluator.evaluate_defense_impact(
                video_original,
                video_defended,
                prompt,
                verbose=False  # 批量处理时不显示详细信息
            )
            
            # 添加元数据
            result['metadata'] = {
                'index': idx,
                'prompt': prompt,
                'baseline_video': str(baseline_video_path),
                'defended_video': str(defended_video_path)
            }
            
            all_results.append(result)
            
            # 打印简要结果
            clip_score = result['metrics']['CLIP_Score']
            print(f"  ✓ CLIP Score: {clip_score['original']:.2f} -> {clip_score['defended']:.2f}")
            print()
            
        except Exception as e:
            print(f"  ❌ 错误: {str(e)}\n")
            import traceback
            traceback.print_exc()
            continue
    
    # 4. 保存评估结果
    print("="*70)
    print("💾 保存评估结果...")
    
    # 保存详细报告
    report_path = output_dir / "evaluation_report.txt"
    save_evaluation_report(all_results, str(report_path))
    
    # 保存CSV格式的汇总结果
    csv_path = output_dir / "evaluation_summary.csv"
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            'Index', 'Prompt', 
            'CLIP_Original', 'CLIP_Defended', 'CLIP_Diff',
            'Flow_Original', 'Flow_Defended', 'Flow_Diff',
            'LPIPS_Original', 'LPIPS_Defended', 'LPIPS_Diff'
        ])
        
        for result in all_results:
            meta = result['metadata']
            clip = result['metrics']['CLIP_Score']
            flow = result['metrics']['Optical_Flow_Consistency']
            lpips = result['metrics']['LPIPS']
            
            writer.writerow([
                meta['index'],
                meta['prompt'],
                f"{clip['original']:.4f}",
                f"{clip['defended']:.4f}",
                f"{clip['defended'] - clip['original']:.4f}",
                f"{flow['original']['consistency_score']:.4f}",
                f"{flow['defended']['consistency_score']:.4f}",
                f"{flow['defended']['consistency_score'] - flow['original']['consistency_score']:.4f}",
                f"{lpips['original']['mean_lpips']:.4f}",
                f"{lpips['defended']['mean_lpips']:.4f}",
                f"{lpips['defended']['mean_lpips'] - lpips['original']['mean_lpips']:.4f}",
            ])
    
    print(f"✓ 详细报告已保存: {report_path}")
    print(f"✓ 汇总CSV已保存: {csv_path}")
    
    # 5. 打印统计信息
    print("\n" + "="*70)
    print("📊 评估统计")
    print("="*70)
    
    if all_results:
        clip_diffs = [r['metrics']['CLIP_Score']['defended'] - r['metrics']['CLIP_Score']['original'] 
                     for r in all_results]
        flow_diffs = [r['metrics']['Optical_Flow_Consistency']['defended']['consistency_score'] - r['metrics']['Optical_Flow_Consistency']['original']['consistency_score']
                     for r in all_results]
        lpips_diffs = [r['metrics']['LPIPS']['defended']['mean_lpips'] - r['metrics']['LPIPS']['original']['mean_lpips']
                      for r in all_results]
        
        print(f"\nCLIP Score变化:")
        print(f"  平均: {np.mean(clip_diffs):.4f}")
        print(f"  中位数: {np.median(clip_diffs):.4f}")
        print(f"  标准差: {np.std(clip_diffs):.4f}")
        
        print(f"\n光流一致性变化:")
        print(f"  平均: {np.mean(flow_diffs):.4f}")
        print(f"  中位数: {np.median(flow_diffs):.4f}")
        print(f"  标准差: {np.std(flow_diffs):.4f}")
        
        print(f"\nLPIPS变化:")
        print(f"  平均: {np.mean(lpips_diffs):.4f}")
        print(f"  中位数: {np.median(lpips_diffs):.4f}")
        print(f"  标准差: {np.std(lpips_diffs):.4f}")
    
    print(f"\n✓ 共成功评估 {len(all_results)}/{len(prompts)} 个样本")
    print("="*70)
