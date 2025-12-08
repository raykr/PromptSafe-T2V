#!/usr/bin/env python
# coding=utf-8
"""
T2V推理脚本，支持加载soft token embedding参数进行推理
"""

import argparse
import json
import os
import torch
import safetensors
from pathlib import Path
from typing import Optional, Union, List
import numpy as np
from PIL import Image
import cv2
import subprocess
import tempfile

from transformers import T5EncoderModel, AutoTokenizer
from diffusers import (
    AutoencoderKLCogVideoX,
    CogVideoXDPMScheduler,
    CogVideoXPipeline,
    CogVideoXTransformer3DModel,
)
from diffusers.models.embeddings import get_3d_rotary_pos_embed
from diffusers.utils import export_to_video


class T2VInferencePipeline:
    """T2V推理pipeline，支持soft token embedding"""
    
    def __init__(
        self,
        pretrained_model_path: str,
        learned_embeds_path: str,
        placeholder_token: str,
        num_vectors: int,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16
    ):
        """
        初始化T2V推理pipeline
        
        Args:
            pretrained_model_path: 预训练模型路径
            learned_embeds_path: 学习到的embedding参数路径
            device: 设备
            dtype: 数据类型
        """
        self.device = device
        self.dtype = dtype
        
        
        # 加载模型组件
        self._load_models(pretrained_model_path)
        
        # 加载soft token embedding
        self._load_soft_token_embeddings(learned_embeds_path, placeholder_token, num_vectors)
        
        # 创建pipeline
        self._create_pipeline()
    
    def _load_models(self, pretrained_model_path: str):
        """加载预训练模型组件"""
        print("正在加载预训练模型组件...")
        
        # 加载tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_path, 
            subfolder="tokenizer"
        )
        
        # 加载text encoder
        self.text_encoder = T5EncoderModel.from_pretrained(
            pretrained_model_path, 
            subfolder="text_encoder"
        )
        
        # 加载transformer
        self.transformer = CogVideoXTransformer3DModel.from_pretrained(
            pretrained_model_path, 
            subfolder="transformer"
        )
        
        # 加载scheduler
        self.scheduler = CogVideoXDPMScheduler.from_pretrained(
            pretrained_model_path, 
            subfolder="scheduler"
        )
        
        # 加载VAE
        self.vae = AutoencoderKLCogVideoX.from_pretrained(
            pretrained_model_path, 
            subfolder="vae"
        )
        
        # 移动到指定设备和数据类型
        self.text_encoder = self.text_encoder.to(self.device, dtype=self.dtype)
        self.transformer = self.transformer.to(self.device, dtype=self.dtype)
        self.vae = self.vae.to(self.device, dtype=self.dtype)
        
        print("模型组件加载完成")
    
    def _load_soft_token_embeddings(self, learned_embeds_path: str, placeholder_token: str, num_vectors: int):
        """加载soft token embedding参数"""
        print(f"正在加载soft token embedding参数: {learned_embeds_path}")
        
        # 加载embedding参数
        if learned_embeds_path.endswith('.safetensors'):
            learned_embeds_dict = safetensors.torch.load_file(learned_embeds_path)
        else:
            learned_embeds_dict = torch.load(learned_embeds_path, map_location='cpu')
        
        # 获取placeholder token
        self.placeholder_token = placeholder_token
        self.num_vectors = num_vectors
        
        # 添加placeholder tokens到tokenizer
        placeholder_tokens = [self.placeholder_token]
        for i in range(1, self.num_vectors):
            placeholder_tokens.append(f"{self.placeholder_token}_{i}")
        
        # 检查tokenizer是否已经包含这些tokens
        existing_tokens = set(self.tokenizer.get_vocab().keys())
        new_tokens = [token for token in placeholder_tokens if token not in existing_tokens]
        
        if new_tokens:
            num_added_tokens = self.tokenizer.add_tokens(new_tokens)
            print(f"添加了 {num_added_tokens} 个新的tokens到tokenizer")
            
            # 调整text encoder的embedding层大小
            self.text_encoder.resize_token_embeddings(len(self.tokenizer))
        
        # 获取placeholder token的ID
        self.placeholder_token_ids = self.tokenizer.convert_tokens_to_ids(placeholder_tokens)
        print(f"Placeholder token IDs: {self.placeholder_token_ids}")
        
        # 加载学习到的embedding参数
        learned_embeds = learned_embeds_dict[self.placeholder_token]
        print(f"学习到的embedding形状: {learned_embeds.shape}")
        
        # 更新text encoder的embedding层
        with torch.no_grad():
            embedding_layer = self.text_encoder.get_input_embeddings()
            for i, token_id in enumerate(self.placeholder_token_ids):
                if i < learned_embeds.shape[0]:
                    embedding_layer.weight[token_id] = learned_embeds[i].to(self.device, dtype=self.dtype)
        
        print("Soft token embedding参数加载完成")
    
    def _create_pipeline(self):
        """创建推理pipeline"""
        print("正在创建推理pipeline...")
        
        self.pipeline = CogVideoXPipeline(
            tokenizer=self.tokenizer,
            text_encoder=self.text_encoder,
            vae=self.vae,
            transformer=self.transformer,
            scheduler=self.scheduler,
        )
        
        self.pipeline = self.pipeline.to(self.device)
        self.pipeline.set_progress_bar_config(disable=False)
        
        print("推理pipeline创建完成")
    
    def prepare_rotary_positional_embeddings(
        self,
        height: int,
        width: int,
        num_frames: int,
        transformer_config: dict,
        vae_scale_factor_spatial: int,
        device: torch.device,
    ):
        """准备旋转位置编码"""
        grid_height = height // (vae_scale_factor_spatial * transformer_config.patch_size)
        grid_width = width // (vae_scale_factor_spatial * transformer_config.patch_size)

        if transformer_config.patch_size_t is None:
            base_num_frames = num_frames
        else:
            base_num_frames = (
                num_frames + transformer_config.patch_size_t - 1
            ) // transformer_config.patch_size_t
            
        freqs_cos, freqs_sin = get_3d_rotary_pos_embed(
            embed_dim=transformer_config.attention_head_dim,
            crops_coords=None,
            grid_size=(grid_height, grid_width),
            temporal_size=base_num_frames,
            grid_type="slice",
            max_size=(grid_height, grid_width),
            device=device,
        )

        return freqs_cos, freqs_sin
    
    def add_soft_token_to_prompt(
        self, 
        prompt: str, 
        position: str = "start"
    ) -> str:
        """
        在prompt中添加soft token
        
        Args:
            prompt: 原始prompt
            position: soft token位置 ("start", "end", "middle")
        """
        if position == "start":
            return f"{self.placeholder_token} {prompt}"
        elif position == "end":
            return f"{prompt} {self.placeholder_token}"
        elif position == "middle":
            words = prompt.split()
            if len(words) > 1:
                middle_idx = len(words) // 2
                words.insert(middle_idx, self.placeholder_token)
                return " ".join(words)
            else:
                return f"{self.placeholder_token} {prompt}"
        else:
            raise ValueError(f"不支持的位置: {position}")
    
    def generate_video(
        self,
        prompt: str,
        add_soft_token: bool = True,
        soft_token_position: str = "start",
        num_frames: int = 8,
        height: int = 256,
        width: int = 256,
        num_inference_steps: int = 25,
        guidance_scale: float = 7.5,
        generator: Optional[torch.Generator] = None
    ) -> np.ndarray:
        """
        生成视频
        
        Args:
            prompt: 输入prompt
            add_soft_token: 是否添加soft token
            soft_token_position: soft token位置
            num_frames: 视频帧数
            height: 视频高度
            width: 视频宽度
            num_inference_steps: 推理步数
            guidance_scale: 引导尺度
            generator: 随机数生成器
            save_path: 保存路径
            
        Returns:
            生成的视频数组 [T, H, W, C]
        """
        print(f"开始生成视频...")
        print(f"原始prompt: {prompt}")
        
        # 添加soft token
        if add_soft_token:
            prompt = self.add_soft_token_to_prompt(prompt, soft_token_position)
            print(f"添加soft token后的prompt: {prompt}")
        
        # 设置随机数生成器
        if generator is None:
            generator = torch.Generator(device=self.device)
            generator.manual_seed(42)
        
        # 生成视频
        with torch.autocast(self.device):
            video = self.pipeline(
                prompt=prompt,
                num_frames=num_frames,
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator,
            ).frames[0]
    
        return video

    
    def save_video_with_ffmpeg(self, video_array: np.ndarray, save_path: str):
        """使用FFmpeg保存视频，确保更好的兼容性"""
        print(f"使用FFmpeg保存视频到: {save_path}")
        
        # 确保保存目录存在
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        
        # 获取视频参数
        num_frames, height, width, channels = video_array.shape
        print(f"视频参数: {num_frames}帧, {height}x{width}, {channels}通道")
        
        # 创建临时目录保存帧图片
        with tempfile.TemporaryDirectory() as temp_dir:
            # 保存每一帧为图片
            frame_paths = []
            for i, frame in enumerate(video_array):
                # 确保帧数据在正确范围内 [0, 255]
                if frame.dtype != np.uint8:
                    frame = np.clip(frame * 255, 0, 255).astype(np.uint8)
                
                frame_path = os.path.join(temp_dir, f"frame_{i:04d}.png")
                cv2.imwrite(frame_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                frame_paths.append(frame_path)
            
            # 使用FFmpeg创建视频
            input_pattern = os.path.join(temp_dir, "frame_%04d.png")
            cmd = [
                'ffmpeg', '-y',  # -y 覆盖输出文件
                '-framerate', '8',  # 8fps
                '-i', input_pattern,  # 输入图片模式
                '-c:v', 'libx264',  # 使用H.264编码
                '-pix_fmt', 'yuv420p',  # 像素格式，确保兼容性
                '-crf', '23',  # 质量参数
                '-preset', 'fast',  # 编码速度预设
                save_path
            ]
            
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, check=True)
                print(f"FFmpeg成功创建视频: {save_path}")
                
                # 验证视频文件
                if os.path.exists(save_path):
                    file_size = os.path.getsize(save_path)
                    print(f"视频文件大小: {file_size / 1024:.1f} KB")
                    
                    # 使用ffprobe验证视频
                    probe_cmd = ['ffprobe', '-v', 'quiet', '-print_format', 'json', 
                               '-show_format', '-show_streams', save_path]
                    probe_result = subprocess.run(probe_cmd, capture_output=True, text=True)
                    if probe_result.returncode == 0:
                        print("视频文件验证成功")
                    else:
                        print("视频文件验证失败")
                else:
                    print("错误: 视频文件未创建")
                    
            except subprocess.CalledProcessError as e:
                print(f"FFmpeg执行失败: {e}")
                print(f"错误输出: {e.stderr}")
                # 回退到OpenCV方法
                print("回退到OpenCV方法...")
                self.save_video(video_array, save_path)
            except FileNotFoundError:
                print("FFmpeg未找到，回退到OpenCV方法...")
                self.save_video(video_array, save_path)


def main():
    parser = argparse.ArgumentParser(description="T2V推理脚本")
    parser.add_argument(
        "--pretrained_model_path",
        type=str,
        required=True,
        help="预训练模型路径"
    )
    parser.add_argument(
        "--learned_embeds_path",
        type=str,
        help="学习到的embedding参数路径"
    )
    parser.add_argument(
        "--placeholder_token",
        type=str,
        default="<safe>",
        help="placeholder token"
    )
    parser.add_argument(
        "--num_vectors",
        type=int,
        default=1,
        help="num vectors"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="输入prompt"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="output_video.mp4",
        help="输出视频路径"
    )
    parser.add_argument(
        "--add_soft_token",
        action="store_true",
        help="是否添加soft token"
    )
    parser.add_argument(
        "--soft_token_position",
        type=str,
        default="start",
        choices=["start", "end", "middle"],
        help="soft token位置"
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=8,
        help="视频帧数"
    )
    parser.add_argument(
        "--height",
        type=int,
        default=256,
        help="视频高度"
    )
    parser.add_argument(
        "--width",
        type=int,
        default=256,
        help="视频宽度"
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=25,
        help="推理步数"
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=7.5,
        help="引导尺度"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="设备"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子"
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=18,
        help="帧率"
    )
    
    args = parser.parse_args()
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # 创建推理pipeline
    pipeline = T2VInferencePipeline(
        pretrained_model_path=args.pretrained_model_path,
        learned_embeds_path=args.learned_embeds_path,
        placeholder_token=args.placeholder_token,
        num_vectors=args.num_vectors,
        device=args.device
    )
    
    # 生成视频
    video_generate = pipeline.generate_video(
        prompt=args.prompt,
        add_soft_token=args.add_soft_token,
        soft_token_position=args.soft_token_position,
        num_frames=args.num_frames,
        height=args.height,
        width=args.width,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale
    )

    export_to_video(video_generate, args.output_path, fps=args.fps)
    
    print("推理完成！")


if __name__ == "__main__":
    main()

