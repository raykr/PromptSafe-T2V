#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Open-Sora Safe Adapter (per-harm-category) training + inference.

Core idea:
- Freeze Open-Sora base model.
- Attach LoRA adapters to the T5 text encoder (or its attention/projection modules).
- Train one adapter per harmful category by minimizing:
    L = lambda_tox * toxic_score_y(embeds) + lambda_preserve * MSE(embeds, embeds_base)

Inference:
- Use toxic classifier to predict harmful category.
- Activate corresponding adapter and set LoRA scale (defense strength).
- Run Open-Sora sampling with the adapted text embeddings.

You MUST implement:
- load_opensora_components(...)
- opensora_encode_prompt(...)
- opensora_sample(...)
according to your local Open-Sora checkout.

This file is intentionally self-contained for your current pipeline style.
"""

import os
import sys
import json
import math
import argparse
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import imageio

from transformers import T5Tokenizer, T5EncoderModel

# PEFT LoRA
from peft import LoraConfig, get_peft_model, PeftModel


HARM_CATEGORIES = ["sexual", "violent", "political", "disturbing"]


# ---------------------------
# Data
# ---------------------------
class PromptLabelDataset(Dataset):
    """
    JSONL format (one sample per line):
      {"prompt": "...", "label": "sexual"}  # label in HARM_CATEGORIES
    """
    def __init__(self, jsonl_path: str):
        self.items = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                assert "prompt" in obj and "label" in obj, "Each line must have prompt/label"
                self.items.append((obj["prompt"], obj["label"]))
        assert len(self.items) > 0, f"Empty dataset: {jsonl_path}"

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int):
        return self.items[idx]


def collate_batch(batch):
    prompts, labels = zip(*batch)
    return list(prompts), list(labels)


# ---------------------------
# Toxic classifier interface
# ---------------------------
class ToxicClassifier(nn.Module):
    """
    Plug your own classifier here.

    Required API:
      forward_text_embeds(embeds, attention_mask) -> logits [B, C]
      predict_label(prompt_str_list) -> (label_str_list, prob_list)

    In your current system you already have a T5Encoder-based toxic classifier.
    You can replace the implementation below with your existing checkpoint loader.
    """
    def __init__(self, num_classes: int = 4):
        super().__init__()
        self.num_classes = num_classes

        # Minimal placeholder: a linear head over mean pooled embeddings.
        # Replace with your real classifier weights for meaningful training.
        self.head = nn.Sequential(
            nn.LayerNorm(4096),  # typical T5-XXL hidden size; change to match your T5
            nn.Linear(4096, num_classes)
        )

    @torch.no_grad()
    def predict_label(self, prompt_list: List[str]) -> Tuple[List[str], List[float]]:
        # Placeholder heuristic. Replace with your actual prompt classifier.
        # Here we just return "disturbing" with prob 0.5 for all.
        return ["disturbing"] * len(prompt_list), [0.5] * len(prompt_list)

    def forward_text_embeds(self, embeds: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        # embeds: [B, L, D]
        # attention_mask: [B, L]
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (embeds * mask).sum(dim=1) / (mask.sum(dim=1).clamp_min(1.0))
        return self.head(pooled)


# ---------------------------
# Adapter bank
# ---------------------------
@dataclass
class AdapterSpec:
    name: str
    out_dir: str
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    # Which modules in T5 to apply LoRA to: you should tune to your T5 variant.
    # Common targets: q, k, v, o projections, or "q", "k", "v", "o" substrings.
    target_modules: Tuple[str, ...] = ("q", "k", "v", "o")


class T5AdapterBank:
    """
    Maintain per-category LoRA adapters on the SAME base T5 encoder.
    """
    def __init__(self, base_t5: T5EncoderModel, adapter_root: str, device: torch.device):
        self.base_t5 = base_t5
        self.adapter_root = adapter_root
        self.device = device

        # PeftModel wrapper will be created lazily.
        self.peft_model: Optional[PeftModel] = None
        self.active_adapter: Optional[str] = None

    def _ensure_peft(self, spec: AdapterSpec):
        if self.peft_model is not None:
            return

        cfg = LoraConfig(
            r=spec.lora_r,
            lora_alpha=spec.lora_alpha,
            lora_dropout=spec.lora_dropout,
            bias="none",
            task_type="FEATURE_EXTRACTION",
            target_modules=list(spec.target_modules),
        )
        self.peft_model = get_peft_model(self.base_t5, cfg)
        self.peft_model.to(self.device)

    def add_adapter(self, spec: AdapterSpec):
        self._ensure_peft(spec)
        assert self.peft_model is not None
        # Create a named adapter config (PEFT supports multiple adapters)
        if spec.name not in self.peft_model.peft_config:
            cfg = LoraConfig(
                r=spec.lora_r,
                lora_alpha=spec.lora_alpha,
                lora_dropout=spec.lora_dropout,
                bias="none",
                task_type="FEATURE_EXTRACTION",
                target_modules=list(spec.target_modules),
            )
            self.peft_model.add_adapter(spec.name, cfg)

    def set_adapter(self, name: Optional[str], scale: float = 1.0):
        """
        name=None means disable adapters (use base).
        scale controls defense strength (LoRA scaling).
        """
        assert self.peft_model is not None
        if name is None:
            # 禁用适配器：不调用 peft_model.set_adapter(None)，因为PEFT不允许
            # 调用者应该直接使用 base_t5 而不是 peft_model
            self.active_adapter = None
            return

        # 激活指定的适配器
        self.peft_model.set_adapter(name)
        self.active_adapter = name
        
        # PEFT LoRA scaling: each lora layer has scaling = alpha/r; we can multiply.
        # Some PEFT versions expose `set_adapter` only; scaling can be done by
        # setting `lora_alpha` or using `peft_model.set_adapter` + manual scaling hook.
        # Here we implement a generic multiplier hook.
        for m in self.peft_model.modules():
            if hasattr(m, "scaling"):
                try:
                    m.scaling = float(m.scaling) * float(scale)
                except Exception:
                    pass

    def trainable_parameters(self) -> List[nn.Parameter]:
        assert self.peft_model is not None
        params = []
        for n, p in self.peft_model.named_parameters():
            if p.requires_grad:
                params.append(p)
        return params

    def save_adapter(self, name: str, out_dir: str):
        assert self.peft_model is not None
        os.makedirs(out_dir, exist_ok=True)
        self.peft_model.save_pretrained(out_dir, selected_adapters=[name])

    def load_adapter(self, name: str, from_dir: str, spec: AdapterSpec):
        self._ensure_peft(spec)
        assert self.peft_model is not None
        self.peft_model.load_adapter(from_dir, adapter_name=name)


# ---------------------------
# Open-Sora bindings (YOU MUST IMPLEMENT)
# ---------------------------
def load_opensora_components(opensora_repo: str, ckpt: str, device: torch.device, load_full_model: bool = False, 
                              offload_model: bool = False, use_multi_gpu: bool = False):
    """
    加载 Open-Sora 相关组件。

    参数:
      - opensora_repo: 本地 Open-Sora 仓库路径（用于加载DiT和VAE）
      - ckpt: 文本编码器模型路径或 HuggingFace 模型 ID
      - device: 加载到的 torch 设备
      - load_full_model: 是否加载完整的生成模型（DiT + VAE），推理时需要

    返回:
      dict 包含:
        - tokenizer: T5Tokenizer
        - text_encoder: T5EncoderModel
        - model: MMDiT模型（如果load_full_model=True）
        - model_ae: VAE解码器（如果load_full_model=True）
        - model_clip: CLIP模型（如果load_full_model=True）
        - opensora_repo: Open-Sora仓库路径
        - config: 配置对象（如果load_full_model=True）
    """
    # 加载 T5 文本编码器和 tokenizer
    tokenizer = None
    text_encoder = None

    # 情况1：diffusers 风格子目录
    tok_dir = os.path.join(ckpt, "tokenizer")
    te_dir = os.path.join(ckpt, "text_encoder")

    try:
        if os.path.isdir(tok_dir) and os.path.isdir(te_dir):
            print(f"[Open-Sora] 使用 diffusers 风格子目录加载 T5：{ckpt}")
            tokenizer = T5Tokenizer.from_pretrained(tok_dir, legacy=True)
            text_encoder = T5EncoderModel.from_pretrained(te_dir)
        else:
            # 情况2：直接作为模型ID或单一目录
            print(f"[Open-Sora] 使用 HuggingFace 模型或本地目录加载 T5：{ckpt}")
            tokenizer = T5Tokenizer.from_pretrained(ckpt, legacy=True)
            text_encoder = T5EncoderModel.from_pretrained(ckpt)
    except Exception as e:
        raise RuntimeError(
            f"[Open-Sora] 无法从 '{ckpt}' 加载 T5 tokenizer/text_encoder，"
            f"请确认 ckpt 是有效的 T5 模型路径或 HuggingFace 模型ID。原始错误: {e}"
        )

    text_encoder = text_encoder.to(device)
    text_encoder.eval()
    print(f"[Open-Sora] T5加载完成")

    # 如果需要加载完整模型（推理时）
    model = None
    model_ae = None
    model_clip = None
    config = None
    
    if load_full_model:
        print(f"[Open-Sora] 开始加载完整生成模型（DiT + VAE + CLIP）...")
        try:
            # 添加Open-Sora代码到路径
            if opensora_repo and os.path.exists(opensora_repo):
                if opensora_repo not in sys.path:
                    sys.path.insert(0, opensora_repo)
                
                # 导入Open-Sora的模块
                from mmengine.config import Config
                from opensora.registry import MODELS, build_module
                from opensora.utils.sampling import prepare_models
                
                # 创建配置（使用默认配置）
                # 模型权重路径
                model_dir = "/home/raykr/models/hpcai-tech/Open-Sora-v2"
                
                config_dict = {
                    "model": {
                        "type": "flux",
                        "from_pretrained": os.path.join(model_dir, "Open_Sora_v2.safetensors"),
                        "guidance_embed": False,
                        "fused_qkv": False,
                        "use_liger_rope": True,
                        "in_channels": 64,
                        "vec_in_dim": 768,
                        "context_in_dim": 4096,
                        "hidden_size": 3072,
                        "mlp_ratio": 4.0,
                        "num_heads": 24,
                        "depth": 19,
                        "depth_single_blocks": 38,
                        "axes_dim": [16, 56, 56],
                        "theta": 10_000,
                        "qkv_bias": True,
                        "cond_embed": True,
                    },
                    "ae": {
                        "type": "hunyuan_vae",
                        "from_pretrained": os.path.join(model_dir, "hunyuan_vae.safetensors"),
                        "in_channels": 3,
                        "out_channels": 3,
                        "layers_per_block": 2,
                        "latent_channels": 16,
                        "use_spatial_tiling": True,
                        "use_temporal_tiling": False,
                    },
                    "t5": {
                        "type": "text_embedder",
                        "from_pretrained": ckpt,  # 使用传入的T5路径
                        "max_length": 512,
                        "shardformer": False,  # 禁用shardformer以便使用我们的adapted版本
                    },
                    "clip": {
                        "type": "text_embedder",
                        "from_pretrained": os.path.join(model_dir, "openai/clip-vit-large-patch14"),
                        "max_length": 77,
                    },
                }
                
                config = Config(config_dict)
                dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
                
                # 清理GPU缓存
                torch.cuda.empty_cache()
                
                # 使用Open-Sora的prepare_models加载
                # offload_model=True 会将模型加载到CPU，推理时再移到GPU
                model, model_ae, _, model_clip, optional_models = prepare_models(
                    config, device, dtype, offload_model=offload_model
                )
                
                # 如果使用多GPU，将模型包装为DataParallel
                if use_multi_gpu and torch.cuda.device_count() > 1:
                    print(f"[Open-Sora] 使用 {torch.cuda.device_count()} 个GPU进行推理")
                    if not offload_model:  # 只有在模型在GPU上时才使用DataParallel
                        model = torch.nn.DataParallel(model)
                        model_ae = torch.nn.DataParallel(model_ae)
                        # CLIP和T5通常较小，可以放在单个GPU上
                
                # 清理缓存
                torch.cuda.empty_cache()
                
                print(f"[Open-Sora] ✅ 完整模型加载成功")
                print(f"  - DiT模型: {type(model)}")
                print(f"  - VAE模型: {type(model_ae)}")
                print(f"  - CLIP模型: {type(model_clip)}")
                
        except Exception as e:
            import traceback
            print(f"[Open-Sora] ❌ 加载完整模型失败: {e}")
            print(f"[Open-Sora] 错误详情:")
            traceback.print_exc()
            raise

    return {
        "tokenizer": tokenizer,
        "text_encoder": text_encoder,
        "model": model,
        "model_ae": model_ae,
        "model_clip": model_clip,
        "opensora_repo": opensora_repo,
        "config": config,
        "offload_model": offload_model,  # 保存offload状态
    }


def opensora_encode_prompt(tokenizer: T5Tokenizer, text_encoder: nn.Module, prompts: List[str], device: torch.device):
    """
    Encode prompts using T5 encoder.
    
    Note: This function does NOT use @torch.no_grad() to allow gradient computation
    during training. Use `with torch.no_grad():` when calling this function during inference.
    
    Return:
      - embeds: [B, L, D]
      - attention_mask: [B, L]
    """
    tok = tokenizer(
        prompts,
        padding="max_length",
        truncation=True,
        max_length=256,  # 可以减小到128以节省内存
        return_tensors="pt",
    )
    input_ids = tok.input_ids.to(device)
    attn = tok.attention_mask.to(device)
    
    # 直接调用，不使用autocast（避免与gradient checkpointing冲突）
    # 如果模型已经是FP16，会自动使用FP16计算
    out = text_encoder(input_ids=input_ids, attention_mask=attn)
    embeds = out.last_hidden_state
    
    return embeds, attn


@torch.no_grad()
def opensora_sample(components: dict, prompt_embeds: torch.Tensor, prompt_mask: torch.Tensor,
                    num_frames: int, height: int, width: int,
                    num_steps: int, guidance: float, seed: int,
                    out_path: str, prompts: List[str] = None):
    """
    Run Open-Sora sampling given prepared prompt embeddings.
    
    使用Open-Sora官方API进行推理。
    
    参数:
      - components: 包含模型组件的字典
      - prompt_embeds: T5编码的文本embeddings [B, L, D]
      - prompt_mask: attention mask [B, L]
      - num_frames: 视频帧数
      - height: 视频高度
      - width: 视频宽度
      - num_steps: 扩散步数
      - guidance: 引导强度
      - seed: 随机种子
      - out_path: 输出视频路径
      - prompts: 原始文本prompts（用于CLIP编码）
    """
    device = prompt_embeds.device
    dtype = prompt_embeds.dtype
    
    # 设置随机种子
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    
    # 获取模型组件
    model = components.get("model")
    model_ae = components.get("model_ae")
    model_clip = components.get("model_clip")
    opensora_repo = components.get("opensora_repo", "")
    offload_model = components.get("offload_model", False)
    
    # 检查模型是否已加载
    if model is None or model_ae is None:
        raise RuntimeError(
            "Open-Sora模型未加载。请确保在load_opensora_components中设置了load_full_model=True"
        )
    
    # 如果模型在CPU上（offload模式），需要移到GPU进行推理
    if offload_model:
        print(f"[opensora_sample] 将模型从CPU移到GPU...")
        model = model.to(device)
        model_ae = model_ae.to(device)
        if model_clip is not None:
            model_clip = model_clip.to(device)
        torch.cuda.empty_cache()
    # 否则模型已经在GPU上，直接使用
    
    # 添加Open-Sora代码到路径
    if opensora_repo and opensora_repo not in sys.path:
        sys.path.insert(0, opensora_repo)
    
    # 导入Open-Sora的采样工具
    from opensora.utils.sampling import (
        SamplingOption,
        prepare_api,
        prepare_ids,
        sanitize_sampling_option,
    )
    
    # 准备采样选项
    sampling_option = SamplingOption(
        height=height,
        width=width,
        num_frames=num_frames,
        num_steps=num_steps,
        guidance=guidance,
        seed=seed,
        method="i2v",  # 使用I2V方法
        shift=True,
        temporal_reduction=4,
        is_causal_vae=True,
    )
    sampling_option = sanitize_sampling_option(sampling_option)
    
    # 准备API函数（使用官方prepare_api）
    # 注意：我们需要创建一个自定义的T5编码器包装，使用我们的adapted embeddings
    api_fn = prepare_api(model, model_ae, None, model_clip, {})  # T5设为None，我们手动处理
    
    print(f"[opensora_sample] 开始生成视频...")
    print(f"  尺寸: {width}x{height}, 帧数: {num_frames}, 步数: {num_steps}, 引导: {guidance}")
    
    # 准备CLIP编码（如果需要）
    if model_clip is not None and prompts is not None:
        from opensora.models.text.conditioner import HFEmbedder
        if isinstance(model_clip, HFEmbedder):
            clip_embeds = model_clip(prompts)
        else:
            clip_embeds = model_clip(prompts)
    else:
        # 如果没有CLIP，创建零向量
        batch_size = prompt_embeds.shape[0]
        clip_embeds = torch.zeros(batch_size, 77, 768, device=device, dtype=dtype)
    
    # 创建初始噪声（使用Open-Sora的get_noise函数）
    from opensora.utils.sampling import get_noise
    batch_size = prompt_embeds.shape[0]
    z = get_noise(
        batch_size,
        height,
        width,
        num_frames,
        device,
        dtype,
        seed,
        patch_size=2,
        channel=16,
    )
    
    # 准备输入（使用prepare_ids，因为我们已经有T5 embeddings了）
    inp = prepare_ids(z, prompt_embeds, clip_embeds)
    
    # 准备采样参数
    # 我们需要手动调用denoiser，因为API函数期望使用T5编码器
    from opensora.utils.sampling import (
        SamplingMethod,
        SamplingMethodDict,
        get_schedule,
        unpack,
    )
    
    # 获取timesteps
    timesteps = get_schedule(
        num_steps,
        (z.shape[-1] * z.shape[-2]) // 4,  # patch_size^2 = 4
        num_frames,
        shift=sampling_option.shift,
    )
    
    # 使用I2V denoiser
    denoiser = SamplingMethodDict[SamplingMethod.I2V]
    
    # 准备guidance（对于t2v，不需要image guidance）
    text = prompts if prompts else [""] * batch_size
    text, additional_inp = denoiser.prepare_guidance(
        text=text,
        optional_models={},
        device=device,
        dtype=dtype,
        neg=None,
        guidance_img=None,
    )
    
    # 更新输入
    inp.update(additional_inp)
    
    # 对于t2v，不需要references
    masks = torch.zeros(batch_size, 1, num_frames, z.shape[-2], z.shape[-1], device=device, dtype=dtype)
    masked_ref = torch.zeros_like(z)
    inp["masks"] = masks
    inp["masked_ref"] = masked_ref
    inp["sigma_min"] = 1e-5
    
    # 运行去噪
    print(f"[opensora_sample] 运行扩散采样...")
    x = denoiser.denoise(
        model,
        **inp,
        timesteps=timesteps,
        guidance=guidance,
        text_osci=False,
        image_osci=False,
        scale_temporal_osci=False,
        flow_shift=None,
        patch_size=2,
    )
    
    # Unpack latent
    x = unpack(x, height, width, num_frames, patch_size=2)
    
    # VAE解码前清理缓存
    torch.cuda.empty_cache()
    
    # VAE解码
    print(f"[opensora_sample] VAE解码...")
    x = model_ae.decode(x)
    x = x[:, :, :num_frames]  # 确保帧数正确
    
    # 如果使用offload，推理完成后将模型移回CPU（可选，节省GPU内存）
    # 注意：如果后续还要推理，可以不移回CPU以保持速度
    # if offload_model:
    #     print(f"[opensora_sample] 将模型移回CPU以释放GPU内存...")
    #     model = model.cpu()
    #     model_ae = model_ae.cpu()
    #     if model_clip is not None:
    #         model_clip = model_clip.cpu()
    #     torch.cuda.empty_cache()
    
    # 保存视频
    print(f"[opensora_sample] 保存视频...")
    try:
        # 转换为numpy
        video_np = x[0].cpu().numpy()  # [C, T, H, W]
        
        # 转换为 [T, H, W, C] 格式
        video_np = np.transpose(video_np, (1, 2, 3, 0))  # [T, H, W, C]
        
        # 归一化到[0, 255]
        if video_np.max() <= 1.0:
            video_np = (video_np * 255).clip(0, 255)
        video_np = video_np.astype(np.uint8)
        
        # 确保RGB格式
        if video_np.shape[-1] == 1:
            video_np = np.repeat(video_np, 3, axis=-1)
        elif video_np.shape[-1] == 4:
            video_np = video_np[:, :, :, :3]
        
        # 保存视频
        os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else ".", exist_ok=True)
        imageio.mimwrite(out_path, video_np, fps=8, codec="libx264", quality=8)
        
        print(f"[opensora_sample] ✅ 视频已保存到: {out_path}")
        
    except Exception as e:
        print(f"[opensora_sample] 保存视频出错: {e}")
        import traceback
        traceback.print_exc()
        raise


# ---------------------------
# Training
# ---------------------------
def train_one_adapter(
    components: dict,
    adapter_bank: T5AdapterBank,
    classifier: ToxicClassifier,
    spec: AdapterSpec,
    train_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    lambda_tox: float,
    lambda_preserve: float,
    grad_clip: float = 1.0,
    use_gradient_checkpointing: bool = False,  # 默认禁用，避免梯度问题
):
    tokenizer: T5Tokenizer = components["tokenizer"]

    adapter_bank.add_adapter(spec)
    adapter_bank.set_adapter(spec.name, scale=1.0)

    # Freeze base text encoder weights except LoRA
    adapter_bank.peft_model.train()
    for n, p in adapter_bank.peft_model.named_parameters():
        # PEFT marks LoRA params trainable by default; freeze others
        if "lora_" not in n:
            p.requires_grad = False

    # Enable gradient checkpointing to save memory
    # 注意：gradient checkpointing可能与某些模型配置不兼容，如果出错可以禁用
    print(f"[{spec.name}] use_gradient_checkpointing={use_gradient_checkpointing}")
    if use_gradient_checkpointing:
        try:
            # 尝试在base_model上启用
            if hasattr(adapter_bank.peft_model, "base_model"):
                base_model = adapter_bank.peft_model.base_model
                if hasattr(base_model, "model") and hasattr(base_model.model, "gradient_checkpointing_enable"):
                    base_model.model.gradient_checkpointing_enable()
                    print(f"[{spec.name}] Gradient checkpointing enabled")
                elif hasattr(base_model, "gradient_checkpointing_enable"):
                    base_model.gradient_checkpointing_enable()
                    print(f"[{spec.name}] Gradient checkpointing enabled (base_model)")
                else:
                    print(f"[{spec.name}] Warning: Model does not support gradient checkpointing, skipping")
        except Exception as e:
            print(f"[{spec.name}] Warning: Failed to enable gradient checkpointing: {e}")
            print(f"[{spec.name}] Continuing without gradient checkpointing...")
    else:
        print(f"[{spec.name}] Gradient checkpointing disabled (default)")

    # Classifier frozen
    classifier.eval()
    for p in classifier.parameters():
        p.requires_grad = False

    optim = torch.optim.AdamW(adapter_bank.trainable_parameters(), lr=lr)

    for ep in range(epochs):
        for batch_idx, (prompts, labels) in enumerate(train_loader):
            # filter by this category
            keep = [i for i, y in enumerate(labels) if y == spec.name]
            if not keep:
                continue
            prompts = [prompts[i] for i in keep]

            # Base embeds (no adapter) - detach to ensure no gradient flow
            adapter_bank.set_adapter(None)
            with torch.no_grad():
                base_t5_model = adapter_bank.base_t5
                base_embeds, attn = opensora_encode_prompt(tokenizer, base_t5_model, prompts, device)
            base_embeds = base_embeds.detach()  # Ensure no gradient
            
            # Clear cache before forward pass
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Adapted embeds - should have gradient for LoRA training
            adapter_bank.set_adapter(spec.name, scale=1.0)
            # Ensure peft_model is in train mode for gradient computation
            adapter_bank.peft_model.train()
            
            # 确保LoRA参数有梯度（双重检查）
            trainable_count = 0
            for name, param in adapter_bank.peft_model.named_parameters():
                if "lora_" in name:
                    param.requires_grad = True
                    trainable_count += 1
            
            if trainable_count == 0:
                raise RuntimeError(f"No LoRA parameters found! Check adapter configuration for {spec.name}")
            
            # 如果启用了gradient checkpointing，需要确保输入有requires_grad
            # 但实际上input_ids不应该有梯度，所以gradient checkpointing可能不适用
            if use_gradient_checkpointing:
                print(f"[{spec.name}] Warning: Gradient checkpointing is enabled but may cause gradient issues with LoRA training")
            
            # 前向传播计算adapted_embeds（应该有梯度）
            adapted_embeds, attn2 = opensora_encode_prompt(tokenizer, adapter_bank.peft_model, prompts, device)
            
            # 验证adapted_embeds可以计算梯度（虽然embeds本身可能不直接requires_grad，
            # 但通过计算图连接到LoRA参数，应该可以反向传播）
            # 这里不检查adapted_embeds.requires_grad，因为它可能为False但仍有grad_fn

            # Toxic score for this category index
            y_idx = HARM_CATEGORIES.index(spec.name)
            logits = classifier.forward_text_embeds(adapted_embeds, attn2)
            # higher logit => more toxic; minimize it
            tox_loss = logits[:, y_idx].mean()

            preserve_loss = F.mse_loss(adapted_embeds, base_embeds)

            loss = lambda_tox * tox_loss + lambda_preserve * preserve_loss
            
            # 检查loss是否可以反向传播
            if loss.grad_fn is None:
                # 如果loss没有grad_fn，说明计算图断开
                print(f"Error: loss has no grad_fn. Checking computation graph...")
                print(f"  loss.requires_grad: {loss.requires_grad}")
                print(f"  loss.grad_fn: {loss.grad_fn}")
                print(f"  adapted_embeds.grad_fn: {adapted_embeds.grad_fn if hasattr(adapted_embeds, 'grad_fn') else 'N/A'}")
                # 检查是否有可训练参数
                trainable_params = [p for p in adapter_bank.peft_model.parameters() if p.requires_grad]
                print(f"  Found {len(trainable_params)} trainable parameters")
                if len(trainable_params) == 0:
                    raise RuntimeError("No trainable parameters! Check LoRA configuration.")
                
                # 如果启用了gradient checkpointing，这是已知问题
                if use_gradient_checkpointing:
                    raise RuntimeError(
                        "Loss has no gradient due to gradient checkpointing. "
                        "Gradient checkpointing is incompatible with LoRA training. "
                        "Please disable gradient checkpointing (set USE_GRADIENT_CHECKPOINTING=0)."
                    )
                else:
                    raise RuntimeError("Loss has no gradient. Check model configuration and training setup.")

            optim.zero_grad(set_to_none=True)
            loss.backward()
            
            # 检查梯度是否存在
            has_grad = any(p.grad is not None for p in adapter_bank.trainable_parameters())
            if not has_grad:
                print(f"Warning: No gradients found after backward pass!")
                # 打印一些调试信息
                for name, param in list(adapter_bank.peft_model.named_parameters())[:5]:
                    if param.requires_grad:
                        print(f"  {name}: requires_grad={param.requires_grad}, grad={param.grad is not None}")
            
            nn.utils.clip_grad_norm_(adapter_bank.trainable_parameters(), grad_clip)
            optim.step()
            
            # Clear cache after each step
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            if batch_idx % 10 == 0:
                print(f"[{spec.name}] epoch {ep+1}/{epochs}, batch {batch_idx}, loss={loss.item():.4f}")

        print(f"[{spec.name}] epoch {ep+1}/{epochs} done")

    out_dir = os.path.join(spec.out_dir, spec.name)
    adapter_bank.save_adapter(spec.name, out_dir)
    print(f"[{spec.name}] saved adapter to: {out_dir}")


# ---------------------------
# Inference
# ---------------------------
@torch.no_grad()
def infer_with_dynamic_adapter(
    components: dict,
    adapter_bank: T5AdapterBank,
    classifier: ToxicClassifier,
    prompt: str,
    defense_scale: float,
    out_path: str,
    num_frames: int,
    height: int,
    width: int,
    num_steps: int,
    guidance: float,
    seed: int,
    force_category: Optional[str] = None,
):
    tokenizer: T5Tokenizer = components["tokenizer"]

    if force_category is None:
        pred_labels, pred_probs = classifier.predict_label([prompt])
        category = pred_labels[0]
        score = pred_probs[0]
    else:
        category, score = force_category, 1.0

    if category not in HARM_CATEGORIES:
        category = "disturbing"

    adapter_bank.set_adapter(category, scale=defense_scale)

    # Encode with adapted T5
    prompt_embeds, prompt_mask = opensora_encode_prompt(tokenizer, adapter_bank.peft_model, [prompt], adapter_bank.device)

    opensora_sample(
        components=components,
        prompt_embeds=prompt_embeds,
        prompt_mask=prompt_mask,
        num_frames=num_frames,
        height=height,
        width=width,
        num_steps=num_steps,
        guidance=guidance,
        seed=seed,
        out_path=out_path,
        prompts=[prompt],  # 传入原始prompt用于CLIP编码
    )
    print(f"[infer] category={category}, score={score:.4f}, scale={defense_scale}, saved={out_path}")


# ---------------------------
# CLI
# ---------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["train", "infer"], required=True)

    # Open-Sora paths
    ap.add_argument("--opensora_repo", type=str, required=True, help="path to your local Open-Sora checkout")
    ap.add_argument("--opensora_ckpt", type=str, required=True)

    # Data/adapters
    ap.add_argument("--train_jsonl", type=str, default="")
    ap.add_argument("--adapter_root", type=str, required=True)

    # Training hyperparams
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lambda_tox", type=float, default=1.0)
    ap.add_argument("--lambda_preserve", type=float, default=0.1)
    ap.add_argument("--use_gradient_checkpointing", action="store_true", help="Enable gradient checkpointing to save memory")
    ap.add_argument("--multi_gpu", action="store_true", help="Use DataParallel for multi-GPU training")

    # Inference
    ap.add_argument("--prompt", type=str, default="")
    ap.add_argument("--offload_model", action="store_true", help="Offload model to CPU to save GPU memory (slower but uses less GPU memory)")
    ap.add_argument("--use_multi_gpu", action="store_true", help="Use multiple GPUs for inference (DataParallel)")
    ap.add_argument("--out", type=str, default="out.mp4")
    ap.add_argument("--defense_scale", type=float, default=1.0)
    ap.add_argument("--force_category", type=str, default="")

    # Video gen params
    ap.add_argument("--num_frames", type=int, default=49)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--width", type=int, default=848)
    ap.add_argument("--num_steps", type=int, default=30)
    ap.add_argument("--guidance", type=float, default=7.5)
    ap.add_argument("--seed", type=int, default=0)

    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Check available GPUs
    num_gpus = torch.cuda.device_count()
    print(f"Available GPUs: {num_gpus}")
    if args.multi_gpu and num_gpus > 1:
        print(f"Using {num_gpus} GPUs with DataParallel")
    else:
        print(f"Using single GPU: {device}")

    # 1) Load Open-Sora + T5
    # For multi-GPU, load on first GPU, then wrap with DataParallel
    primary_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # 推理时需要加载完整模型（DiT + VAE）
    load_full = (args.mode == "infer")
    
    # 内存优化选项
    offload_model = args.mode == "infer" and args.offload_model
    use_multi_gpu = args.mode == "infer" and args.use_multi_gpu
    
    # 清理GPU内存
    if args.mode == "infer":
        torch.cuda.empty_cache()
        import gc
        gc.collect()
    
    components = load_opensora_components(
        args.opensora_repo, 
        args.opensora_ckpt, 
        device=primary_device,
        load_full_model=load_full,
        offload_model=offload_model,
        use_multi_gpu=use_multi_gpu
    )
    tokenizer = components["tokenizer"]
    text_encoder = components["text_encoder"]
    text_encoder.to(primary_device).eval()
    
    # Wrap with DataParallel if multi-GPU
    if args.multi_gpu and num_gpus > 1:
        text_encoder = nn.DataParallel(text_encoder)
        print("Text encoder wrapped with DataParallel")

    # 2) Build adapter bank on T5
    # For DataParallel, need to access the underlying model
    base_t5_for_bank = text_encoder.module if isinstance(text_encoder, nn.DataParallel) else text_encoder
    bank = T5AdapterBank(base_t5=base_t5_for_bank, adapter_root=args.adapter_root, device=primary_device)

    # 3) Load classifier (replace with your real one)
    # IMPORTANT: change hidden size in ToxicClassifier if your T5 dim != 4096.
    classifier = ToxicClassifier(num_classes=len(HARM_CATEGORIES)).to(primary_device).eval()
    if args.multi_gpu and num_gpus > 1:
        classifier = nn.DataParallel(classifier)
        print("Classifier wrapped with DataParallel")

    if args.mode == "train":
        assert args.train_jsonl, "--train_jsonl required in train mode"
        ds = PromptLabelDataset(args.train_jsonl)
        dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_batch, num_workers=2)

        os.makedirs(args.adapter_root, exist_ok=True)

        for cat in HARM_CATEGORIES:
            spec = AdapterSpec(name=cat, out_dir=args.adapter_root)
            train_one_adapter(
                components=components,
                adapter_bank=bank,
                classifier=classifier,
                spec=spec,
                train_loader=dl,
                device=primary_device,
                epochs=args.epochs,
                lr=args.lr,
                lambda_tox=args.lambda_tox,
                lambda_preserve=args.lambda_preserve,
                use_gradient_checkpointing=args.use_gradient_checkpointing,  # 默认False（因为action="store_true"，只有显式传入--use_gradient_checkpointing才是True）
            )

    else:  # infer
        # load all adapters
        for cat in HARM_CATEGORIES:
            spec = AdapterSpec(name=cat, out_dir=args.adapter_root)
            bank.add_adapter(spec)
            # PEFT保存的路径可能是 adapter_root/cat/cat/ 或 adapter_root/cat/
            adapter_dir1 = os.path.join(args.adapter_root, cat, cat)  # 嵌套结构
            adapter_dir2 = os.path.join(args.adapter_root, cat)  # 扁平结构
            adapter_dir = None
            if os.path.isdir(adapter_dir1) and os.path.exists(os.path.join(adapter_dir1, "adapter_config.json")):
                adapter_dir = adapter_dir1
            elif os.path.isdir(adapter_dir2) and os.path.exists(os.path.join(adapter_dir2, "adapter_config.json")):
                adapter_dir = adapter_dir2
            elif os.path.isdir(adapter_dir2):
                # 检查子目录
                for subdir in os.listdir(adapter_dir2):
                    subpath = os.path.join(adapter_dir2, subdir)
                    if os.path.isdir(subpath) and os.path.exists(os.path.join(subpath, "adapter_config.json")):
                        adapter_dir = subpath
                        break
            
            if adapter_dir and os.path.isdir(adapter_dir):
                print(f"[infer] Loading adapter from: {adapter_dir}")
                bank.load_adapter(cat, adapter_dir, spec)
            else:
                print(f"[warn] adapter dir not found for {cat} (checked: {adapter_dir1}, {adapter_dir2})")

        force = args.force_category.strip() or None
        infer_with_dynamic_adapter(
            components=components,
            adapter_bank=bank,
            classifier=classifier,
            prompt=args.prompt,
            defense_scale=args.defense_scale,
            out_path=args.out,
            num_frames=args.num_frames,
            height=args.height,
            width=args.width,
            num_steps=args.num_steps,
            guidance=args.guidance,
            seed=args.seed,
            force_category=force,
        )


if __name__ == "__main__":
    main()
