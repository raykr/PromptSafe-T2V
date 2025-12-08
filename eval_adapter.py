import torch
import pandas as pd
import os
from tqdm import tqdm
from diffusers import CogVideoXPipeline, CogVideoXDPMScheduler
from diffusers.utils import export_to_video
from transformers import AutoTokenizer, T5EncoderModel
from models import AdapterRouter, SafeAdapter, WrappedTextEncoderRouter
from models import WrappedTextEncoder
from classifier import PromptSafetyClassifier


def inject_safe_adapter(pipe, adapter_path, rank=256, hidden_size=4096):
    # 1) 构建并加载 Adapter
    adapter = SafeAdapter(hidden_size, rank)
    sd = torch.load(adapter_path, map_location="cpu")
    adapter.load_state_dict(sd)

    # 2) 包装原始 text_encoder
    wrapped = WrappedTextEncoder(pipe.text_encoder, adapter)

    # 3) 先放到与 pipeline 一致的 dtype/device（常见 float16 + cuda）
    wrapped = wrapped.to(device=pipe.device, dtype=getattr(pipe.text_encoder, "dtype", torch.float16))

    # 4) 替换并“正规”注册到 pipeline
    pipe.text_encoder = wrapped
    components = pipe.components.copy()
    components["text_encoder"] = pipe.text_encoder
    pipe.register_modules(**components)

    # 5) 再统一到 cuda（有些模块可能被 register 重置了 device）
    pipe.to(pipe.device)

    print("✅ SafeAdapter 已注入到 pipeline.text_encoder 并完成重新注册/对齐")
    return pipe


def inject_multi_safe_adapters(
    pipe,
    adapter_ckpt_map: dict[str, str],
    rank=256,
    hidden_size=4096,
):
    """
    adapter_ckpt_map: {category_name: ckpt_path}
      例如:
        {
          "sexual": "checkpoints/sexual_adapter.pt",
          "violent": "checkpoints/violent_adapter.pt",
          ...
        }
    """
    adapters = {}
    for cat, ckpt_path in adapter_ckpt_map.items():
        adp = SafeAdapter(hidden_size, rank)
        sd = torch.load(ckpt_path, map_location="cpu")
        adp.load_state_dict(sd)
        adapters[cat] = adp

    router = AdapterRouter(adapters)
    wrapped = WrappedTextEncoderRouter(pipe.text_encoder, router)

    wrapped = wrapped.to(device=pipe.device, dtype=getattr(pipe.text_encoder, "dtype", torch.float16))

    pipe.text_encoder = wrapped
    components = pipe.components.copy()
    components["text_encoder"] = pipe.text_encoder
    pipe.register_modules(**components)
    pipe.to(pipe.device)

    print(f"✅ Multi SafeAdapters 已注入: {list(adapter_ckpt_map.keys())}")
    return pipe


def load_prompt_classifier(args):
    """
    从 ckpt 加载 PromptSafetyClassifier，返回 (classifier, tokenizer, label_cols)
    """
    device = args.device
    state = torch.load(args.cls_ckpt_path, map_location="cpu")
    label_cols = state["label_cols"]

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, subfolder="tokenizer")
    base = T5EncoderModel.from_pretrained(args.model_path, subfolder="text_encoder").to(device)

    model = PromptSafetyClassifier(
        t5_encoder=base,
        hidden_size=args.hidden_size,
        num_labels=len(label_cols),
    ).to(device)
    model.load_state_dict(state["state_dict"])
    model.eval()

    return model, tokenizer, label_cols


@torch.no_grad()
def compute_severity(probs: torch.Tensor) -> torch.Tensor:
    """
    一个简单的 severity 定义示例：
      - probs: [B, num_labels]
      - 返回 severity: [B] ∈ [0,1]
    这里直接取所有类别概率的 max 作为整体有害程度，你可以根据需要改成加权和等。
    """
    severity, _ = probs.max(dim=-1)
    return severity  # [B]

def eval_adapter(args):
    # 1) 原始（未注入）pipeline
    pipe_raw = CogVideoXPipeline.from_pretrained(args.model_path, torch_dtype=torch.float16)
    pipe_raw.scheduler = CogVideoXDPMScheduler.from_config(pipe_raw.scheduler.config, timestep_spacing="trailing")
    pipe_raw.to(args.device)
    pipe_raw.vae.enable_slicing()
    pipe_raw.vae.enable_tiling()

    # 2) 注入 SafeAdapter 的 pipeline
    pipe_safe = CogVideoXPipeline.from_pretrained(args.model_path, torch_dtype=torch.float16)
    pipe_safe.scheduler = CogVideoXDPMScheduler.from_config(pipe_safe.scheduler.config, timestep_spacing="trailing")
    pipe_safe.to(args.device)
    pipe_safe.vae.enable_slicing()
    pipe_safe.vae.enable_tiling()
    pipe_safe = inject_safe_adapter(pipe_safe, args.adapter_path, args.rank, args.hidden_size)

    # 3) 加载 prompt 分类器（用于动态路由/强度控制）
    cls_model, cls_tokenizer, cls_label_cols = load_prompt_classifier(args)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # 从 testset_path 中读取 prompts
    data = pd.read_csv(args.testset_path)
    prompts = data["prompt"].tolist()
    for i, prompt in enumerate(prompts):
        # ---- 3.1 先用分类器预测该 prompt 的各类风险概率 ----
        tok = cls_tokenizer([prompt], padding=True, truncation=True, return_tensors="pt").to(args.device)
        logits = cls_model(tok["input_ids"], tok["attention_mask"])    # [1,num_labels]
        probs = torch.sigmoid(logits)                                  # [1,num_labels]

        severity = compute_severity(probs)[0].item()  # 标量 ∈ [0,1]
        # 你可以根据需要对 severity 做一个映射，比如:
        #   scale = 0.2 + 0.8 * severity
        # 代表最低 0.2 强度，最高 1.0 强度
        scale = 0.2 + 0.8 * severity

        # 将动态 scale 写入 text_encoder
        if hasattr(pipe_safe.text_encoder, "set_adapter_scale"):
            pipe_safe.text_encoder.set_adapter_scale(scale)
        else:
            # 兼容性：旧版可以直接写属性
            pipe_safe.text_encoder.adapter_scale = scale

        print(f"[{i:03d}] prompt = {prompt[:40]}..., severity = {severity:.3f}, scale = {scale:.3f}")

        # ---- 4) 生成未注入（raw） ----
        video_raw = pipe_raw(
            prompt=prompt,
            num_frames=args.num_frames,
            height=args.height,
            width=args.width,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
        ).frames[0]
        export_to_video(video_raw, f"{args.output_dir}/adapter_{i:03d}_raw.mp4", fps=args.fps)
        print(f"✅ 视频已保存到 {args.output_dir}/adapter_{i:03d}_raw.mp4")

        # ---- 5) 生成已注入（safe，动态强度）----
        video_safe = pipe_safe(
            prompt=prompt,
            num_frames=args.num_frames,
            height=args.height,
            width=args.width,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
        ).frames[0]
        export_to_video(video_safe, f"{args.output_dir}/adapter_{i:03d}_safe.mp4", fps=args.fps)
        print(f"✅ 视频已保存到 {args.output_dir}/adapter_{i:03d}_safe.mp4")


@torch.no_grad()
def route_from_probs(probs: torch.Tensor, label_cols: list[str], thresh: float = 0.3):
    """
    probs: [1, C]
    返回: (category_name or None, scale)
    """
    # 取最大概率及其类别
    max_prob, idx = probs.max(dim=-1)  # [1]
    max_prob = max_prob.item()
    idx = idx.item()

    if max_prob < thresh:
        # 整体风险很低，直接不启用任何 adapter
        return None, 0.0

    cat = label_cols[idx]

    # 一个简单的强度映射：scale ∈ [0.2, 1.0]
    scale = 0.2 + 0.8 * max_prob
    return cat, scale


def eval_adapter_multi(args):
    # 1) raw pipeline
    pipe_raw = CogVideoXPipeline.from_pretrained(args.model_path, torch_dtype=torch.float16)
    pipe_raw.scheduler = CogVideoXDPMScheduler.from_config(pipe_raw.scheduler.config, timestep_spacing="trailing")
    pipe_raw.to(args.device)
    pipe_raw.vae.enable_slicing()
    pipe_raw.vae.enable_tiling()

    # 2) safe pipeline with multi adapters
    pipe_safe = CogVideoXPipeline.from_pretrained(args.model_path, torch_dtype=torch.float16)
    pipe_safe.scheduler = CogVideoXDPMScheduler.from_config(pipe_safe.scheduler.config, timestep_spacing="trailing")
    pipe_safe.to(args.device)
    pipe_safe.vae.enable_slicing()
    pipe_safe.vae.enable_tiling()

    # 2.1 注入多 adapter
    adapter_ckpt_map = args.adapter_ckpt_map  # dict[str,str]
    pipe_safe = inject_multi_safe_adapters(pipe_safe, adapter_ckpt_map, args.rank, args.hidden_size)

    # 3) 加载 prompt 分类器
    cls_model, cls_tokenizer, label_cols = load_prompt_classifier(args)

    os.makedirs(args.output_dir, exist_ok=True)
    data = pd.read_csv(args.testset_path)
    prompts = data["prompt"].tolist()

    for i, prompt in enumerate(prompts):
        # 3.1 先跑分类器
        tok = cls_tokenizer([prompt], padding=True, truncation=True, return_tensors="pt").to(args.device)
        logits = cls_model(tok["input_ids"], tok["attention_mask"])
        probs = torch.sigmoid(logits)  # [1,C]

        category, scale = route_from_probs(probs, label_cols, thresh=args.route_thresh)

        print(f"[{i:03d}] prompt={prompt[:40]}..., cat={category}, scale={scale:.3f}")

        # 3.2 设置当前路由
        pipe_safe.text_encoder.set_adapter_route(category=category, scale=scale)

        # 4) raw 生成
        video_raw = pipe_raw(
            prompt=prompt,
            num_frames=args.num_frames,
            height=args.height,
            width=args.width,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
        ).frames[0]
        export_to_video(video_raw, f"{args.output_dir}/multi_{i:03d}_raw.mp4", fps=args.fps)

        # 5) safe 生成（自动选 adapter + 强度）
        video_safe = pipe_safe(
            prompt=prompt,
            num_frames=args.num_frames,
            height=args.height,
            width=args.width,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
        ).frames[0]
        export_to_video(video_safe, f"{args.output_dir}/multi_{i:03d}_safe.mp4", fps=args.fps)
