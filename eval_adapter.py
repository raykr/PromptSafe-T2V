import torch
import pandas as pd
import os
import argparse
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
    # 1) 原始（未注入）pipeline (只在需要生成baseline时加载)
    pipe_raw = None
    if args.generate_baseline:
        pipe_raw = CogVideoXPipeline.from_pretrained(args.model_path, torch_dtype=torch.float16)
        pipe_raw.scheduler = CogVideoXDPMScheduler.from_config(pipe_raw.scheduler.config, timestep_spacing="trailing")
        pipe_raw.to(args.device)
        pipe_raw.vae.enable_slicing()
        pipe_raw.vae.enable_tiling()

    # 2) 注入 SafeAdapter 的 pipeline (只在需要生成防御视频时加载)
    pipe_safe = None
    if args.generate_defense:
        pipe_safe = CogVideoXPipeline.from_pretrained(args.model_path, torch_dtype=torch.float16)
        pipe_safe.scheduler = CogVideoXDPMScheduler.from_config(pipe_safe.scheduler.config, timestep_spacing="trailing")
        pipe_safe.to(args.device)
        pipe_safe.vae.enable_slicing()
        pipe_safe.vae.enable_tiling()
        pipe_safe = inject_safe_adapter(pipe_safe, args.adapter_path, args.rank, args.hidden_size)

    # 3) 加载 prompt 分类器（用于动态路由/强度控制）(只在需要生成防御视频时加载)
    cls_model = None
    cls_tokenizer = None
    cls_label_cols = None
    if args.generate_defense:
        cls_model, cls_tokenizer, cls_label_cols = load_prompt_classifier(args)

    os.makedirs(args.output_dir, exist_ok=True)

    # 从 testset_path 中读取 prompts
    data = pd.read_csv(args.testset_path)
    prompts = data["prompt"].tolist()
    
    for i, prompt in enumerate(prompts):
        # 生成baseline视频
        if args.generate_baseline:
            baseline_path = f"{args.output_dir}/adapter_{i:03d}_raw.mp4"
            if os.path.exists(baseline_path) and args.skip_existing:
                print(f"[{i:03d}] Baseline视频已存在，跳过: {baseline_path}")
            else:
                print(f"[{i:03d}] 生成baseline视频: {prompt[:40]}...")
                video_raw = pipe_raw(
                    prompt=prompt,
                    num_frames=args.num_frames,
                    height=args.height,
                    width=args.width,
                    num_inference_steps=args.num_inference_steps,
                    guidance_scale=args.guidance_scale,
                ).frames[0]
                export_to_video(video_raw, baseline_path, fps=args.fps)
                print(f"✅ Baseline视频已保存: {baseline_path}")

        # 生成防御视频
        if args.generate_defense:
            defense_path = f"{args.output_dir}/adapter_{i:03d}_safe.mp4"
            if os.path.exists(defense_path) and args.skip_existing:
                print(f"[{i:03d}] 防御视频已存在，跳过: {defense_path}")
            else:
                # 3.1 先用分类器预测该 prompt 的各类风险概率
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

                # 5) 生成已注入（safe，动态强度）
                video_safe = pipe_safe(
                    prompt=prompt,
                    num_frames=args.num_frames,
                    height=args.height,
                    width=args.width,
                    num_inference_steps=args.num_inference_steps,
                    guidance_scale=args.guidance_scale,
                ).frames[0]
                export_to_video(video_safe, defense_path, fps=args.fps)
                print(f"✅ 防御视频已保存: {defense_path}")


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
    # 1) raw pipeline (只在需要生成baseline时加载)
    pipe_raw = None
    if args.generate_baseline:
        pipe_raw = CogVideoXPipeline.from_pretrained(args.model_path, torch_dtype=torch.float16)
        pipe_raw.scheduler = CogVideoXDPMScheduler.from_config(pipe_raw.scheduler.config, timestep_spacing="trailing")
        pipe_raw.to(args.device)
        pipe_raw.vae.enable_slicing()
        pipe_raw.vae.enable_tiling()

    # 2) safe pipeline with multi adapters (只在需要生成防御视频时加载)
    pipe_safe = None
    if args.generate_defense:
        pipe_safe = CogVideoXPipeline.from_pretrained(args.model_path, torch_dtype=torch.float16)
        pipe_safe.scheduler = CogVideoXDPMScheduler.from_config(pipe_safe.scheduler.config, timestep_spacing="trailing")
        pipe_safe.to(args.device)
        pipe_safe.vae.enable_slicing()
        pipe_safe.vae.enable_tiling()

        # 2.1 注入多 adapter
        adapter_ckpt_map = args.adapter_ckpt_map  # dict[str,str]
        pipe_safe = inject_multi_safe_adapters(pipe_safe, adapter_ckpt_map, args.rank, args.hidden_size)

    # 3) 加载 prompt 分类器 (只在需要生成防御视频时加载)
    cls_model = None
    cls_tokenizer = None
    label_cols = None
    if args.generate_defense:
        cls_model, cls_tokenizer, label_cols = load_prompt_classifier(args)

    os.makedirs(args.output_dir, exist_ok=True)
    data = pd.read_csv(args.testset_path)
    prompts = data["prompt"].tolist()

    for i, prompt in enumerate(prompts):
        # 生成baseline视频
        if args.generate_baseline:
            baseline_path = f"{args.output_dir}/multi_{i:03d}_raw.mp4"
            if os.path.exists(baseline_path) and args.skip_existing:
                print(f"[{i:03d}] Baseline视频已存在，跳过: {baseline_path}")
            else:
                print(f"[{i:03d}] 生成baseline视频: {prompt[:40]}...")
                video_raw = pipe_raw(
                    prompt=prompt,
                    num_frames=args.num_frames,
                    height=args.height,
                    width=args.width,
                    num_inference_steps=args.num_inference_steps,
                    guidance_scale=args.guidance_scale,
                ).frames[0]
                export_to_video(video_raw, baseline_path, fps=args.fps)
                print(f"✅ Baseline视频已保存: {baseline_path}")

        # 生成防御视频
        if args.generate_defense:
            defense_path = f"{args.output_dir}/multi_{i:03d}_safe.mp4"
            if os.path.exists(defense_path) and args.skip_existing:
                print(f"[{i:03d}] 防御视频已存在，跳过: {defense_path}")
            else:
                # 3.1 先跑分类器
                tok = cls_tokenizer([prompt], padding=True, truncation=True, return_tensors="pt").to(args.device)
                logits = cls_model(tok["input_ids"], tok["attention_mask"])
                probs = torch.sigmoid(logits)  # [1,C]

                category, scale = route_from_probs(probs, label_cols, thresh=args.route_thresh)

                print(f"[{i:03d}] prompt={prompt[:40]}..., cat={category}, scale={scale:.3f}")

                # 3.2 设置当前路由
                pipe_safe.text_encoder.set_adapter_route(category=category, scale=scale)

                # 5) safe 生成（自动选 adapter + 强度）
                video_safe = pipe_safe(
                    prompt=prompt,
                    num_frames=args.num_frames,
                    height=args.height,
                    width=args.width,
                    num_inference_steps=args.num_inference_steps,
                    guidance_scale=args.guidance_scale,
                ).frames[0]
                export_to_video(video_safe, defense_path, fps=args.fps)
                print(f"✅ 防御视频已保存: {defense_path}")


def parse_adapter_map(adapter_map_str: str) -> dict[str, str]:
    """
    解析adapter映射字符串，格式: "category1:path1,category2:path2"
    例如: "sexual:checkpoints/sexual/safe_adapter.pt,violent:checkpoints/violent/safe_adapter.pt"
    """
    adapter_map = {}
    for item in adapter_map_str.split(','):
        if ':' not in item:
            raise ValueError(f"无效的adapter映射格式: {item}，应为 'category:path'")
        category, path = item.split(':', 1)
        adapter_map[category.strip()] = path.strip()
    return adapter_map


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="评估SafeAdapter模型（单adapter或多adapter）")
    
    # 模式选择
    parser.add_argument("--mode", type=str, choices=["single", "multi"], default="multi",
                        help="评估模式: single（单adapter）或 multi（多adapter路由）")
    
    # 必需参数
    parser.add_argument("--testset_path", type=str, required=True,
                        help="测试集CSV文件路径")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="输出目录")
    
    # Adapter路径（根据模式选择）
    parser.add_argument("--adapter_path", type=str, default=None,
                        help="单adapter模式：Adapter checkpoint路径")
    parser.add_argument("--adapter_map", type=str, default=None,
                        help="多adapter模式：Adapter映射，格式: 'category1:path1,category2:path2'")
    
    # 模型相关参数
    parser.add_argument("--model_path", type=str,
                        default="/home/beihang/jzl/models/zai-org/CogVideoX-5b",
                        help="基础模型路径")
    parser.add_argument("--hidden_size", type=int, default=4096,
                        help="隐藏层大小")
    parser.add_argument("--rank", type=int, default=256,
                        help="Adapter的rank")
    
    # 分类器相关参数
    parser.add_argument("--cls_ckpt_path", type=str, default=None,
                        help="分类器checkpoint路径（生成防御视频时需要）")
    
    # 生成控制参数
    parser.add_argument("--generate_baseline", action="store_true",
                        help="生成baseline视频（未使用adapter）")
    parser.add_argument("--generate_defense", action="store_true",
                        help="生成防御视频（使用adapter）")
    parser.add_argument("--skip_existing", action="store_true",
                        help="如果视频已存在则跳过")
    
    # 视频生成参数
    parser.add_argument("--num_frames", type=int, default=49,
                        help="视频帧数")
    parser.add_argument("--height", type=int, default=480,
                        help="视频高度")
    parser.add_argument("--width", type=int, default=480,
                        help="视频宽度")
    parser.add_argument("--num_inference_steps", type=int, default=50,
                        help="推理步数")
    parser.add_argument("--guidance_scale", type=float, default=7.5,
                        help="Guidance scale")
    parser.add_argument("--fps", type=int, default=8,
                        help="视频FPS")
    
    # 路由相关参数（仅multi模式）
    parser.add_argument("--route_thresh", type=float, default=0.3,
                        help="路由阈值（低于此值不启用adapter，仅multi模式）")
    
    # 设备参数
    parser.add_argument("--device", type=str, default="cuda",
                        help="设备 (cuda/cpu)")
    
    args = parser.parse_args()
    
    # 验证至少选择一种生成模式
    if not args.generate_baseline and not args.generate_defense:
        parser.error("必须至少选择 --generate_baseline 或 --generate_defense 之一")
    
    # 验证分类器路径（仅在生成防御视频时需要）
    if args.generate_defense and args.cls_ckpt_path is None:
        parser.error("生成防御视频需要 --cls_ckpt_path 参数")
    
    # 根据模式验证参数
    if args.mode == "single":
        if args.adapter_path is None:
            parser.error("single模式需要 --adapter_path 参数")
    elif args.mode == "multi":
        if args.adapter_map is None:
            parser.error("multi模式需要 --adapter_map 参数")
        # 解析adapter映射
        args.adapter_ckpt_map = parse_adapter_map(args.adapter_map)
    
    print("=" * 60)
    print("评估配置:")
    print(f"  模式: {args.mode}")
    print(f"  测试集: {args.testset_path}")
    print(f"  输出目录: {args.output_dir}")
    print(f"  生成baseline: {args.generate_baseline}")
    print(f"  生成防御: {args.generate_defense}")
    print(f"  跳过已存在: {args.skip_existing}")
    if args.mode == "single":
        print(f"  Adapter路径: {args.adapter_path}")
    else:
        print(f"  Adapter映射: {args.adapter_ckpt_map}")
    print("=" * 60)
    
    # 根据模式调用不同的函数
    if args.mode == "single":
        eval_adapter(args)
    else:
        eval_adapter_multi(args)
