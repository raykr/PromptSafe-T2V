import torch
import pandas as pd
import os
import argparse
import random
import numpy as np
from contextlib import contextmanager
from tqdm import tqdm
from diffusers import CogVideoXPipeline, CogVideoXDPMScheduler
from diffusers.utils import export_to_video
from transformers import AutoTokenizer, T5EncoderModel
from models import AdapterRouter, SafeAdapter, WrappedTextEncoderRouter
from models import WrappedTextEncoder
from train_classifier import PromptSafetyClassifier

# 尝试导入 WanPipeline（如果可用）
# 先导入 ftfy 并确保它在全局命名空间中（WanPipeline 内部需要但可能没有正确导入）
try:
    import ftfy
    import sys
    import builtins
    # 确保 ftfy 在 sys.modules 和 builtins 中，这样 WanPipeline 内部可以找到它
    sys.modules['ftfy'] = ftfy
    builtins.ftfy = ftfy
except ImportError:
    ftfy = None
    print("⚠️ 警告: ftfy 未安装，WanPipeline 可能需要它。请运行: pip install ftfy")

# 延迟导入 WanPipeline，在需要时再导入并修复 ftfy 问题
WAN_PIPELINE_AVAILABLE = False
WanPipeline = None

def _ensure_ftfy_for_wan():
    """确保 ftfy 在 WanPipeline 模块中可用"""
    try:
        import ftfy
        import sys
        import builtins
        # 确保 ftfy 在多个位置可用
        sys.modules['ftfy'] = ftfy
        builtins.ftfy = ftfy
        
        # 尝试在 pipeline_wan 模块中注入 ftfy
        try:
            # 先导入 diffusers.pipelines.wan 包
            import diffusers.pipelines.wan
            # 然后导入 pipeline_wan 模块
            import diffusers.pipelines.wan.pipeline_wan as wan_pipeline_module
            # 在模块级别注入 ftfy
            wan_pipeline_module.ftfy = ftfy
            # 也尝试在包的 __init__ 中注入
            if hasattr(diffusers.pipelines.wan, '__init__'):
                diffusers.pipelines.wan.ftfy = ftfy
        except (ImportError, AttributeError) as e:
            # 如果导入失败，尝试直接访问已加载的模块
            import importlib
            try:
                wan_pipeline_module = sys.modules.get('diffusers.pipelines.wan.pipeline_wan')
                if wan_pipeline_module:
                    wan_pipeline_module.ftfy = ftfy
            except:
                pass
        return True
    except ImportError:
        return False

def _try_import_wan_pipeline():
    """尝试导入 WanPipeline"""
    global WAN_PIPELINE_AVAILABLE, WanPipeline
    if _ensure_ftfy_for_wan():
        try:
            from diffusers import WanPipeline
            WAN_PIPELINE_AVAILABLE = True
            return WanPipeline
        except (ImportError, AttributeError):
            try:
                from diffusers.pipelines.wan.pipeline_wan import WanPipeline
                WAN_PIPELINE_AVAILABLE = True
                return WanPipeline
            except ImportError:
                pass
    return None


def load_pipeline(model_path: str, model_type: str, device: str = "cuda", torch_dtype=torch.float16):
    """
    根据模型类型加载相应的 pipeline
    model_type: 'cogvideox' 或 'wan'
    """
    if model_type == "wan":
        # 尝试导入 WanPipeline（如果还没有导入）
        global WanPipeline, WAN_PIPELINE_AVAILABLE
        if WanPipeline is None:
            WanPipeline = _try_import_wan_pipeline()
        
        if not WAN_PIPELINE_AVAILABLE or WanPipeline is None:
            raise ImportError("WanPipeline 不可用，请确保安装了支持 Wan2.1 的 diffusers 版本")
        
        # 在加载 pipeline 之前，确保 ftfy 在全局命名空间中可用
        _ensure_ftfy_for_wan()
        
        pipe = WanPipeline.from_pretrained(model_path, torch_dtype=torch_dtype)
        # WanPipeline 使用 UniPCMultistepScheduler，通常不需要额外配置
        # 但可以检查是否需要 enable_slicing/tiling
        if hasattr(pipe.vae, 'enable_slicing'):
            pipe.vae.enable_slicing()
        if hasattr(pipe.vae, 'enable_tiling'):
            pipe.vae.enable_tiling()
    else:  # cogvideox
        pipe = CogVideoXPipeline.from_pretrained(model_path, torch_dtype=torch.float16)
        pipe.scheduler = CogVideoXDPMScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
        pipe.vae.enable_slicing()
        pipe.vae.enable_tiling()
    
    pipe.to(device)
    return pipe, model_type


def get_pipeline_kwargs(model_type: str, prompt: str, num_frames: int, height: int, width: int,
                       num_inference_steps: int, guidance_scale: float, generator: torch.Generator):
    """
    根据模型类型生成 pipeline 调用参数
    """
    kwargs = {
        "prompt": prompt,
        "num_frames": num_frames,
        "height": height,
        "width": width,
        "num_inference_steps": num_inference_steps,
        "guidance_scale": guidance_scale,
        "generator": generator,
    }
    # CogVideoX 需要 use_dynamic_cfg，Wan 可能不需要
    if model_type == "cogvideox":
        kwargs["use_dynamic_cfg"] = True
    
    return kwargs


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


@contextmanager
def temporary_device(model, target_device):
    """
    临时将模型移到目标设备，使用完后移回原设备
    """
    original_device = next(model.parameters()).device
    moved = False
    if original_device != target_device:
        model.to(target_device)
        moved = True
    try:
        yield model
    finally:
        if moved:
            model.to(original_device)


def load_prompt_classifier(args):
    """
    从 ckpt 加载 PromptSafetyClassifier，返回 (classifier, tokenizer, label_cols)
    支持将分类器加载到不同的设备（如 CPU）以节省显存
    """
    # 如果指定了 cls_device，使用它；否则默认使用 CPU 以节省显存
    cls_device = getattr(args, "cls_device", None)
    if cls_device is None:
        cls_device = "cpu"  # 默认 offload 到 CPU
    
    state = torch.load(args.cls_ckpt_path, map_location="cpu")
    label_cols = state["label_cols"]

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, subfolder="tokenizer")
    # 分类器加载到指定设备（通常是 CPU）
    base = T5EncoderModel.from_pretrained(args.model_path, subfolder="text_encoder").to(cls_device)

    model = PromptSafetyClassifier(
        t5_encoder=base,
        hidden_size=args.hidden_size,
        num_labels=len(label_cols),
    ).to(cls_device)
    
    # 加载分类头参数（兼容旧格式：如果包含 encoder 参数则过滤掉）
    saved_state = state["state_dict"]
    model_state = model.state_dict()
    
    # 如果保存的是完整模型，只提取分类头部分
    if "encoder" in saved_state or any(k.startswith("encoder.") for k in saved_state.keys()):
        # 旧格式：包含 T5 encoder，只加载分类头部分
        classifier_state = {k: v for k, v in saved_state.items() 
                           if k.startswith("ln.") or k.startswith("head.")}
    else:
        # 新格式：只包含分类头
        classifier_state = saved_state
    
    model.load_state_dict(classifier_state, strict=False)
    model.eval()
    
    print(f"✅ 分类器已加载到设备: {cls_device} (Pipeline 在 {args.device})")

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
    # 0) 固定随机种子，保证可复现
    if getattr(args, "seed", None) is not None:
        seed = int(args.seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    # 1) 原始（未注入）pipeline (只在需要生成baseline时加载)
    pipe_raw = None
    if args.generate_baseline:
        pipe_raw, _ = load_pipeline(args.model_path, args.model_type, args.device, torch.float16)

    # 2) 注入 SafeAdapter 的 pipeline (只在需要生成防御视频时加载)
    pipe_safe = None
    if args.generate_defense:
        pipe_safe, _ = load_pipeline(args.model_path, args.model_type, args.device, torch.float16)
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
                # 为每个样本构造独立但可复现的 generator
                gen = torch.Generator(device=args.device)
                if getattr(args, "seed", None) is not None:
                    gen.manual_seed(int(args.seed) + i)
                pipe_kwargs = get_pipeline_kwargs(
                    args.model_type, prompt, args.num_frames, args.height, args.width,
                    args.num_inference_steps, args.guidance_scale, gen
                )
                video_raw = pipe_raw(**pipe_kwargs).frames[0]
                export_to_video(video_raw, baseline_path, fps=args.fps)
                print(f"✅ Baseline视频已保存: {baseline_path}")

        # 生成防御视频
        if args.generate_defense:
            defense_path = f"{args.output_dir}/adapter_{i:03d}_safe.mp4"
            if os.path.exists(defense_path) and args.skip_existing:
                print(f"[{i:03d}] 防御视频已存在，跳过: {defense_path}")
            else:
                # 3.1 先用分类器预测该 prompt 的各类风险概率
                # 如果分类器在 CPU，临时移到 GPU 进行推理
                cls_device = getattr(args, "cls_device", "cpu")
                inference_device = args.device if cls_device == "cpu" else cls_device
                
                with temporary_device(cls_model, inference_device):
                    tok = cls_tokenizer([prompt], padding=True, truncation=True, return_tensors="pt").to(inference_device)
                    logits = cls_model(tok["input_ids"], tok["attention_mask"])    # [1,num_labels]
                    probs = torch.sigmoid(logits)                                  # [1,num_labels]
                    # 移到 CPU 以便后续处理
                    probs = probs.cpu()

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
                gen = torch.Generator(device=args.device)
                if getattr(args, "seed", None) is not None:
                    gen.manual_seed(int(args.seed) + i)
                pipe_kwargs = get_pipeline_kwargs(
                    args.model_type, prompt, args.num_frames, args.height, args.width,
                    args.num_inference_steps, args.guidance_scale, gen
                )
                video_safe = pipe_safe(**pipe_kwargs).frames[0]
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
    # 0) 固定随机种子，保证可复现
    if getattr(args, "seed", None) is not None:
        seed = int(args.seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    
    os.makedirs(args.output_dir, exist_ok=True)
    data = pd.read_csv(args.testset_path)
    prompts = data["prompt"].tolist()
    
    # ========== 阶段1: 批量分类所有 prompts（如果生成防御视频） ==========
    classification_results = None
    if args.generate_defense:
        print("=" * 60)
        print("阶段1: 批量分类所有 prompts")
        print("=" * 60)
        
        # 加载分类器
        cls_model, cls_tokenizer, label_cols = load_prompt_classifier(args)
        cls_device = getattr(args, "cls_device", "cpu")
        
        # 智能设备选择：处理 CUDA_VISIBLE_DEVICES 的情况
        # 如果 cls_device 是 cuda 设备但不可用，则使用 pipeline 设备或 CPU
        def get_available_device(preferred_device):
            """获取可用的设备，如果 preferred_device 不可用则回退"""
            if preferred_device == "cpu":
                return "cpu"
            if preferred_device.startswith("cuda"):
                try:
                    # 检查设备是否可用
                    device_id = int(preferred_device.split(":")[1]) if ":" in preferred_device else 0
                    if device_id < torch.cuda.device_count():
                        test_tensor = torch.zeros(1).to(preferred_device)
                        return preferred_device
                except (RuntimeError, AssertionError, IndexError, ValueError):
                    pass
                # 设备不可用，尝试使用 cuda:0（在 CUDA_VISIBLE_DEVICES 下可能是唯一可用设备）
                if torch.cuda.is_available():
                    try:
                        test_tensor = torch.zeros(1).to("cuda:0")
                        return "cuda:0"
                    except:
                        pass
            return "cpu"
        
        # 确定推理设备
        if cls_device == "cpu":
            # 如果分类器在 CPU，推理时可以使用 pipeline 设备（更高效）
            inference_device = get_available_device(args.device)
        else:
            # 如果指定了 cls_device，优先使用它
            inference_device = get_available_device(cls_device)
            if inference_device != cls_device:
                print(f"⚠️ 警告: {cls_device} 不可用，改用 {inference_device} 进行分类")
        
        # 批量分类所有 prompts
        classification_results = []
        batch_size = getattr(args, "cls_batch_size", 32)  # 可以批量处理提高效率
        
        with temporary_device(cls_model, inference_device):
            for i in range(0, len(prompts), batch_size):
                batch_prompts = prompts[i:i+batch_size]
                batch_indices = list(range(i, min(i+batch_size, len(prompts))))
                
                # 批量 tokenize
                tok = cls_tokenizer(
                    batch_prompts, 
                    padding=True, 
                    truncation=True, 
                    max_length=512,
                    return_tensors="pt"
                ).to(inference_device)
                
                # 批量推理
                with torch.no_grad():
                    logits = cls_model(tok["input_ids"], tok["attention_mask"])  # [B, C]
                    probs = torch.sigmoid(logits).cpu()  # [B, C]
                
                # 保存每个 prompt 的分类结果
                for j, prob in enumerate(probs):
                    idx = batch_indices[j]
                    category, scale = route_from_probs(prob.unsqueeze(0), label_cols, thresh=args.route_thresh)
                    classification_results.append({
                        "index": idx,
                        "prompt": batch_prompts[j],
                        "category": category,
                        "scale": scale,
                        "probs": prob.numpy()
                    })
                    print(f"[{idx:03d}] prompt={batch_prompts[j][:40]}..., cat={category}, scale={scale:.3f}")
        
        # 释放分类器显存
        del cls_model, cls_tokenizer
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        print("✅ 分类完成，分类器显存已释放")
        print("=" * 60)
    
    # ========== 阶段2: 生成视频 ==========
    print("阶段2: 生成视频")
    print("=" * 60)
    
    # 1) raw pipeline (只在需要生成baseline时加载)
    pipe_raw = None
    if args.generate_baseline:
        pipe_raw, _ = load_pipeline(args.model_path, args.model_type, args.device, torch.float16)

    # 2) safe pipeline with multi adapters (只在需要生成防御视频时加载)
    pipe_safe = None
    if args.generate_defense:
        pipe_safe, _ = load_pipeline(args.model_path, args.model_type, args.device, torch.float16)
        # 2.1 注入多 adapter
        adapter_ckpt_map = args.adapter_ckpt_map  # dict[str,str]
        pipe_safe = inject_multi_safe_adapters(pipe_safe, adapter_ckpt_map, args.rank, args.hidden_size)

    # 按索引排序分类结果（如果有）
    if classification_results:
        classification_results.sort(key=lambda x: x["index"])
        classification_dict = {r["index"]: r for r in classification_results}

    for i, prompt in enumerate(prompts):
        # 生成baseline视频
        if args.generate_baseline:
            baseline_path = f"{args.output_dir}/multi_{i:03d}_raw.mp4"
            if os.path.exists(baseline_path) and args.skip_existing:
                print(f"[{i:03d}] Baseline视频已存在，跳过: {baseline_path}")
            else:
                print(f"[{i:03d}] 生成baseline视频: {prompt[:40]}...")
                gen = torch.Generator(device=args.device)
                if getattr(args, "seed", None) is not None:
                    gen.manual_seed(int(args.seed) + i)
                pipe_kwargs = get_pipeline_kwargs(
                    args.model_type, prompt, args.num_frames, args.height, args.width,
                    args.num_inference_steps, args.guidance_scale, gen
                )
                video_raw = pipe_raw(**pipe_kwargs).frames[0]
                export_to_video(video_raw, baseline_path, fps=args.fps)
                print(f"✅ Baseline视频已保存: {baseline_path}")

        # 生成防御视频
        if args.generate_defense:
            defense_path = f"{args.output_dir}/multi_{i:03d}_safe.mp4"
            if os.path.exists(defense_path) and args.skip_existing:
                print(f"[{i:03d}] 防御视频已存在，跳过: {defense_path}")
            else:
                # 从预分类结果中获取路由信息
                cls_result = classification_dict[i]
                category = cls_result["category"]
                scale = cls_result["scale"]

                # 设置当前路由
                pipe_safe.text_encoder.set_adapter_route(category=category, scale=scale)

                # 生成视频
                gen = torch.Generator(device=args.device)
                if getattr(args, "seed", None) is not None:
                    gen.manual_seed(int(args.seed) + i)
                pipe_kwargs = get_pipeline_kwargs(
                    args.model_type, prompt, args.num_frames, args.height, args.width,
                    args.num_inference_steps, args.guidance_scale, gen
                )
                video_safe = pipe_safe(**pipe_kwargs).frames[0]
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
                        default="/home/raykr/models/zai-org/CogVideoX-2b",
                        help="基础模型路径")
    parser.add_argument("--model_type", type=str, choices=["cogvideox", "wan"], default="cogvideox",
                        help="模型类型: cogvideox 或 wan")
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
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子（保证生成可复现）")
    
    # 路由相关参数（仅multi模式）
    parser.add_argument("--route_thresh", type=float, default=0.3,
                        help="路由阈值（低于此值不启用adapter，仅multi模式）")
    
    # 设备参数
    parser.add_argument("--device", type=str, default="cuda",
                        help="Pipeline 设备 (cuda/cpu)")
    parser.add_argument("--cls_device", type=str, default="cpu",
                        help="分类器设备 (默认 cpu 以节省显存，可设为 cuda/cuda:0/cuda:1 等)")
    parser.add_argument("--cls_batch_size", type=int, default=32,
                        help="分类器批量处理大小（用于批量分类所有 prompts，提高效率）")
    
    args = parser.parse_args()
    
    # 验证至少选择一种生成模式
    if not args.generate_baseline and not args.generate_defense:
        parser.error("必须至少选择 --generate_baseline 或 --generate_defense 之一")
    
    # 验证分类器路径（仅在生成防御视频时需要）
    if args.generate_defense and args.cls_ckpt_path is None:
        parser.error("生成防御视频需要 --cls_ckpt_path 参数")
    
    # 根据模式验证参数
    if not args.generate_baseline:
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
    print(f"  模型类型: {args.model_type}")
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
