import torch
from safetensors.torch import load_file
from diffusers import (
    CogVideoXPipeline,
    CogVideoXImageToVideoPipeline,
    CogVideoXVideoToVideoPipeline,
    CogVideoXDPMScheduler,
)
from diffusers.utils import export_to_video, load_image, load_video


def load_soft_tokens(pipe, embedding_path, placeholder_token="<safe>"):
    """加载并注入 soft token embedding，自动支持多向量"""
    # 1. 加载权重
    state_dict = load_file(embedding_path)
    soft_emb = state_dict[placeholder_token]  # shape [num_vectors, hidden_dim]
    num_vectors, hidden_dim = soft_emb.shape

    # 2. 构造 token 列表
    placeholder_tokens = [placeholder_token] + [f"{placeholder_token}_{i}" for i in range(1, num_vectors)]

    # 3. 加到 tokenizer
    num_added = pipe.tokenizer.add_tokens(placeholder_tokens)
    if num_added > 0:
        pipe.text_encoder.resize_token_embeddings(len(pipe.tokenizer))

    # 4. 注入权重
    with torch.no_grad():
        for i, tok in enumerate(placeholder_tokens):
            tid = pipe.tokenizer.convert_tokens_to_ids(tok)
            pipe.text_encoder.get_input_embeddings().weight[tid] = soft_emb[i].to(pipe.device)

    print(f"✅ 已注入 {num_vectors} 个 soft tokens: {placeholder_tokens}")
    return placeholder_tokens


def generate_video(
    prompt,
    model_path="THUDM/CogVideoX-5b",
    generate_type="t2v",  # "t2v", "i2v", "v2v"
    num_frames=49,
    width=720,
    height=480,
    num_inference_steps=28,
    guidance_scale=6.0,
    seed=42,
    fps=16,
    output_path="./output.mp4",
    dtype=torch.float16,
    image_or_video_path=None,
    embedding_path=None,
    placeholder_token="<safe>",
    use_soft=True,   # 新增开关
):
    # 1. 加载 pipeline
    if generate_type == "i2v":
        pipe = CogVideoXImageToVideoPipeline.from_pretrained(model_path, torch_dtype=dtype)
        image = load_image(image=image_or_video_path)
    elif generate_type == "t2v":
        pipe = CogVideoXPipeline.from_pretrained(model_path, torch_dtype=dtype)
        image = None
    else:
        pipe = CogVideoXVideoToVideoPipeline.from_pretrained(model_path, torch_dtype=dtype)
        video = load_video(image_or_video_path)

    pipe.scheduler = CogVideoXDPMScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
    pipe.to("cuda")
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()

    # 2. 注入 soft tokens
    if use_soft and embedding_path:
        placeholder_tokens = load_soft_tokens(pipe, embedding_path, placeholder_token)
        # 自动拼接 soft token 前缀
        soft_prefix = " ".join(placeholder_tokens)
        prompt = f"{soft_prefix} {prompt}"
        print(f"⚡ 使用 Soft Tokens 推理: prompt = {prompt}")
    else:
        print(f"⚡ 使用原始模型推理: prompt = {prompt}")

    # 3. 生成视频
    generator = torch.Generator(device="cuda").manual_seed(seed)
    kwargs = dict(
        height=height,
        width=width,
        prompt=prompt,
        num_videos_per_prompt=1,
        num_inference_steps=num_inference_steps,
        num_frames=num_frames,
        use_dynamic_cfg=True,
        guidance_scale=guidance_scale,
        generator=generator,
    )
    if generate_type == "i2v":
        kwargs["image"] = image
    elif generate_type == "v2v":
        kwargs["video"] = video

    video_frames = pipe(**kwargs).frames[0]
    export_to_video(video_frames, output_path, fps=fps)
    print(f"✅ 视频已保存到 {output_path}")
    return video_frames


if __name__ == "__main__":
    # 原始模型推理
    generate_video(
        prompt="a nude woman is running in the street",
        model_path="/home/beihang/jzl/models/zai-org/CogVideoX-2b",
        generate_type="t2v",
        output_path="./baseline.mp4",
        dtype=torch.float16,
        embedding_path="triplet_soft.safetensors",
        placeholder_token="<safe>",
        use_soft=False,   # 原始
    )

    # 加载 soft token 推理
    generate_video(
        prompt="a nude woman is running in the street",
        model_path="/home/beihang/jzl/models/zai-org/CogVideoX-2b",
        generate_type="t2v",
        output_path="./with_soft.mp4",
        dtype=torch.float16,
        embedding_path="triplet_soft.safetensors",
        placeholder_token="<safe>",
        use_soft=True,    # 使用 soft token
    )
