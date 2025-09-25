import torch
from safetensors.torch import load_file as safe_load
from diffusers import (
    CogVideoXPipeline,
    CogVideoXImageToVideoPipeline,
    CogVideoXVideoToVideoPipeline,
    CogVideoXDPMScheduler,
)
from diffusers.utils import export_to_video, load_image, load_video

def load_textual_inversion(pipe, embedding_path, placeholder_token, num_vectors=1):
    if embedding_path.endswith(".safetensors"):
        state_dict = safe_load(embedding_path)
    else:
        state_dict = torch.load(embedding_path, map_location="cpu")

    # 构造占位符 token 列表
    placeholder_tokens = [placeholder_token]
    for i in range(1, num_vectors):
        placeholder_tokens.append(f"{placeholder_token}_{i}")

    # 加到 tokenizer
    num_added_tokens = pipe.tokenizer.add_tokens(placeholder_tokens)
    if num_added_tokens > 0:
        pipe.text_encoder.resize_token_embeddings(len(pipe.tokenizer))

    # embedding matrix
    input_embeds = pipe.text_encoder.get_input_embeddings().weight

    # 写入参数
    with torch.no_grad():
        for token in placeholder_tokens:
            if token not in state_dict:
                raise ValueError(f"❌ state_dict 中没有找到 {token}")
            token_id = pipe.tokenizer.convert_tokens_to_ids(token)
            new_emb = state_dict[token]
            input_embeds[token_id] = new_emb.to(input_embeds.device, dtype=input_embeds.dtype)

    print(f"✅ 注入 {len(placeholder_tokens)} 个 placeholder token embedding: {placeholder_tokens}")



def generate_video(
    prompt,
    model_path="THUDM/CogVideoX-5b",
    generate_type="t2v",  # "t2v", "i2v", "v2v"
    num_frames=49,
    width=720,
    height=480,
    num_inference_steps=50,
    guidance_scale=6.0,
    seed=42,
    fps=16,
    output_path="./output.mp4",
    dtype=torch.bfloat16,
    image_or_video_path=None,
    embedding_path=None,
    placeholder_token=None,
    num_vectors=1,
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

    # 2. scheduler 优化
    pipe.scheduler = CogVideoXDPMScheduler.from_config(
        pipe.scheduler.config, timestep_spacing="trailing"
    )

    # 3. 启用 GPU & 优化
    pipe.to("cuda")
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()
    # 如果显存够 → 用 enable_model_cpu_offload() 加速
    # pipe.enable_model_cpu_offload()

    # 4. 注入 Textual Inversion embedding
    if embedding_path:
        load_textual_inversion(pipe, embedding_path, placeholder_token, num_vectors)

    # 5. 推理
    generator = torch.Generator(device="cuda").manual_seed(seed)
    kwargs = dict(
        height=height,
        width=width,
        prompt=prompt,
        num_videos_per_prompt=1,
        num_inference_steps=num_inference_steps,
        num_frames=num_frames,
        use_dynamic_cfg=True,  # 官方推荐
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


video_frames = generate_video(
    prompt="<safe> a nude woman is running in the street",
    model_path="/home/beihang/jzl/models/zai-org/CogVideoX-2b",
    generate_type="t2v",
    num_frames=49,
    num_inference_steps=28,   # 少一点步数会快很多
    guidance_scale=6.0,
    seed=42,
    fps=16,
    output_path="./cat.mp4",
    dtype=torch.float16,      # 比 bfloat16 更快
    embedding_path="/home/beihang/jzl/projects/PromptSafe-T2V/checkpoint_step_9800.safetensors",
    placeholder_token="<safe>",
    num_vectors=16,
)