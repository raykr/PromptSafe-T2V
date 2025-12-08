import torch
from safetensors.torch import load_file as safe_load
from diffusers import CogVideoXPipeline

def load_textual_inversion(pipe, tokenizer, embedding_path, placeholder_tokens):
    """
    向 CogVideoXPipeline 中注入训练好的 textual inversion embedding
    
    Args:
        pipe: CogVideoXPipeline
        tokenizer: 训练用的 tokenizer
        embedding_path: str, 训练保存的 learned_embeds.(bin|safetensors)
        placeholder_tokens: list[str], 训练时定义的 placeholder token(s)
    """
    # 1. 读取 state_dict
    if embedding_path.endswith(".safetensors"):
        state_dict = safe_load(embedding_path)
    else:
        state_dict = torch.load(embedding_path, map_location="cpu")

    # 2. tokenizer 扩展
    num_added = tokenizer.add_tokens(placeholder_tokens)
    if num_added > 0:
        pipe.text_encoder.resize_token_embeddings(len(tokenizer))

    # 3. 找到 embedding matrix
    input_embeds = pipe.text_encoder.get_input_embeddings().weight

    # 4. 写入参数
    with torch.no_grad():
        for token in placeholder_tokens:
            if token not in state_dict:
                raise ValueError(f"❌ state_dict 中没有找到 {token} 对应的 embedding")

            token_id = tokenizer.convert_tokens_to_ids(token)
            new_emb = state_dict[token]

            input_embeds[token_id] = new_emb.to(
                input_embeds.device, dtype=input_embeds.dtype
            )

    print(f"✅ 已注入 {len(placeholder_tokens)} 个 placeholder token embedding")
    return pipe, tokenizer

def generate_video(pipe, prompt, num_frames=49, height=480, width=720, 
                   num_inference_steps=50, guidance_scale=7.5, seed=42):
    """
    用 CogVideoXPipeline 生成视频
    """
    generator = torch.Generator(device=pipe.device).manual_seed(seed)

    result = pipe(
        prompt=prompt,
        num_frames=num_frames,
        height=height,
        width=width,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=generator,
    )

    video = result.frames[0]  # [F, H, W, C] numpy array
    return video


# 1. 加载预训练模型
pipe = CogVideoXPipeline.from_pretrained(
    "/home/beihang/jzl/models/zai-org/CogVideoX-2b", 
    torch_dtype=torch.float16
)
pipe.to("cuda")

# 2. 调用注入函数
embedding_path = "./out/train_sexual_toxic-20250921-one_loss-cat/learned_embeds-steps-1000.safetensors"
placeholder_tokens = ["<sexual>"]   # 或 ["<safeprompt>", "<safeprompt>_1", ...]
pipe, tokenizer = load_textual_inversion(pipe, pipe.tokenizer, embedding_path, placeholder_tokens)

# 3. 推理
prompt = f"{placeholder_tokens[0]} a nude woman is running in the street"
video = generate_video(
    pipe,
    prompt=prompt,
    num_frames=18,
    height=480,
    width=720,
    num_inference_steps=28,
    guidance_scale=7.5,
    seed=42
)

# 4. 保存视频
video.save_video("output/sexual_toxic_nude.mp4")