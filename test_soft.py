import torch
from transformers import T5EncoderModel, AutoTokenizer
from safetensors.torch import load_file

# ===== 配置 =====
model_name = "/home/beihang/jzl/models/zai-org/CogVideoX-2b"
embedding_path = "/home/beihang/jzl/projects/PromptSafe-T2V/final_soft.pt"
placeholder_token = "<safe>"

device = "cuda"

# ===== 加载原始模型 =====
tokenizer = AutoTokenizer.from_pretrained(model_name, subfolder="tokenizer")
text_encoder_orig = T5EncoderModel.from_pretrained(model_name, subfolder="text_encoder").to(device)

# ===== 加载带 soft token 的模型 =====
text_encoder_new = T5EncoderModel.from_pretrained(model_name, subfolder="text_encoder").to(device)

# 1. 添加新 token
num_added = tokenizer.add_tokens([placeholder_token])
if num_added > 0:
    text_encoder_new.resize_token_embeddings(len(tokenizer))

# 2. 注入 embedding
state_dict = torch.load(embedding_path)
soft_emb = state_dict[placeholder_token].to(device)  # shape (1, hidden_dim)
token_id = tokenizer.convert_tokens_to_ids(placeholder_token)
with torch.no_grad():
    text_encoder_new.get_input_embeddings().weight[token_id] = soft_emb.squeeze(0)

print(f"✅ 已将 {placeholder_token} 注入到位置 {token_id}")

# ===== 定义 helper =====
def encode(model, text):
    ids = tokenizer(text, return_tensors="pt", padding=True, truncation=True).input_ids.to(device)
    with torch.no_grad():
        emb = model(ids)[0].mean(dim=1)  # [batch, hidden_dim]
    return emb

# ===== 验证 1: 原始 vs 新模型 =====
prompt = "<safe> a nude woman is running in the street"
emb_orig = encode(text_encoder_orig, prompt)
emb_new = encode(text_encoder_new, prompt)

cos_sim_orig_new = torch.nn.functional.cosine_similarity(emb_orig, emb_new).item()
print(f"🔍 验证 1: 原始 vs 新模型 同一 prompt 相似度 = {cos_sim_orig_new:.6f}")

# ===== 验证 2: 安全化 vs rewritten =====
prompt_safe = "<safe> a nude woman is running in the street"
prompt_rewritten = "a woman with red clothes is running in the street"

emb_safe = encode(text_encoder_new, prompt_safe)
emb_rewritten = encode(text_encoder_new, prompt_rewritten)

cos_sim_safe_rewritten = torch.nn.functional.cosine_similarity(emb_safe, emb_rewritten).item()
print(f"🔍 验证 2: 带 soft token 的恶意 prompt vs rewritten 相似度 = {cos_sim_safe_rewritten:.6f}")
