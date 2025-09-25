import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, T5EncoderModel
from torch.optim import AdamW
from tqdm import tqdm
import time
from safetensors.torch import save_file


# ===== EmbeddingWrapper =====
class EmbeddingWrapper(nn.Module):
    """
    替换 T5 的 embedding 层：
    - 冻结原始 embedding
    - 特殊 soft token 用 nn.Parameter 训练
    """
    def __init__(self, base_embedding: nn.Embedding, placeholder_ids, initializer_id):
        super().__init__()
        self.base_embedding = base_embedding
        self.base_embedding.weight.requires_grad_(False)  # 冻结原始 embedding

        self.placeholder_ids = placeholder_ids
        self.initializer_id = initializer_id

        # 初始化 soft embedding = initializer 的 embedding 拷贝
        init_embeds = base_embedding.weight[initializer_id].detach().clone().unsqueeze(0).repeat(len(placeholder_ids), 1)
        self.soft_embeddings = nn.Parameter(init_embeds)  # 可训练参数

    def forward(self, input_ids):
        # 基础 embedding
        embeds = self.base_embedding(input_ids)

        # 替换 soft token 对应位置
        for i, token_id in enumerate(self.placeholder_ids):
            mask = (input_ids == token_id).unsqueeze(-1)  # [B, L, 1]
            embeds = torch.where(mask, self.soft_embeddings[i], embeds)
        return embeds


# ===== 数据集 =====
class PromptPairDataset(Dataset):
    """
    每个样本包含 malicious, rewritten prompt
    """
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        malicious, rewritten = self.data[idx]
        return {
            "malicious": malicious,
            "rewritten": rewritten
        }


# ===== Trainer =====
class SoftTokenTextEncoderTrainer:
    def __init__(self, model_name, placeholder_token="<safe>", initializer_token="safe",
                 num_vectors=16, device="cuda", lr=5e-4):
        self.device = device
        self.num_vectors = num_vectors
        self.placeholder_token = placeholder_token
        self.initializer_token = initializer_token

        # 1. 加载 tokenizer 和 encoder
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, subfolder="tokenizer")
        self.text_encoder = T5EncoderModel.from_pretrained(model_name, subfolder="text_encoder").to(self.device)
        self.text_encoder.requires_grad_(False)  # 冻结 backbone

        # 2. 添加 placeholder token
        placeholder_tokens = [self.placeholder_token]
        for i in range(1, self.num_vectors):
            placeholder_tokens.append(f"{self.placeholder_token}_{i}")
        num_added = self.tokenizer.add_tokens(placeholder_tokens)
        if num_added != self.num_vectors:
            raise ValueError("token 已经存在，请换一个名字")

        # id 获取
        self.placeholder_ids = self.tokenizer.convert_tokens_to_ids(placeholder_tokens)
        init_ids = self.tokenizer.encode(self.initializer_token, add_special_tokens=False)
        if len(init_ids) != 1:
            raise ValueError("initializer_token 必须是单 token")
        self.initializer_id = init_ids[0]

        # 3. 替换 embedding 层
        self.text_encoder.resize_token_embeddings(len(self.tokenizer))
        emb_layer = self.text_encoder.get_input_embeddings()
        wrapper = EmbeddingWrapper(emb_layer, self.placeholder_ids, self.initializer_id)
        self.text_encoder.set_input_embeddings(wrapper)

        # 4. 优化器只更新 soft_embeddings
        self.optimizer = AdamW([wrapper.soft_embeddings], lr=lr)

    def encode(self, texts):
        token_ids = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt").input_ids.to(self.device)
        inputs_embeds = self.text_encoder.get_input_embeddings()(token_ids)
        outputs = self.text_encoder.encoder(inputs_embeds=inputs_embeds)
        return outputs.last_hidden_state.mean(dim=1)

    def training_step(self, batch):
        malicious = [f"{self.placeholder_token} {t}" for t in batch["malicious"]]
        rewritten = batch["rewritten"]

        emb_mal_safe = self.encode(malicious)
        emb_rewritten = self.encode(rewritten)

        loss = F.mse_loss(emb_mal_safe, emb_rewritten)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {"loss": loss.item()}

    def train(self, dataloader, num_steps=500, log_interval=50, save_interval=100):
        print(f"🚀 开始训练 Soft Token，对齐步数={num_steps}")
        total_loss, best_loss = 0.0, float("inf")
        pbar = tqdm(range(num_steps), desc="训练进度", unit="step")
        start_time = time.time()

        for step in pbar:
            batch = next(iter(dataloader))
            logs = self.training_step(batch)

            total_loss += logs["loss"]
            best_loss = min(best_loss, logs["loss"])

            pbar.set_postfix({"Loss": f"{logs['loss']:.4f}", "Best": f"{best_loss:.4f}"})

            if step % log_interval == 0:
                elapsed = time.time() - start_time
                avg_loss = total_loss / (step + 1)
                print(f"\n📈 Step {step:4d} | Loss={logs['loss']:.4f} (avg {avg_loss:.4f}) | Time {elapsed:.1f}s")

            if step % save_interval == 0 and step > 0:
                self.save_soft(f"checkpoint_step_{step}.pt")

        print("🎉 训练完成")
        self.save_soft("final_soft.pt")

    def save_soft(self, path):
        state_dict = {self.placeholder_token: self.text_encoder.get_input_embeddings().soft_embeddings.detach().cpu()}
        torch.save(state_dict, path)
        print(f"✅ Soft token 已保存到 {path}")


class TripletSoftTokenTrainer:
    def __init__(self, model_name, placeholder_token="<safe>", initializer_token="safe", 
                 num_vectors=4, device="cuda", lr=5e-4, margin=0.1):
        self.device = device
        self.num_vectors = num_vectors
        self.placeholder_token = placeholder_token
        self.initializer_token = initializer_token
        self.margin = margin

        # 加载 T5Encoder 和 tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, subfolder="tokenizer")
        self.text_encoder = T5EncoderModel.from_pretrained(model_name, subfolder="text_encoder").to(device)

        # 设置 placeholder tokens
        self.setup_placeholder_tokens()

        # 冻结模型参数
        self.text_encoder.requires_grad_(False)

        # 可训练 soft token embedding
        self.soft_token_embedding = (
            self.text_encoder.get_input_embeddings().weight[self.placeholder_token_ids]
            .clone().detach().requires_grad_(True)
        )
        self.optimizer = torch.optim.AdamW([self.soft_token_embedding], lr=lr)

    def setup_placeholder_tokens(self):
        placeholder_tokens = [self.placeholder_token]
        for i in range(1, self.num_vectors):
            placeholder_tokens.append(f"{self.placeholder_token}_{i}")

        num_added = self.tokenizer.add_tokens(placeholder_tokens)
        if num_added != self.num_vectors:
            raise ValueError("Token 已存在或数量不对")

        token_ids = self.tokenizer.encode(self.initializer_token, add_special_tokens=False)
        if len(token_ids) > 1:
            raise ValueError("Initializer token 必须是单个 token")

        self.initializer_token_id = token_ids[0]
        self.placeholder_token_ids = self.tokenizer.convert_tokens_to_ids(placeholder_tokens)

        # resize embedding
        self.text_encoder.resize_token_embeddings(len(self.tokenizer))
        token_embeds = self.text_encoder.get_input_embeddings().weight.data
        with torch.no_grad():
            for tid in self.placeholder_token_ids:
                token_embeds[tid] = token_embeds[self.initializer_token_id].clone()

    def _encode_with_soft_token(self, texts):
        token_ids = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt").input_ids.to(self.device)

        embedding_matrix = self.text_encoder.get_input_embeddings().weight
        custom_embeddings = embedding_matrix.clone()
        custom_embeddings[self.placeholder_token_ids] = self.soft_token_embedding

        inputs_embeds = F.embedding(token_ids, custom_embeddings)

        outputs = self.text_encoder.encoder(inputs_embeds=inputs_embeds)
        return outputs.last_hidden_state.mean(dim=1)

    def encode(self, texts):
        token_ids = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt").input_ids.to(self.device)
        outputs = self.text_encoder(token_ids)[0]
        return outputs.mean(dim=1)

    def training_step(self, malicious, rewritten, benign=None, lambda_benign=0.1):
        # Anchor: <safe> malicious
        anchor = self._encode_with_soft_token([f"{self.placeholder_token} {t}" for t in malicious])
        # Positive: rewritten
        positive = self.encode(rewritten)
        # Negative: malicious
        negative = self.encode(malicious)

        # Triplet Loss
        d_ap = F.pairwise_distance(anchor, positive)
        d_an = F.pairwise_distance(anchor, negative)
        triplet_loss = torch.relu(d_ap - d_an + self.margin).mean()

        # Benign consistency
        if benign is not None:
            benign_orig = self.encode(benign)
            benign_safe = self._encode_with_soft_token([f"{self.placeholder_token} {t}" for t in benign])
            benign_loss = F.mse_loss(benign_orig, benign_safe)
        else:
            benign_loss = torch.tensor(0.0, device=self.device)

        loss = triplet_loss + lambda_benign * benign_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {
            "loss": loss.item(),
            "triplet_loss": triplet_loss.item(),
            "benign_loss": benign_loss.item(),
            "d_ap": d_ap.mean().item(),
            "d_an": d_an.mean().item(),
        }

    def train(self, malicious, rewritten, benign=None, lambda_benign=0.1, num_steps=1000, log_interval=50):
        print(f"🚀 开始训练 Triplet Soft Token，对齐步数={num_steps}")
        best_loss = float("inf")
        pbar = tqdm(range(num_steps), desc="训练进度", unit="step")
        start_time = time.time()

        for step in pbar:
            logs = self.training_step(malicious, rewritten, benign, lambda_benign)
            if logs["loss"] < best_loss:
                best_loss = logs["loss"]

            pbar.set_postfix({
                "Loss": f"{logs['loss']:.4f}",
                "Triplet": f"{logs['triplet_loss']:.4f}",
                "Benign": f"{logs['benign_loss']:.4f}",
                "d_ap": f"{logs['d_ap']:.3f}",
                "d_an": f"{logs['d_an']:.3f}",
            })

            if step % log_interval == 0:
                elapsed = time.time() - start_time
                print(f"\n📈 Step {step:4d} | Loss={logs['loss']:.4f} | Triplet={logs['triplet_loss']:.4f} "
                      f"| d_ap={logs['d_ap']:.3f} | d_an={logs['d_an']:.3f} | Time={elapsed:.1f}s")

        print("🎉 训练完成")

    def save_soft_embedding(self, path="learned_triplet.safetensors"):
        state_dict = {self.placeholder_token: self.soft_token_embedding.detach().cpu()}
        save_file(state_dict, path)
        print(f"✅ Soft token embedding saved to {path}")




# 模拟数据: (malicious, rewritten)
# data = [
#     ("a nude woman is running in the street",
#      "a woman with red clothes is running in the street"),
# ]

# # 初始化
# model_name = "/home/beihang/jzl/models/zai-org/CogVideoX-2b"
# trainer = SoftTokenTextEncoderTrainer(model_name=model_name, placeholder_token="<safe>", device="cuda")

# # 数据加载
# dataset = PromptPairDataset(data)
# dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# # 训练
# trainer.train(dataloader, num_steps=500)


trainer = TripletSoftTokenTrainer(
    model_name="/home/beihang/jzl/models/zai-org/CogVideoX-2b",
    placeholder_token="<safe>",
    initializer_token="safe",
    num_vectors=8,  # 建议用 8 个向量
    device="cuda",
    margin=0.5
)

malicious = ["a nude woman is running in the street"]
rewritten = ["a woman with red clothes is running in the street"]
benign = ["a man is running in the park"]

trainer.train(malicious, rewritten, None, lambda_benign=0.1, num_steps=2000)
trainer.save_soft_embedding("triplet_soft.safetensors")

