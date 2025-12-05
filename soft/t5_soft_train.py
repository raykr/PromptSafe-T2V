import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, T5EncoderModel
from torch.optim import AdamW
from tqdm import tqdm
import time
import csv
from safetensors.torch import save_file
import random


# ===== 工具函数 =====
def read_prompts_from_csv(csv_path):
    malicious_list, rewritten_list = [] , []
    with open(csv_path, "r", encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f)
        for row in reader:
            mal = (row.get("prompt") or "").strip()
            rew = (row.get("rewritten_prompt") or "").strip()
            if mal and rew:
                malicious_list.append(mal)
                rewritten_list.append(rew)
    return malicious_list, rewritten_list


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
                 num_vectors=4, device="cuda", lr=5e-4, margin=0.1, max_length=128):
        self.device = device
        self.num_vectors = num_vectors
        self.placeholder_token = placeholder_token
        self.initializer_token = initializer_token
        self.margin = margin
        self.max_length = max_length

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
        # 仅用于加速训练的编码缓存（与 soft token 无关的 positive/negative）
        self._cache_positive = None
        self._cache_negative = None
        self._text_to_index_malicious = None
        self._text_to_index_rewritten = None

    def _encode_list(self, texts, batch_size=256):
        """将一组文本分批编码为 [N, D]，不保留梯度，用于缓存。"""
        self.text_encoder.eval()
        outputs = []
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i+batch_size]
                emb = self.encode(batch)  # encode 已含 attention_mask 与 nan 防护
                outputs.append(emb.detach())
        if outputs:
            cached = torch.cat(outputs, dim=0)
        else:
            cached = torch.empty(0, device=self.device)
        return cached

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
        encoded_inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        token_ids = encoded_inputs.input_ids.to(self.device)
        attention_mask = encoded_inputs.attention_mask.to(self.device)

        base_embedding_layer = self.text_encoder.get_input_embeddings()
        inputs_embeds = base_embedding_layer(token_ids)
        for i, tid in enumerate(self.placeholder_token_ids):
            mask = (token_ids == tid).unsqueeze(-1)
            replacement = self.soft_token_embedding[i].unsqueeze(0).unsqueeze(0)
            inputs_embeds = torch.where(mask, replacement, inputs_embeds)

        outputs = self.text_encoder.encoder(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        encoded = outputs.last_hidden_state.mean(dim=1)
        encoded = torch.nan_to_num(encoded)
        return encoded

    def encode(self, texts):
        encoded_inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        input_ids = encoded_inputs.input_ids.to(self.device)
        attention_mask = encoded_inputs.attention_mask.to(self.device)

        outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)[0]
        encoded = outputs.mean(dim=1)
        encoded = torch.nan_to_num(encoded)
        return encoded

    def training_step(self, malicious, rewritten, benign=None, lambda_benign=0.1):
        # Anchor: <safe> malicious
        anchor = self._encode_with_soft_token([f"{self.placeholder_token} {t}" for t in malicious])
        # Positive: rewritten
        if self._cache_positive is not None and self._text_to_index_rewritten is not None:
            idx = torch.tensor([self._text_to_index_rewritten[t] for t in rewritten], device=self.device, dtype=torch.long)
            positive = self._cache_positive.index_select(0, idx)
        else:
            with torch.no_grad():
                positive = self.encode(rewritten)
        # Negative: malicious
        if self._cache_negative is not None and self._text_to_index_malicious is not None:
            idx = torch.tensor([self._text_to_index_malicious[t] for t in malicious], device=self.device, dtype=torch.long)
            negative = self._cache_negative.index_select(0, idx)
        else:
            with torch.no_grad():
                negative = self.encode(malicious)

        # Triplet Loss
        anchor = torch.nan_to_num(anchor)
        positive = torch.nan_to_num(positive)
        negative = torch.nan_to_num(negative)
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
        # 梯度裁剪，避免梯度爆炸
        try:
            torch.nn.utils.clip_grad_norm_([self.soft_token_embedding], max_norm=1.0)
        except Exception:
            pass
        self.optimizer.step()

        return {
            "loss": loss.item(),
            "triplet_loss": triplet_loss.item(),
            "benign_loss": benign_loss.item(),
            "d_ap": d_ap.mean().item(),
            "d_an": d_an.mean().item(),
        }

    def train(self, malicious, rewritten, benign=None, lambda_benign=0.1, num_steps=1000, log_interval=50, batch_size=32):
        print(f"🚀 开始训练 Triplet Soft Token，对齐步数={num_steps}")
        best_loss = float("inf")
        pbar = tqdm(range(num_steps), desc="训练进度", unit="step")
        start_time = time.time()

        # 预计算并缓存 rewritten/malicious 的编码（它们与 soft token 无关），显著减少每步计算量
        self._text_to_index_malicious = {t: i for i, t in enumerate(malicious)}
        self._text_to_index_rewritten = {t: i for i, t in enumerate(rewritten)}
        if len(malicious) > 0:
            self._cache_negative = self._encode_list(malicious, batch_size=256)
        if len(rewritten) > 0:
            self._cache_positive = self._encode_list(rewritten, batch_size=256)

        for step in pbar:
            # 随机采样一个小批次，避免一次性将全部样本放入显存
            data_size = min(len(malicious), len(rewritten))
            if data_size == 0:
                raise ValueError("空数据集：请检查 CSV 的 prompt 与 rewritten_prompt 列是否为空")
            batch_indices = random.sample(range(data_size), k=min(batch_size, data_size))
            mal_batch = [malicious[i] for i in batch_indices]
            rew_batch = [rewritten[i] for i in batch_indices]
            logs = self.training_step(mal_batch, rew_batch, None, lambda_benign)
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
    num_vectors=1,  # 建议用 8 个向量
    device="cuda",
    margin=0.1
)

# 从 CSV 读取 malicious(prompt) 与 rewritten(rewritten_prompt)
csv_path = "datasets/train/1.csv"
malicious, rewritten = read_prompts_from_csv(csv_path)
benign = ["a man is running in the park"]

trainer.train(malicious, rewritten, None, lambda_benign=0.1, num_steps=2000, batch_size=32)
trainer.save_soft_embedding("triplet_soft.safetensors")

