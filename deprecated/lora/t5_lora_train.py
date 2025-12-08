import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, T5EncoderModel
from peft import LoraConfig, get_peft_model
import csv
from torch.optim import AdamW
from tqdm import tqdm
import time


# ===== 数据集 =====
class PromptPairDataset(Dataset):
    """
    每个样本包含 malicious, rewritten, benign prompt
    """
    def __init__(self, data, tokenizer, max_length=64):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        malicious, rewritten = self.data[idx]
        return {
            "malicious": malicious,
            "rewritten": rewritten
        }


# ===== Trainer =====
class LoraTextEncoderTrainer:
    def __init__(self, model_name, device="cuda", r=16, alpha=32, dropout=0.05, lr=1e-4):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, subfolder="tokenizer")
        self.text_encoder = T5EncoderModel.from_pretrained(model_name, subfolder="text_encoder")

        # 冻结原始参数
        self.text_encoder.requires_grad_(False)

        # 配置 LoRA
        lora_config = LoraConfig(
            r=r,
            lora_alpha=alpha,
            lora_dropout=dropout,
            bias="none",
            target_modules=["q", "v"],  # 注意 q_proj/v_proj 模块名称要和 T5 实现对上
            task_type="FEATURE_EXTRACTION",  # 对于 T5EncoderModel 使用 FEATURE_EXTRACTION
        )
        self.text_encoder = get_peft_model(self.text_encoder, lora_config)
        self.text_encoder.to(device)

        # 优化器
        self.optimizer = AdamW(self.text_encoder.parameters(), lr=lr)

    def encode(self, texts):
        token_ids = self.tokenizer(
            texts, padding=True, truncation=True, return_tensors="pt", max_length=64
        ).input_ids.to(self.device)
        # 直接传递 token_ids，让 PEFT 处理参数传递
        outputs = self.text_encoder(token_ids)
        return outputs[0].mean(dim=1)  # 取平均表示

    def training_step(self, batch):
        emb_mal = self.encode(batch["malicious"])
        emb_safe = self.encode(batch["rewritten"])

        # 主 loss: 对齐
        align_loss = F.mse_loss(emb_mal, emb_safe)

        loss = align_loss


        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return {"loss": loss.item(), "align_loss": align_loss.item()}

    def train(self, dataloader, num_steps=1000, log_interval=50, save_interval=100):
        print(f"🚀 开始训练 LoRA，对齐步数={num_steps}")
        total_loss, best_loss = 0.0, float("inf")
        pbar = tqdm(range(num_steps), desc="训练进度", unit="step")
        start_time = time.time()

        for step in pbar:
            batch = next(iter(dataloader))  # 简化，循环取 batch
            logs = self.training_step(batch)

            total_loss += logs["loss"]
            if logs["loss"] < best_loss:
                best_loss = logs["loss"]

            pbar.set_postfix({
                "Loss": f"{logs['loss']:.4f}",
                "Align": f"{logs['align_loss']:.4f}",
                "Best": f"{best_loss:.4f}",
            })

            if step % log_interval == 0:
                elapsed = time.time() - start_time
                avg_loss = total_loss / (step + 1)
                print(f"\n📈 Step {step:4d} | Loss={logs['loss']:.4f} (avg {avg_loss:.4f}) | Time {elapsed:.1f}s")

            if step % save_interval == 0 and step > 0:
                self.save_lora(f"checkpoint_step_{step}")

        print("🎉 训练完成")
        self.save_lora("final_lora")

    def save_lora(self, path):
        self.text_encoder.save_pretrained(path)
        print(f"✅ LoRA 权重已保存到 {path}")


# 从 CSV 读取: (malicious(prompt), rewritten(rewritten_prompt))
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

# 初始化
model_name = "/home/beihang/jzl/models/zai-org/CogVideoX-2b"
trainer = LoraTextEncoderTrainer(model_name=model_name, device="cuda")

# 数据加载
csv_path = "/home/beihang/jzl/projects/PromptSafe-T2V/datasets/train/1.csv"
malicious_list, rewritten_list = read_prompts_from_csv(csv_path)
data = list(zip(malicious_list, rewritten_list))

# 如果没有从CSV读取到数据，使用模拟数据作为备选
if not data:
    print("⚠️ CSV 数据为空，使用模拟数据")
    data = [
        ("a nude woman is running in the street",
         "a woman with red clothes is running in the street",),
    ]

dataset = PromptPairDataset(data, trainer.tokenizer)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# 训练
trainer.train(dataloader, num_steps=500)