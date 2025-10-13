import torch
import torch.nn.functional as F
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, T5EncoderModel
from model_utils import SafeAdapter, WrappedTextEncoder


class SafeAdapterTrainer:
    def __init__(self, model_path, lr=1e-4, rank=256, hidden_size=4096, device="cuda"):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, subfolder="tokenizer")
        self.t5 = T5EncoderModel.from_pretrained(model_path, subfolder="text_encoder").to(device)
        self.adapter = SafeAdapter(hidden_size, rank).to(device)
        self.optimizer = torch.optim.AdamW(self.adapter.parameters(), lr=lr)

    def encode(self, texts):
        tokens = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(self.device)
        return self.t5(**tokens).last_hidden_state.mean(dim=1)

    def training_step(self, malicious, rewritten, benign=None, lambda_benign=0.1):
        emb_mal = self.encode(malicious)
        emb_safe = self.encode(rewritten)

        # 对齐loss
        align_loss = F.mse_loss(self.adapter(emb_mal), emb_safe)

        if benign is not None:
            emb_ben = self.encode(benign)
            benign_loss = F.mse_loss(self.adapter(emb_ben), emb_ben)
        else:
            benign_loss = torch.tensor(0.0, device=self.device)

        loss = align_loss + lambda_benign * benign_loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {"loss": loss.item(), "align_loss": align_loss.item(), "benign_loss": benign_loss.item()}

    def train(self, csv_path, num_epochs=5, batch_size=2, lambda_benign=0.1, save_path="checkpoints/safe_adapter.pt"):
        data = pd.read_csv(csv_path)
        total_steps = num_epochs * len(data)
        print(f"🚀 开始训练 SafeAdapter，共 {total_steps} steps")

        for epoch in range(num_epochs):
            pbar = tqdm(data.iterrows(), total=len(data), desc=f"Epoch {epoch+1}")
            for _, row in pbar:
                logs = self.training_step(
                    [row["malicious"]],
                    [row["rewritten"]],
                    [row["benign"]],
                    lambda_benign
                )
                pbar.set_postfix(logs)
        torch.save(self.adapter.state_dict(), save_path)
        print(f"✅ SafeAdapter 已保存到 {save_path}")


if __name__ == "__main__":
    trainer = SafeAdapterTrainer(
        model_path="./models/CogVideoX-2b",
        lr=1e-4,
        rank=256,
        hidden_size=4096,
        device="cuda"
    )
    trainer.train(csv_path="./data/safety_pairs.csv", num_epochs=10, save_path="./adapters/safe_adapter.pt")
