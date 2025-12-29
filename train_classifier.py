import argparse
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, T5EncoderModel
from tqdm import tqdm
import os


class PromptClsDataset(Dataset):
    """
    用于 prompt 安全分类的数据集
    假定 csv 至少有一列 'prompt'，以及若干 0/1 标签列，例如:
      sexual, violent, hate, self_harm, ...
    """
    def __init__(self, csv_path: str, label_cols: list[str]):
        self.df = pd.read_csv(csv_path)
        self.prompts = self.df["prompt"].tolist()
        self.labels = self.df[label_cols].values.astype("float32")
        self.label_cols = label_cols
        print(f"[PromptClsDataset] Loaded {len(self.prompts)} samples, labels = {label_cols}")

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return {
            "prompt": self.prompts[idx],
            "label": torch.from_numpy(self.labels[idx])
        }


class PromptSafetyClassifier(nn.Module):
    """
    基于 T5 encoder 的多标签安全分类器：
      - 冻结 T5 参数，只训练顶层分类 head
      - 输出 logits: [B, num_labels]，后接 sigmoid 得到各类别风险概率
    """
    def __init__(self, t5_encoder: T5EncoderModel, hidden_size: int, num_labels: int):
        super().__init__()
        self.encoder = t5_encoder
        for p in self.encoder.parameters():
            p.requires_grad_(False)

        self.ln = nn.LayerNorm(hidden_size)
        self.head = nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            out = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state  # [B,L,D]
        pooled = out.mean(dim=1)   # [B,D]
        pooled = self.ln(pooled)
        logits = self.head(pooled)  # [B,num_labels]
        return logits


class PromptSafetyTrainer:
    def __init__(self, model_path, hidden_size, label_cols, lr=1e-4, device="cuda"):
        self.device = device
        self.label_cols = label_cols

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, subfolder="tokenizer")
        base = T5EncoderModel.from_pretrained(model_path, subfolder="text_encoder").to(device)

        self.model = PromptSafetyClassifier(
            t5_encoder=base,
            hidden_size=hidden_size,
            num_labels=len(label_cols),
        ).to(device)

        # 只训练分类 head 参数
        self.opt = torch.optim.AdamW(self.model.head.parameters(), lr=lr)
        self.criterion = nn.BCEWithLogitsLoss()

    def step(self, batch):
        texts = batch["prompt"]
        labels = batch["label"].to(self.device)  # [B,num_labels]

        tok = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(self.device)
        logits = self.model(tok["input_ids"], tok["attention_mask"])  # [B,num_labels]

        loss = self.criterion(logits, labels)

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        with torch.no_grad():
            probs = torch.sigmoid(logits)
        return {
            "loss": loss.item(),
            "probs": probs.detach().cpu(),
        }

    @torch.no_grad()
    def infer_probs(self, texts: list[str]) -> torch.Tensor:
        """
        推理接口：输入一组 prompt 文本，输出各类别风险概率 [B,num_labels]
        """
        self.model.eval()
        tok = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(self.device)
        logits = self.model(tok["input_ids"], tok["attention_mask"])
        probs = torch.sigmoid(logits)  # [B,num_labels]
        return probs


def train_prompt_classifier(args):
    """
    args.label_cols: list[str]，例如 ["sexual", "violent", "political", "disturbing"]
    args.cls_trainset_path: 分类器训练集 csv
    args.cls_ckpt_path: 分类器权重保存路径
    """
    trainer = PromptSafetyTrainer(
        model_path=args.model_path,
        hidden_size=args.hidden_size,
        label_cols=args.label_cols,
        lr=args.cls_lr,
        device=args.device,
    )
    dataset = PromptClsDataset(args.cls_trainset_path, args.label_cols)
    loader = DataLoader(dataset, batch_size=args.cls_batch_size, shuffle=True)

    for epoch in range(args.cls_num_epochs):
        pbar = tqdm(loader, desc=f"[CLS] Epoch {epoch+1}/{args.cls_num_epochs}", dynamic_ncols=True, leave=False)
        for batch in pbar:
            logs = trainer.step(batch)
            pbar.set_postfix({"loss": f"{logs['loss']:.3f}"})

        if hasattr(args, "cls_save_every") and args.cls_save_every and (epoch + 1) % args.cls_save_every == 0:
            os.makedirs(os.path.dirname(args.cls_ckpt_path), exist_ok=True)
            ckpt_path = args.cls_ckpt_path.replace(".pt", f"_epoch{epoch+1}.pt")
            torch.save({
                "state_dict": trainer.model.state_dict(),
                "label_cols": args.label_cols,
            }, ckpt_path)
            print(f"✅ 分类器周期性保存: {ckpt_path}")

    os.makedirs(os.path.dirname(args.cls_ckpt_path), exist_ok=True)
    torch.save({
        "state_dict": trainer.model.state_dict(),
        "label_cols": args.label_cols,
    }, args.cls_ckpt_path)
    print(f"✅ PromptSafetyClassifier 已保存到 {args.cls_ckpt_path}")


if __name__ == "__main__":
    cfg = {
        "model_path": "/home/raykr/models/Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        "hidden_size": 4096,
        "label_cols": ["sexual", "violent", "political", "disturbing"],
        "cls_trainset_path": "datasets/train/classification.csv",
        "cls_ckpt_path": "checkpoints/wan2.1-t2v-1.3b-diffusers/classifier/prompt_classifier.pt",
        "cls_lr": 1e-4,
        "cls_batch_size": 2,
        "cls_num_epochs": 10,
        "cls_save_every": 5,
        "device": "cuda:0",
    }
    args = argparse.Namespace(**cfg)
    train_prompt_classifier(args)