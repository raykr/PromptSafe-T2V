import argparse
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, T5EncoderModel
from models import SafeAdapter
from models import WrappedTextEncoder
from datasets import PairDataset
from tqdm import tqdm
import os
from torch.utils.data import DataLoader

class SafeAdapterTrainer:
    def __init__(self, model_path, hidden_size=4096, rank=256, lr=5e-4, device="cuda", use_benign=False):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, subfolder="tokenizer")
        base = T5EncoderModel.from_pretrained(model_path, subfolder="text_encoder").to(device)
        adapter = SafeAdapter(hidden_size, rank)
        self.model = WrappedTextEncoder(base, adapter).to(device)
        self.opt = torch.optim.AdamW(self.model.adapter.parameters(), lr=lr)
        self.use_benign = use_benign

    def _encode(self, texts):
        batch = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(self.device)
        with torch.no_grad():  # 只对 adapter 求梯度
            out = self.model.t5(**batch).last_hidden_state
        # adapter 参与梯度；训练阶段 scale 固定为 1.0
        out = self.model.adapter(out, scale=1.0)  # [B, L, D]
        sent = out.mean(dim=1)  # 简单句向量池化
        return sent

    def step(self, m_list, r_list, b_list=None, margin=0.3, lam_benign=0.1):
        # Anchor = adapter(T5(malicious)), Positive = T5(rewritten) 经 adapter
        # 注意：正样本也过 adapter，适配器学“整体分布”映射更稳
        a = self._encode(m_list)
        p = self._encode(r_list)
        n = self._encode(m_list)  # malicious 本身作 negative（不加 <safe>）

        d_ap = F.pairwise_distance(a, p)
        d_an = F.pairwise_distance(a, n)
        triplet = torch.relu(d_ap - d_an + margin).mean()

        if self.use_benign:
            b0 = self._encode(b_list)  # benign 过 adapter 前后尽量不变（adapter 已经在 _encode 内）
            # 为了做“恒等约束”，再跑一遍“冻结 adapter”的版本获取目标
            with torch.no_grad():
                batch_b = self.tokenizer(b_list, padding=True, truncation=True, return_tensors="pt").to(self.device)
                b_ref = self.model.t5(**batch_b).last_hidden_state.mean(dim=1)
            benign_cons = F.mse_loss(b0, b_ref)
        else:
            benign_cons = torch.tensor(0.0, device=self.device)

        loss = triplet + lam_benign * benign_cons

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        return {
            "loss": loss.item(),
            "triplet": triplet.item(),
            "benign": benign_cons.item(),
            "d_ap": d_ap.mean().item(),
            "d_an": d_an.mean().item(),
        }


def train_adapter(args):
    trainer = SafeAdapterTrainer(
        model_path=args.model_path,
        hidden_size=args.hidden_size, rank=args.rank, lr=args.lr, device=args.device, use_benign=args.use_benign
    )
    loader = DataLoader(PairDataset(args.trainset_path), batch_size=args.batch_size, shuffle=True)
    for epoch in range(args.num_epochs):
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{args.num_epochs}", dynamic_ncols=True, leave=False)
        for batch in pbar:
            logs = trainer.step(
                batch["malicious"], batch["rewritten"], batch["benign"],
                margin=args.margin, lam_benign=args.lam_benign
            )
            # 在进度条尾部展示关键指标，避免打断进度条
            pbar.set_postfix({
                "loss": f"{logs['loss']:.3f}",
                "triplet": f"{logs['triplet']:.3f}",
                "benign": f"{logs['benign']:.3f}",
            })

        # 按间隔保存 checkpoint
        if hasattr(args, "save_every") and args.save_every and (epoch + 1) % args.save_every == 0:
            os.makedirs(os.path.dirname(args.adapter_path), exist_ok=True)
            ckpt_path = args.adapter_path.replace('.pt', f"_epoch{epoch+1}.pt")
            torch.save(trainer.model.adapter.state_dict(), ckpt_path)
            print(f"✅ 周期性保存: {ckpt_path}")
    torch.save(trainer.model.adapter.state_dict(), args.adapter_path)
    print(f"✅ SafeAdapter 已保存到 {args.adapter_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="训练SafeAdapter模型")
    
    # 必需参数
    parser.add_argument("--category", type=str, required=True,
                        help="分类类别 (sexual/violence/political/disturbing)")
    
    # 模型相关参数
    parser.add_argument("--model_path", type=str,
                        default="/home/beihang/jzl/models/zai-org/CogVideoX-5b",
                        help="基础模型路径")
    parser.add_argument("--hidden_size", type=int, default=4096,
                        help="隐藏层大小")
    parser.add_argument("--rank", type=int, default=256,
                        help="Adapter的rank")
    
    # 训练相关参数
    parser.add_argument("--lr", type=float, default=5e-4,
                        help="学习率")
    parser.add_argument("--num_epochs", type=int, default=100,
                        help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="批次大小")
    parser.add_argument("--margin", type=float, default=0.1,
                        help="Triplet loss的margin")
    parser.add_argument("--lam_benign", type=float, default=0.1,
                        help="Benign loss的权重")
    
    # 数据相关参数
    parser.add_argument("--trainset_path", type=str, default=None,
                        help="训练集路径 (默认: datasets/train/{category}.csv)")
    parser.add_argument("--adapter_path", type=str, default=None,
                        help="Adapter保存路径 (默认: checkpoints/{category}/safe_adapter.pt)")
    
    # 其他参数
    parser.add_argument("--device", type=str, default="cuda",
                        help="设备 (cuda/cpu)")
    parser.add_argument("--use_benign", action="store_true",
                        help="是否使用benign样本")
    parser.add_argument("--save_every", type=int, default=5,
                        help="每N个epoch保存一次checkpoint (0表示不保存中间checkpoint)")
    
    args = parser.parse_args()
    
    # 设置默认路径
    if args.trainset_path is None:
        args.trainset_path = f"datasets/train/{args.category}.csv"
    if args.adapter_path is None:
        args.adapter_path = f"checkpoints/{args.category}/safe_adapter.pt"
    
    train_adapter(args)