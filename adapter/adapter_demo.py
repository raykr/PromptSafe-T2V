import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from transformers import AutoTokenizer, T5EncoderModel
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from diffusers import CogVideoXPipeline, CogVideoXDPMScheduler
import pandas as pd
from diffusers.utils import export_to_video

from classifier import PromptSafetyClassifier

class SafeAdapter(nn.Module):
    """
    冻结 T5 输出后，做一个小瓶颈 + 残差的安全映射层
    H_safe = H + gate * scale * MLP(LN(H))
    其中 scale 由外部动态控制（例如来自 prompt 分类器）
    """

    def __init__(self, hidden_size: int, rank: int = 256, init_gate: float = 0.5):
        super().__init__()
        self.ln = nn.LayerNorm(hidden_size)
        self.down = nn.Linear(hidden_size, rank, bias=False)
        self.up = nn.Linear(rank, hidden_size, bias=False)
        self.act = nn.GELU()
        # 可学习 base gate
        self.gate = nn.Parameter(torch.tensor(init_gate))

        # 小初始化，避免一开始破坏分布
        nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.up.weight)

    def forward(self, hidden_states: torch.Tensor, scale: torch.Tensor | float = 1.0):
        """
        scale: 动态防御强度系数
          - 可以是标量 float
          - 也可以是 [B] 或 [B, 1, 1] 的 Tensor（会自动广播）
        """
        x = self.ln(hidden_states)
        delta = self.up(self.act(self.down(x)))  # [B, L, D]

        gate = self.gate
        if not torch.is_tensor(scale):
            scale = torch.tensor(scale, device=hidden_states.device, dtype=hidden_states.dtype)
        # scale 形状调整为 [B, 1, 1] 或 [1, 1, 1]，方便广播
        while scale.dim() < hidden_states.dim():
            scale = scale.unsqueeze(-1)

        eff_gate = gate * scale  # [B,1,1] or [1,1,1]

        return hidden_states + eff_gate * delta


class TemporalSafeAdapter(nn.Module):
    """
    对 SafeAdapter 加入时序门控：gate -> gate(τ)
    τ ∈ [0, 1] 为帧归一化索引
    """
    def __init__(self, hidden_size, rank=256, init_gate=0.5, gamma_min=0.5, gamma_max=1.5):
        super().__init__()
        self.ln = nn.LayerNorm(hidden_size)
        self.down = nn.Linear(hidden_size, rank, bias=False)
        self.up = nn.Linear(rank, hidden_size, bias=False)
        self.act = nn.GELU()

        # learnable base gate (scalar)
        self.base_gate = nn.Parameter(torch.tensor(init_gate))

        # temporal controller（可学习时间函数）
        self.time_mlp = nn.Sequential(
            nn.Linear(1, 16),
            nn.SiLU(),
            nn.Linear(16, 1),
        )
        self.gamma_min, self.gamma_max = gamma_min, gamma_max

        nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.up.weight)

    def forward(self, hidden_states: torch.Tensor, tau: float = 0.0):
        """
        tau: 当前帧归一化时间 (0 ~ 1)
        """
        x = self.ln(hidden_states)
        delta = self.up(self.act(self.down(x)))

        # 根据帧索引调节 gate
        tau_tensor = torch.tensor([[tau]], device=hidden_states.device, dtype=hidden_states.dtype)
        gamma = torch.sigmoid(self.time_mlp(tau_tensor))  # (0,1)
        gamma = self.gamma_min + (self.gamma_max - self.gamma_min) * gamma
        gate = self.base_gate * gamma

        return hidden_states + gate * delta



class WrappedTextEncoder(nn.Module):
    def __init__(self, t5_encoder: nn.Module, adapter: SafeAdapter):
        super().__init__()
        self.t5 = t5_encoder
        for p in self.t5.parameters():
            p.requires_grad_(False)
        self.adapter = adapter

        # 默认动态 scale = 1.0，可以在推理前由外部修改
        self.adapter_scale = 1.0

    # 🔧 对外暴露一个设置接口，方便在生成前根据 prompt 分类结果动态调节
    def set_adapter_scale(self, scale: torch.Tensor | float):
        """
        scale 可以是:
          - float 标量：对当前 batch 使用统一防御强度
          - [B] Tensor：对 batch 内每个样本用不同强度
        """
        self.adapter_scale = scale

    # diffusers 的 pipeline.to() 会读取这些属性
    @property
    def dtype(self):
        try:
            return next(self.parameters()).dtype
        except StopIteration:
            return torch.float32

    @property
    def device(self):
        try:
            return next(self.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    def forward(self, input_ids=None, attention_mask=None, inputs_embeds=None, **kwargs):
        outputs = self.t5(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            output_hidden_states=False,
            return_dict=True,
        )
        hs = outputs.last_hidden_state  # [B, L, D]
        # 将当前对象的 adapter_scale 传给 adapter
        hs_safe = self.adapter(hs, scale=self.adapter_scale)
        outputs.last_hidden_state = hs_safe
        return outputs


class PairDataset(Dataset):
    # items: (malicious, rewritten, benign)
    def __init__(self, csv_path):
        # prompt, rewritten_prompt, benign_prompt -> malicious, rewritten, benign
        self.data = pd.read_csv(csv_path)
        # 直接使用DataFrame的列，不需要转置
        self.malicious = self.data['prompt'].tolist()
        self.rewritten = self.data['rewritten_prompt'].tolist()
        self.benign = self.data['benign_prompt'].tolist()
        print(f"Loaded {len(self.malicious)} samples")

    def __len__(self):
        return len(self.malicious)

    def __getitem__(self, i):
        return {
            "malicious": self.malicious[i], 
            "rewritten": self.rewritten[i], 
            "benign": self.benign[i]
        }


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


def inject_safe_adapter(pipe, adapter_path, rank=256, hidden_size=4096):
    # 1) 构建并加载 Adapter
    adapter = SafeAdapter(hidden_size, rank)
    sd = torch.load(adapter_path, map_location="cpu")
    adapter.load_state_dict(sd)

    # 2) 包装原始 text_encoder
    wrapped = WrappedTextEncoder(pipe.text_encoder, adapter)

    # 3) 先放到与 pipeline 一致的 dtype/device（常见 float16 + cuda）
    wrapped = wrapped.to(device=pipe.device, dtype=getattr(pipe.text_encoder, "dtype", torch.float16))

    # 4) 替换并“正规”注册到 pipeline
    pipe.text_encoder = wrapped
    components = pipe.components.copy()
    components["text_encoder"] = pipe.text_encoder
    pipe.register_modules(**components)

    # 5) 再统一到 cuda（有些模块可能被 register 重置了 device）
    pipe.to(pipe.device)

    print("✅ SafeAdapter 已注入到 pipeline.text_encoder 并完成重新注册/对齐")
    return pipe


def load_prompt_classifier(args):
    """
    从 ckpt 加载 PromptSafetyClassifier，返回 (classifier, tokenizer, label_cols)
    """
    device = args.device
    state = torch.load(args.cls_ckpt_path, map_location="cpu")
    label_cols = state["label_cols"]

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, subfolder="tokenizer")
    base = T5EncoderModel.from_pretrained(args.model_path, subfolder="text_encoder").to(device)

    model = PromptSafetyClassifier(
        t5_encoder=base,
        hidden_size=args.hidden_size,
        num_labels=len(label_cols),
    ).to(device)
    model.load_state_dict(state["state_dict"])
    model.eval()

    return model, tokenizer, label_cols


@torch.no_grad()
def compute_severity(probs: torch.Tensor) -> torch.Tensor:
    """
    一个简单的 severity 定义示例：
      - probs: [B, num_labels]
      - 返回 severity: [B] ∈ [0,1]
    这里直接取所有类别概率的 max 作为整体有害程度，你可以根据需要改成加权和等。
    """
    severity, _ = probs.max(dim=-1)
    return severity  # [B]

def eval_adapter(args):
    # 1) 原始（未注入）pipeline
    pipe_raw = CogVideoXPipeline.from_pretrained(args.model_path, torch_dtype=torch.float16)
    pipe_raw.scheduler = CogVideoXDPMScheduler.from_config(pipe_raw.scheduler.config, timestep_spacing="trailing")
    pipe_raw.to(args.device)
    pipe_raw.vae.enable_slicing()
    pipe_raw.vae.enable_tiling()

    # 2) 注入 SafeAdapter 的 pipeline
    pipe_safe = CogVideoXPipeline.from_pretrained(args.model_path, torch_dtype=torch.float16)
    pipe_safe.scheduler = CogVideoXDPMScheduler.from_config(pipe_safe.scheduler.config, timestep_spacing="trailing")
    pipe_safe.to(args.device)
    pipe_safe.vae.enable_slicing()
    pipe_safe.vae.enable_tiling()
    pipe_safe = inject_safe_adapter(pipe_safe, args.adapter_path, args.rank, args.hidden_size)

    # 3) 加载 prompt 分类器（用于动态路由/强度控制）
    cls_model, cls_tokenizer, cls_label_cols = load_prompt_classifier(args)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # 从 testset_path 中读取 prompts
    data = pd.read_csv(args.testset_path)
    prompts = data["prompt"].tolist()
    for i, prompt in enumerate(prompts):
        # ---- 3.1 先用分类器预测该 prompt 的各类风险概率 ----
        tok = cls_tokenizer([prompt], padding=True, truncation=True, return_tensors="pt").to(args.device)
        logits = cls_model(tok["input_ids"], tok["attention_mask"])    # [1,num_labels]
        probs = torch.sigmoid(logits)                                  # [1,num_labels]

        severity = compute_severity(probs)[0].item()  # 标量 ∈ [0,1]
        # 你可以根据需要对 severity 做一个映射，比如:
        #   scale = 0.2 + 0.8 * severity
        # 代表最低 0.2 强度，最高 1.0 强度
        scale = 0.2 + 0.8 * severity

        # 将动态 scale 写入 text_encoder
        if hasattr(pipe_safe.text_encoder, "set_adapter_scale"):
            pipe_safe.text_encoder.set_adapter_scale(scale)
        else:
            # 兼容性：旧版可以直接写属性
            pipe_safe.text_encoder.adapter_scale = scale

        print(f"[{i:03d}] prompt = {prompt[:40]}..., severity = {severity:.3f}, scale = {scale:.3f}")

        # ---- 4) 生成未注入（raw） ----
        video_raw = pipe_raw(
            prompt=prompt,
            num_frames=args.num_frames,
            height=args.height,
            width=args.width,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
        ).frames[0]
        export_to_video(video_raw, f"{args.output_dir}/adapter_{i:03d}_raw.mp4", fps=args.fps)
        print(f"✅ 视频已保存到 {args.output_dir}/adapter_{i:03d}_raw.mp4")

        # ---- 5) 生成已注入（safe，动态强度）----
        video_safe = pipe_safe(
            prompt=prompt,
            num_frames=args.num_frames,
            height=args.height,
            width=args.width,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
        ).frames[0]
        export_to_video(video_safe, f"{args.output_dir}/adapter_{i:03d}_safe.mp4", fps=args.fps)
        print(f"✅ 视频已保存到 {args.output_dir}/adapter_{i:03d}_safe.mp4")



if __name__ == "__main__":
    cfg = {
        "model_path": "/home/beihang/jzl/models/zai-org/CogVideoX-5b",
        "hidden_size": 4096,
        "rank": 256,
        "lr": 5e-4,
        "device": "cuda",
        "num_epochs": 100,
        "batch_size": 8,
        "margin": 0.1,
        "lam_benign": 0.1,
        "adapter_path": "checkpoints/4/safe_adapter.pt",
        "cls_ckpt_path": "checkpoints/prompt_classifier.pt",
        "trainset_path": "datasets/train/4.csv",
        "testset_path": "datasets/train/4.csv",
        "output_dir": "out/4_cls",
        "num_frames": 81,
        "height": 480,
        "width": 720,
        "num_inference_steps": 50,
        "guidance_scale": 6.0,
        "use_benign": False,
        "save_every": 5,
        "fps": 16,
    }
    args = argparse.Namespace(**cfg)

    # train_adapter(args)
    eval_adapter(args)