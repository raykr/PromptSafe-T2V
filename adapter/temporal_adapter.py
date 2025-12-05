# temporal_adapter.py
import argparse
import os
import math
from typing import List, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from transformers import AutoTokenizer, T5EncoderModel
from diffusers import CogVideoXPipeline, CogVideoXDPMScheduler
from diffusers.utils import export_to_video


# =========================
# 1) Temporal Controller
# =========================
class TemporalController(nn.Module):
    r"""
    输入归一化帧索引/时间标量 tau \in [0,1]，输出门控系数 gamma(tau) \in [gamma_min, gamma_max]
    轻量 MLP，可换 1D conv/小 transformer
    """
    def __init__(self, hidden=16, gamma_min=0.5, gamma_max=1.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )
        self.gamma_min = gamma_min
        self.gamma_max = gamma_max

    def forward(self, tau: torch.Tensor):
        """
        tau: [B,1] 或 [1,1]，数值范围建议 [0,1]
        """
        x = self.net(tau)
        g = torch.sigmoid(x)  # (0,1)
        return self.gamma_min + (self.gamma_max - self.gamma_min) * g


# =========================
# 2) Temporal-Safe Adapter
# =========================
class TemporalSafeAdapter(nn.Module):
    """
    在 T5 encoder 输出空间做低秩瓶颈映射，并用时间门控控制强度：
        H_safe = H + gate(τ) * Up(GELU(Down(LN(H))))
    """
    def __init__(self, hidden_size: int, rank: int = 256, init_gate: float = 0.5,
                 gamma_min: float = 0.5, gamma_max: float = 1.5):
        super().__init__()
        self.ln = nn.LayerNorm(hidden_size)
        self.down = nn.Linear(hidden_size, rank, bias=False)
        self.up = nn.Linear(rank, hidden_size, bias=False)
        self.act = nn.GELU()

        # learnable base gate（标量基线，可学习）
        self.base_gate = nn.Parameter(torch.tensor(init_gate, dtype=torch.float32))

        # temporal controller
        self.controller = TemporalController(hidden=16, gamma_min=gamma_min, gamma_max=gamma_max)

        # 初始化
        nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.up.weight)

    def forward(self, hidden_states: torch.Tensor, tau: float = 0.0):
        """
        hidden_states: [B, L, D]
        tau: 归一化时间标量 (0~1)，用于产生门控
        """
        x = self.ln(hidden_states)
        delta = self.up(self.act(self.down(x)))

        tau_tensor = torch.tensor([[tau]], device=hidden_states.device, dtype=hidden_states.dtype)
        gamma = self.controller(tau_tensor)  # [1,1]
        gate = self.base_gate * gamma  # 广播到 [B,L,D] 自动完成

        return hidden_states + gate * delta


# =========================
# 3) 包装 T5 Encoder（可被 pipeline 直接使用）
# =========================
class WrappedTextEncoder(nn.Module):
    """
    包装原始 T5Encoder，使其 forward 时通过 TemporalSafeAdapter 做时序门控映射。
    通过 set_tau() 接口控制当前 τ；若不显式设置则 τ=0.0。
    """
    def __init__(self, t5_encoder: nn.Module, adapter: TemporalSafeAdapter):
        super().__init__()
        self.t5 = t5_encoder
        self.adapter = adapter

        for p in self.t5.parameters():
            p.requires_grad_(False)  # 冻结底座

        self._current_tau = 0.0  # 外部可修改

    # diffusers pipeline 会用到
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

    def set_tau(self, tau: float):
        self._current_tau = float(max(0.0, min(1.0, tau)))

    def forward(self, input_ids=None, attention_mask=None, inputs_embeds=None, **kwargs):
        outputs = self.t5(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            output_hidden_states=False,
            return_dict=True,
        )
        hs = outputs.last_hidden_state  # [B, L, D]
        hs_safe = self.adapter(hs, tau=self._current_tau)
        outputs.last_hidden_state = hs_safe
        return outputs


# =========================
# 4) 数据集
# =========================
class PairDataset(Dataset):
    """
    CSV 列包含: prompt, rewritten_prompt, benign_prompt
    """
    def __init__(self, csv_path: str):
        self.data = pd.read_csv(csv_path)
        self.malicious = self.data['prompt'].astype(str).tolist()
        self.rewritten = self.data['rewritten_prompt'].astype(str).tolist()
        self.benign = self.data['benign_prompt'].astype(str).tolist() if 'benign_prompt' in self.data.columns else []
        if not self.benign:
            self.benign = self.rewritten  # 没有就用 rewritten 代替
        print(f"Loaded {len(self.malicious)} samples from {csv_path}")

    def __len__(self):
        return len(self.malicious)

    def __getitem__(self, i):
        return {
            "malicious": self.malicious[i],
            "rewritten": self.rewritten[i],
            "benign": self.benign[i],
        }


def collate_fn(batch: List[Dict[str, str]]):
    return {
        "malicious": [b["malicious"] for b in batch],
        "rewritten": [b["rewritten"] for b in batch],
        "benign": [b["benign"] for b in batch],
    }


# =========================
# 5) Trainer
# =========================
class TemporalSafeAdapterTrainer:
    """
    训练仅更新 Adapter（含 controller），冻结 T5 base。
    损失：
      - Align: 恶意 vs 重写（逐τ）
      - Temporal: 相邻 τ 嵌入平滑
      - Benign-keep: 良性保持（适配器前后）
    """
    def __init__(self, model_path, hidden_size=4096, rank=256, lr=5e-4, device="cuda",
                 lambda_temporal=0.5, lambda_benign=0.1, T_steps=8,
                 gamma_min=0.5, gamma_max=1.5):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, subfolder="tokenizer")
        base = T5EncoderModel.from_pretrained(model_path, subfolder="text_encoder").to(device)

        adapter = TemporalSafeAdapter(hidden_size=hidden_size, rank=rank,
                                      init_gate=0.5, gamma_min=gamma_min, gamma_max=gamma_max).to(device)
        self.model = WrappedTextEncoder(base, adapter).to(device)

        # 仅训练 adapter（包括时间 controller）
        for p in self.model.parameters():
            p.requires_grad_(False)
        for p in self.model.adapter.parameters():
            p.requires_grad_(True)

        self.opt = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=lr
        )
        self.lambda_temporal = lambda_temporal
        self.lambda_benign = lambda_benign
        self.T_steps = T_steps

        # 冻结的 base encoder（做 benign 保持的参考）
        self.base_encoder = T5EncoderModel.from_pretrained(model_path, subfolder="text_encoder").to(device)
        self.base_encoder.eval().requires_grad_(False)
        self.base_tokenizer = AutoTokenizer.from_pretrained(model_path, subfolder="tokenizer")

    @torch.no_grad()
    def _base_encode(self, texts: List[str]) -> torch.Tensor:
        toks = self.base_tokenizer(texts, padding=True, truncation=True, max_length=64, return_tensors="pt").to(self.device)
        out = self.base_encoder(**toks).last_hidden_state  # [B,L,D]
        return out.mean(dim=1)  # [B,D]

    def _encode_with_tau(self, texts: List[str], tau: float) -> torch.Tensor:
        batch = self.tokenizer(texts, padding=True, truncation=True, max_length=64, return_tensors="pt").to(self.device)
        self.model.set_tau(tau)
        # T5 冻结，adapter 参与梯度
        with torch.no_grad():
            hs = self.model.t5(**batch).last_hidden_state  # [B,L,D]
        hs_safe = self.model.adapter(hs, tau=tau)  # [B,L,D]
        return hs_safe.mean(dim=1)  # [B,D]

    def step(self, batch: Dict[str, List[str]]) -> Dict[str, float]:
        self.opt.zero_grad()
        mal, rew, ben = batch["malicious"], batch["rewritten"], batch["benign"]

        # 逐 τ 采样（等距或随机），这里用等距
        taus = [t / max(self.T_steps - 1, 1) for t in range(self.T_steps)]

        # 编码序列
        mal_seq, rew_seq, ben_seq = [], [], []
        for tau in taus:
            mal_seq.append(self._encode_with_tau(mal, tau))  # [B,D]
            rew_seq.append(self._encode_with_tau(rew, tau))
            ben_seq.append(self._encode_with_tau(ben, tau))

        mal_seq = torch.stack(mal_seq, dim=0)  # [T,B,D]
        rew_seq = torch.stack(rew_seq, dim=0)  # [T,B,D]
        ben_seq = torch.stack(ben_seq, dim=0)  # [T,B,D]

        # 1) 对齐损失（逐 τ）
        align = F.mse_loss(mal_seq, rew_seq)

        # 2) 时序平滑（相邻 τ）
        temporal = F.mse_loss(mal_seq[1:], mal_seq[:-1]) + F.mse_loss(rew_seq[1:], rew_seq[:-1])

        # 3) 良性保持（与 base encoder 的句向量对齐）
        with torch.no_grad():
            ben_base = self._base_encode(ben)  # [B,D]
            ben_base = ben_base.unsqueeze(0).expand(self.T_steps, -1, -1)
        benign_keep = F.mse_loss(ben_seq, ben_base)

        loss = align + self.lambda_temporal * temporal + self.lambda_benign * benign_keep
        loss.backward()
        self.opt.step()

        return {
            "loss": float(loss.item()),
            "align": float(align.item()),
            "temporal": float(temporal.item()),
            "benign_keep": float(benign_keep.item()),
        }

    def save_adapter(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.model.adapter.state_dict(), path)
        print(f"✅ TemporalSafeAdapter saved to {path}")

    def load_adapter(self, path: str):
        sd = torch.load(path, map_location="cpu")
        self.model.adapter.load_state_dict(sd, strict=True)
        print(f"✅ TemporalSafeAdapter loaded from {path}")


# =========================
# 6) Pipeline 注入/评估（分段门控）
# =========================
def inject_temporal_adapter(pipe: CogVideoXPipeline, adapter_path: str,
                            hidden_size=4096, rank=256,
                            gamma_min=0.5, gamma_max=1.5):
    # 构建 + 加载 adapter
    adapter = TemporalSafeAdapter(hidden_size=hidden_size, rank=rank,
                                  init_gate=0.5, gamma_min=gamma_min, gamma_max=gamma_max)
    sd = torch.load(adapter_path, map_location="cpu")
    adapter.load_state_dict(sd, strict=True)

    # 包装 text_encoder
    wrapped = WrappedTextEncoder(pipe.text_encoder, adapter)
    wrapped = wrapped.to(device=pipe.device, dtype=getattr(pipe.text_encoder, "dtype", torch.float16))

    # 注册回 pipeline
    pipe.text_encoder = wrapped
    components = pipe.components.copy()
    components["text_encoder"] = pipe.text_encoder
    pipe.register_modules(**components)
    pipe.to(pipe.device)

    print("✅ TemporalSafeAdapter injected into pipeline.text_encoder")
    return pipe


def generate_video_with_segments(
    pipe: CogVideoXPipeline,
    prompt: str,
    total_frames: int,
    num_segments: int,
    height: int,
    width: int,
    steps: int,
    guidance: float,
    fps: int,
    out_path: str,
    controller_schedule: str = "learned",  # "learned" | "linear" | "const"
):
    """
    分段推理：每段设置不同 tau，生成子视频（若干帧），最后合并成一个视频。
    兼容 pipeline 内部“单次文本编码”的假设。
    """
    assert hasattr(pipe.text_encoder, "set_tau"), "text_encoder must be WrappedTextEncoder with set_tau()."

    frames_all = []
    frames_per_seg = max(1, total_frames // num_segments)
    rest = total_frames - frames_per_seg * num_segments

    for seg in range(num_segments):
        seg_frames = frames_per_seg + (1 if seg < rest else 0)

        # 计算该段的 tau（0~1）。这里提供三种简单策略：
        if controller_schedule == "linear":
            tau = seg / max(num_segments - 1, 1)
        elif controller_schedule == "const":
            tau = 0.5
        else:  # "learned"：沿用训练时的习惯，用段索引映射到 [0,1]；若 adapter 内有更复杂 controller 亦可拓展
            tau = seg / max(num_segments - 1, 1)

        pipe.text_encoder.set_tau(tau)

        out = pipe(
            prompt=prompt,
            num_frames=seg_frames,
            height=height,
            width=width,
            num_inference_steps=steps,
            guidance_scale=guidance,
        ).frames[0]  # list of PIL images
        frames_all.extend(out)

    export_to_video(frames_all, out_path, fps=fps)
    print(f"🎬 saved merged video to {out_path}")


# =========================
# 7) 训练 / 评估入口
# =========================
def train_adapter(args):
    trainer = TemporalSafeAdapterTrainer(
        model_path=args.model_path,
        hidden_size=args.hidden_size,
        rank=args.rank,
        lr=args.lr,
        device=args.device,
        lambda_temporal=args.lambda_temporal,
        lambda_benign=args.lambda_benign,
        T_steps=args.T_steps,
        gamma_min=args.gamma_min,
        gamma_max=args.gamma_max,
    )

    ds = PairDataset(args.trainset_path)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

    for epoch in range(args.num_epochs):
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{args.num_epochs}", dynamic_ncols=True)
        for batch in pbar:
            logs = trainer.step(batch)
            pbar.set_postfix({k: f"{v:.3f}" for k, v in logs.items()})

        if args.save_every and (epoch + 1) % args.save_every == 0:
            ckpt_path = args.adapter_path.replace(".pt", f"_epoch{epoch+1}.pt")
            trainer.save_adapter(ckpt_path)

    trainer.save_adapter(args.adapter_path)


def eval_adapter(args):
    # baseline pipeline（未注入）
    pipe_raw = CogVideoXPipeline.from_pretrained(args.model_path, torch_dtype=torch.float16)
    pipe_raw.scheduler = CogVideoXDPMScheduler.from_config(pipe_raw.scheduler.config, timestep_spacing="trailing")
    pipe_raw.to(args.device)
    pipe_raw.vae.enable_slicing()
    pipe_raw.vae.enable_tiling()

    # 注入 adapter 的 pipeline
    pipe_safe = CogVideoXPipeline.from_pretrained(args.model_path, torch_dtype=torch.float16)
    pipe_safe.scheduler = CogVideoXDPMScheduler.from_config(pipe_safe.scheduler.config, timestep_spacing="trailing")
    pipe_safe.to(args.device)
    pipe_safe.vae.enable_slicing()
    pipe_safe.vae.enable_tiling()
    pipe_safe = inject_temporal_adapter(
        pipe_safe, args.adapter_path, hidden_size=args.hidden_size, rank=args.rank,
        gamma_min=args.gamma_min, gamma_max=args.gamma_max
    )

    os.makedirs(args.output_dir, exist_ok=True)

    data = pd.read_csv(args.testset_path)
    prompts = data["prompt"].astype(str).tolist()

    for i, prompt in enumerate(prompts):
        # raw
        video_raw = pipe_raw(
            prompt=prompt,
            num_frames=args.num_frames,
            height=args.height,
            width=args.width,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
        ).frames[0]
        path_raw = f"{args.output_dir}/tempSafe_{i:03d}_raw.mp4"
        export_to_video(video_raw, path_raw, fps=args.fps)
        print(f"✅ RAW saved to {path_raw}")

        # safe（分段门控）
        path_safe = f"{args.output_dir}/tempSafe_{i:03d}_safe.mp4"
        generate_video_with_segments(
            pipe=pipe_safe,
            prompt=prompt,
            total_frames=args.num_frames,
            num_segments=args.num_segments,
            height=args.height,
            width=args.width,
            steps=args.num_inference_steps,
            guidance=args.guidance_scale,
            fps=args.fps,
            out_path=path_safe,
            controller_schedule=args.controller_schedule,  # "learned"|"linear"|"const"
        )


def build_args():
    cfg = {
        "mode": "eval",  # "train" or "eval"
        "model_path": "/home/beihang/jzl/models/zai-org/CogVideoX-5b",
        "hidden_size": 4096,
        "rank": 256,
        "gamma_min": 0.5,
        "gamma_max": 1.5,
        "lr": 5e-4,
        "device": "cuda",

        # train
        "num_epochs": 10,
        "batch_size": 8,
        "T_steps": 8,                # 每个 batch 的 τ 采样步数
        "lambda_temporal": 0.5,
        "lambda_benign": 0.1,
        "save_every": 2,
        "trainset_path": "datasets/train/2.csv",
        "adapter_path": "checkpoints/temporal_safe_adapter.pt",

        # eval / gen
        "testset_path": "datasets/test/demo.csv",
        "output_dir": "out/temporal_demo",
        "num_frames": 49,
        "height": 480,
        "width": 720,
        "num_inference_steps": 28,
        "guidance_scale": 6.0,
        "fps": 24,

        # 分段门控设置
        "num_segments": 7,                 # 把视频分成 7 段，每段单独设 tau
        "controller_schedule": "learned",  # "learned"|"linear"|"const"
    }
    return argparse.Namespace(**cfg)


if __name__ == "__main__":
    args = build_args()
    if args.mode == "train":
        train_adapter(args)
    else:
        eval_adapter(args)
