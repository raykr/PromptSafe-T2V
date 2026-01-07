#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
VideoCrafter2 Safe Adapter (per-harm-category) training + inference.

VideoCrafter2 is an open-source toolbox including Text2Video. :contentReference[oaicite:3]{index=3}
We:
- Load VideoCrafter2 T2V model (LVDM style).
- Identify its text conditioning encoder (usually model.cond_stage_model).
- Attach LoRA adapters to cond_stage_model's transformer projection layers.
- Train adapters using the same classifier-guided embedding editing objective.

You MUST implement:
- load_videocrafter2_model(...)
- vc2_encode_prompt(...)
- vc2_sample(...)
according to your local VideoCrafter2 checkout (configs/scripts).

"""

import os
import json
import argparse
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from peft import LoraConfig, get_peft_model, PeftModel

HARM_CATEGORIES = ["sexual", "violent", "political", "disturbing"]


# ---------------------------
# Data
# ---------------------------
class PromptLabelDataset(Dataset):
    def __init__(self, jsonl_path: str):
        self.items = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                self.items.append((obj["prompt"], obj["label"]))
        assert self.items, "Empty dataset"

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


def collate_batch(batch):
    prompts, labels = zip(*batch)
    return list(prompts), list(labels)


# ---------------------------
# Toxic classifier interface (same as Open-Sora file)
# ---------------------------
class ToxicClassifier(nn.Module):
    def __init__(self, embed_dim: int, num_classes: int = 4):
        super().__init__()
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_classes),
        )

    @torch.no_grad()
    def predict_label(self, prompt_list: List[str]) -> Tuple[List[str], List[float]]:
        return ["disturbing"] * len(prompt_list), [0.5] * len(prompt_list)

    def forward_text_embeds(self, embeds: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (embeds * mask).sum(dim=1) / (mask.sum(dim=1).clamp_min(1.0))
        return self.head(pooled)


# ---------------------------
# Adapter bank for VC2 cond encoder
# ---------------------------
@dataclass
class AdapterSpec:
    name: str
    out_dir: str
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    target_modules: Tuple[str, ...] = ("q", "k", "v", "out_proj", "to_q", "to_k", "to_v", "to_out")


class CondEncoderAdapterBank:
    """
    Wrap VideoCrafter2 cond_stage_model with PEFT so we can manage multiple adapters.
    """
    def __init__(self, cond_encoder: nn.Module, adapter_root: str, device: torch.device):
        self.base = cond_encoder
        self.adapter_root = adapter_root
        self.device = device
        self.peft_model: Optional[PeftModel] = None

    def _ensure_peft(self, spec: AdapterSpec):
        if self.peft_model is not None:
            return
        cfg = LoraConfig(
            r=spec.lora_r,
            lora_alpha=spec.lora_alpha,
            lora_dropout=spec.lora_dropout,
            bias="none",
            task_type="FEATURE_EXTRACTION",
            target_modules=list(spec.target_modules),
        )
        self.peft_model = get_peft_model(self.base, cfg).to(self.device)

    def add_adapter(self, spec: AdapterSpec):
        self._ensure_peft(spec)
        assert self.peft_model is not None
        if spec.name not in self.peft_model.peft_config:
            cfg = LoraConfig(
                r=spec.lora_r,
                lora_alpha=spec.lora_alpha,
                lora_dropout=spec.lora_dropout,
                bias="none",
                task_type="FEATURE_EXTRACTION",
                target_modules=list(spec.target_modules),
            )
            self.peft_model.add_adapter(spec.name, cfg)

    def set_adapter(self, name: Optional[str], scale: float = 1.0):
        assert self.peft_model is not None
        if name is None:
            self.peft_model.set_adapter(None)
            return
        self.peft_model.set_adapter(name)
        for m in self.peft_model.modules():
            if hasattr(m, "scaling"):
                try:
                    m.scaling = float(m.scaling) * float(scale)
                except Exception:
                    pass

    def trainable_parameters(self) -> List[nn.Parameter]:
        assert self.peft_model is not None
        return [p for p in self.peft_model.parameters() if p.requires_grad]

    def save_adapter(self, name: str, out_dir: str):
        assert self.peft_model is not None
        os.makedirs(out_dir, exist_ok=True)
        self.peft_model.save_pretrained(out_dir, selected_adapters=[name])

    def load_adapter(self, name: str, from_dir: str, spec: AdapterSpec):
        self._ensure_peft(spec)
        assert self.peft_model is not None
        self.peft_model.load_adapter(from_dir, adapter_name=name)


# ---------------------------
# VideoCrafter2 bindings (YOU MUST IMPLEMENT)
# ---------------------------
def load_videocrafter2_model(videocrafter_repo: str, config_path: str, ckpt_path: str, device: torch.device) -> dict:
    """
    Return dict with:
      - model: the diffusion model wrapper (LVDM)
      - cond_encoder: model.cond_stage_model (text encoder)
      - tokenizer (if exists) OR functions for encode
      - sampler/scheduler
      - any other needed components

    Implement by adapting VideoCrafter2 scripts/run_text2video.sh + python entry.
    """
    raise NotImplementedError("Please implement this using your local VideoCrafter2 checkout.")


@torch.no_grad()
def vc2_encode_prompt(components: dict, cond_encoder: nn.Module, prompts: List[str], device: torch.device):
    """
    Encode prompt for VideoCrafter2 conditioning.
    Return:
      - embeds: [B, L, D]
      - attention_mask: [B, L]

    Many LVDM pipelines expose something like:
      c = cond_encoder(prompts)  # possibly already tokenizes inside
    If your cond encoder returns [B, D] instead of [B, L, D], adapt accordingly.
    """
    raise NotImplementedError("Please implement to match your VC2 cond encoder API.")


@torch.no_grad()
def vc2_sample(components: dict, prompt_embeds: torch.Tensor, prompt_mask: torch.Tensor,
              num_frames: int, height: int, width: int,
              num_steps: int, guidance: float, seed: int,
              out_path: str):
    """
    Run VideoCrafter2 sampling with adapted prompt embeddings and save mp4/gif.
    """
    raise NotImplementedError("Please implement by adapting VC2 inference script.")


# ---------------------------
# Training
# ---------------------------
def train_one_adapter(
    components: dict,
    bank: CondEncoderAdapterBank,
    classifier: ToxicClassifier,
    spec: AdapterSpec,
    train_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    lambda_tox: float,
    lambda_preserve: float,
):
    bank.add_adapter(spec)
    bank.set_adapter(spec.name, scale=1.0)

    bank.peft_model.train()
    for n, p in bank.peft_model.named_parameters():
        if "lora_" not in n:
            p.requires_grad = False

    classifier.eval()
    for p in classifier.parameters():
        p.requires_grad = False

    optim = torch.optim.AdamW(bank.trainable_parameters(), lr=lr)

    for ep in range(epochs):
        for prompts, labels in train_loader:
            keep = [i for i, y in enumerate(labels) if y == spec.name]
            if not keep:
                continue
            prompts = [prompts[i] for i in keep]

            # base embeds
            bank.set_adapter(None)
            base_embeds, attn = vc2_encode_prompt(components, bank.base, prompts, device)

            # adapted embeds
            bank.set_adapter(spec.name, scale=1.0)
            adapted_embeds, attn2 = vc2_encode_prompt(components, bank.peft_model, prompts, device)

            y_idx = HARM_CATEGORIES.index(spec.name)
            logits = classifier.forward_text_embeds(adapted_embeds, attn2)
            tox_loss = logits[:, y_idx].mean()
            preserve_loss = F.mse_loss(adapted_embeds, base_embeds)
            loss = lambda_tox * tox_loss + lambda_preserve * preserve_loss

            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()

        print(f"[{spec.name}] epoch {ep+1}/{epochs} done")

    out_dir = os.path.join(spec.out_dir, spec.name)
    bank.save_adapter(spec.name, out_dir)
    print(f"[{spec.name}] saved adapter to: {out_dir}")


# ---------------------------
# Inference
# ---------------------------
@torch.no_grad()
def infer_dynamic(
    components: dict,
    bank: CondEncoderAdapterBank,
    classifier: ToxicClassifier,
    prompt: str,
    defense_scale: float,
    out_path: str,
    num_frames: int,
    height: int,
    width: int,
    num_steps: int,
    guidance: float,
    seed: int,
    force_category: Optional[str] = None,
):
    if force_category is None:
        pred_labels, pred_probs = classifier.predict_label([prompt])
        category = pred_labels[0]
    else:
        category = force_category

    if category not in HARM_CATEGORIES:
        category = "disturbing"

    bank.set_adapter(category, scale=defense_scale)
    embeds, mask = vc2_encode_prompt(components, bank.peft_model, [prompt], bank.device)
    vc2_sample(
        components=components,
        prompt_embeds=embeds,
        prompt_mask=mask,
        num_frames=num_frames,
        height=height,
        width=width,
        num_steps=num_steps,
        guidance=guidance,
        seed=seed,
        out_path=out_path,
    )
    print(f"[infer] category={category}, scale={defense_scale}, saved={out_path}")


# ---------------------------
# CLI
# ---------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["train", "infer"], required=True)

    ap.add_argument("--videocrafter_repo", type=str, required=True)
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--ckpt", type=str, required=True)

    ap.add_argument("--train_jsonl", type=str, default="")
    ap.add_argument("--adapter_root", type=str, required=True)

    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lambda_tox", type=float, default=1.0)
    ap.add_argument("--lambda_preserve", type=float, default=0.1)

    ap.add_argument("--prompt", type=str, default="")
    ap.add_argument("--out", type=str, default="out.mp4")
    ap.add_argument("--defense_scale", type=float, default=1.0)
    ap.add_argument("--force_category", type=str, default="")

    ap.add_argument("--num_frames", type=int, default=16)
    ap.add_argument("--height", type=int, default=320)
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--num_steps", type=int, default=50)
    ap.add_argument("--guidance", type=float, default=12.0)
    ap.add_argument("--seed", type=int, default=0)

    # classifier embedding dim (set to match your VC2 cond encoder output dim)
    ap.add_argument("--embed_dim", type=int, default=768)

    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    components = load_videocrafter2_model(
        videocrafter_repo=args.videocrafter_repo,
        config_path=args.config,
        ckpt_path=args.ckpt,
        device=device,
    )
    cond_encoder: nn.Module = components["cond_encoder"].to(device).eval()

    bank = CondEncoderAdapterBank(cond_encoder=cond_encoder, adapter_root=args.adapter_root, device=device)

    classifier = ToxicClassifier(embed_dim=args.embed_dim, num_classes=len(HARM_CATEGORIES)).to(device).eval()

    if args.mode == "train":
        assert args.train_jsonl
        ds = PromptLabelDataset(args.train_jsonl)
        dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_batch, num_workers=2)

        os.makedirs(args.adapter_root, exist_ok=True)
        for cat in HARM_CATEGORIES:
            spec = AdapterSpec(name=cat, out_dir=args.adapter_root)
            train_one_adapter(
                components=components,
                bank=bank,
                classifier=classifier,
                spec=spec,
                train_loader=dl,
                device=device,
                epochs=args.epochs,
                lr=args.lr,
                lambda_tox=args.lambda_tox,
                lambda_preserve=args.lambda_preserve,
            )
    else:
        for cat in HARM_CATEGORIES:
            spec = AdapterSpec(name=cat, out_dir=args.adapter_root)
            bank.add_adapter(spec)
            adapter_dir = os.path.join(args.adapter_root, cat)
            if os.path.isdir(adapter_dir):
                bank.load_adapter(cat, adapter_dir, spec)
            else:
                print(f"[warn] adapter dir not found: {adapter_dir}")

        force = args.force_category.strip() or None
        infer_dynamic(
            components=components,
            bank=bank,
            classifier=classifier,
            prompt=args.prompt,
            defense_scale=args.defense_scale,
            out_path=args.out,
            num_frames=args.num_frames,
            height=args.height,
            width=args.width,
            num_steps=args.num_steps,
            guidance=args.guidance,
            seed=args.seed,
            force_category=force,
        )


if __name__ == "__main__":
    main()
