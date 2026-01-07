#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Open-Sora Safe Adapter (per-harm-category) training + inference.

Core idea:
- Freeze Open-Sora base model.
- Attach LoRA adapters to the T5 text encoder (or its attention/projection modules).
- Train one adapter per harmful category by minimizing:
    L = lambda_tox * toxic_score_y(embeds) + lambda_preserve * MSE(embeds, embeds_base)

Inference:
- Use toxic classifier to predict harmful category.
- Activate corresponding adapter and set LoRA scale (defense strength).
- Run Open-Sora sampling with the adapted text embeddings.

You MUST implement:
- load_opensora_components(...)
- opensora_encode_prompt(...)
- opensora_sample(...)
according to your local Open-Sora checkout.

This file is intentionally self-contained for your current pipeline style.
"""

import os
import json
import math
import argparse
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from transformers import T5Tokenizer, T5EncoderModel

# PEFT LoRA
from peft import LoraConfig, get_peft_model, PeftModel


HARM_CATEGORIES = ["sexual", "violent", "political", "disturbing"]


# ---------------------------
# Data
# ---------------------------
class PromptLabelDataset(Dataset):
    """
    JSONL format (one sample per line):
      {"prompt": "...", "label": "sexual"}  # label in HARM_CATEGORIES
    """
    def __init__(self, jsonl_path: str):
        self.items = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                assert "prompt" in obj and "label" in obj, "Each line must have prompt/label"
                self.items.append((obj["prompt"], obj["label"]))
        assert len(self.items) > 0, f"Empty dataset: {jsonl_path}"

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int):
        return self.items[idx]


def collate_batch(batch):
    prompts, labels = zip(*batch)
    return list(prompts), list(labels)


# ---------------------------
# Toxic classifier interface
# ---------------------------
class ToxicClassifier(nn.Module):
    """
    Plug your own classifier here.

    Required API:
      forward_text_embeds(embeds, attention_mask) -> logits [B, C]
      predict_label(prompt_str_list) -> (label_str_list, prob_list)

    In your current system you already have a T5Encoder-based toxic classifier.
    You can replace the implementation below with your existing checkpoint loader.
    """
    def __init__(self, num_classes: int = 4):
        super().__init__()
        self.num_classes = num_classes

        # Minimal placeholder: a linear head over mean pooled embeddings.
        # Replace with your real classifier weights for meaningful training.
        self.head = nn.Sequential(
            nn.LayerNorm(4096),  # typical T5-XXL hidden size; change to match your T5
            nn.Linear(4096, num_classes)
        )

    @torch.no_grad()
    def predict_label(self, prompt_list: List[str]) -> Tuple[List[str], List[float]]:
        # Placeholder heuristic. Replace with your actual prompt classifier.
        # Here we just return "disturbing" with prob 0.5 for all.
        return ["disturbing"] * len(prompt_list), [0.5] * len(prompt_list)

    def forward_text_embeds(self, embeds: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        # embeds: [B, L, D]
        # attention_mask: [B, L]
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (embeds * mask).sum(dim=1) / (mask.sum(dim=1).clamp_min(1.0))
        return self.head(pooled)


# ---------------------------
# Adapter bank
# ---------------------------
@dataclass
class AdapterSpec:
    name: str
    out_dir: str
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    # Which modules in T5 to apply LoRA to: you should tune to your T5 variant.
    # Common targets: q, k, v, o projections, or "q", "k", "v", "o" substrings.
    target_modules: Tuple[str, ...] = ("q", "k", "v", "o")


class T5AdapterBank:
    """
    Maintain per-category LoRA adapters on the SAME base T5 encoder.
    """
    def __init__(self, base_t5: T5EncoderModel, adapter_root: str, device: torch.device):
        self.base_t5 = base_t5
        self.adapter_root = adapter_root
        self.device = device

        # PeftModel wrapper will be created lazily.
        self.peft_model: Optional[PeftModel] = None
        self.active_adapter: Optional[str] = None

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
        self.peft_model = get_peft_model(self.base_t5, cfg)
        self.peft_model.to(self.device)

    def add_adapter(self, spec: AdapterSpec):
        self._ensure_peft(spec)
        assert self.peft_model is not None
        # Create a named adapter config (PEFT supports multiple adapters)
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
        """
        name=None means disable adapters (use base).
        scale controls defense strength (LoRA scaling).
        """
        assert self.peft_model is not None
        if name is None:
            self.peft_model.set_adapter(None)
            self.active_adapter = None
            return

        self.peft_model.set_adapter(name)
        self.active_adapter = name
        # PEFT LoRA scaling: each lora layer has scaling = alpha/r; we can multiply.
        # Some PEFT versions expose `set_adapter` only; scaling can be done by
        # setting `lora_alpha` or using `peft_model.set_adapter` + manual scaling hook.
        # Here we implement a generic multiplier hook.
        for m in self.peft_model.modules():
            if hasattr(m, "scaling"):
                try:
                    m.scaling = float(m.scaling) * float(scale)
                except Exception:
                    pass

    def trainable_parameters(self) -> List[nn.Parameter]:
        assert self.peft_model is not None
        params = []
        for n, p in self.peft_model.named_parameters():
            if p.requires_grad:
                params.append(p)
        return params

    def save_adapter(self, name: str, out_dir: str):
        assert self.peft_model is not None
        os.makedirs(out_dir, exist_ok=True)
        self.peft_model.save_pretrained(out_dir, selected_adapters=[name])

    def load_adapter(self, name: str, from_dir: str, spec: AdapterSpec):
        self._ensure_peft(spec)
        assert self.peft_model is not None
        self.peft_model.load_adapter(from_dir, adapter_name=name)


# ---------------------------
# Open-Sora bindings (YOU MUST IMPLEMENT)
# ---------------------------
def load_opensora_components(opensora_repo: str, ckpt: str, device: torch.device):
    """
    Return a dict with at least:
      - tokenizer: T5Tokenizer
      - text_encoder: T5EncoderModel (or compatible)
      - opensora_model: the denoiser/DiT model
      - vae: video VAE (optional depending on your sampler)
      - scheduler/sampler: (optional)
    You should build this by referencing your local Open-Sora inference script/config.

    Open-Sora uses T5 encoder in their described pipeline. :contentReference[oaicite:1]{index=1}
    """
    raise NotImplementedError("Please implement this using your local Open-Sora checkout.")


@torch.no_grad()
def opensora_encode_prompt(tokenizer: T5Tokenizer, text_encoder: nn.Module, prompts: List[str], device: torch.device):
    """
    Return:
      - embeds: [B, L, D]
      - attention_mask: [B, L]
    """
    tok = tokenizer(
        prompts,
        padding="max_length",
        truncation=True,
        max_length=256,
        return_tensors="pt",
    )
    input_ids = tok.input_ids.to(device)
    attn = tok.attention_mask.to(device)
    out = text_encoder(input_ids=input_ids, attention_mask=attn)
    embeds = out.last_hidden_state
    return embeds, attn


@torch.no_grad()
def opensora_sample(components: dict, prompt_embeds: torch.Tensor, prompt_mask: torch.Tensor,
                    num_frames: int, height: int, width: int,
                    num_steps: int, guidance: float, seed: int,
                    out_path: str):
    """
    Run Open-Sora sampling given prepared prompt embeddings.

    You need to:
      - create initial noise latent
      - do diffusion steps with CFG (if applicable)
      - decode with VAE
      - save video (e.g., mp4)
    """
    raise NotImplementedError("Please implement this by adapting Open-Sora scripts/inference.py in your repo.")


# ---------------------------
# Training
# ---------------------------
def train_one_adapter(
    components: dict,
    adapter_bank: T5AdapterBank,
    classifier: ToxicClassifier,
    spec: AdapterSpec,
    train_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    lambda_tox: float,
    lambda_preserve: float,
    grad_clip: float = 1.0,
):
    tokenizer: T5Tokenizer = components["tokenizer"]

    adapter_bank.add_adapter(spec)
    adapter_bank.set_adapter(spec.name, scale=1.0)

    # Freeze base text encoder weights except LoRA
    adapter_bank.peft_model.train()
    for n, p in adapter_bank.peft_model.named_parameters():
        # PEFT marks LoRA params trainable by default; freeze others
        if "lora_" not in n:
            p.requires_grad = False

    # Classifier frozen
    classifier.eval()
    for p in classifier.parameters():
        p.requires_grad = False

    optim = torch.optim.AdamW(adapter_bank.trainable_parameters(), lr=lr)

    for ep in range(epochs):
        for prompts, labels in train_loader:
            # filter by this category
            keep = [i for i, y in enumerate(labels) if y == spec.name]
            if not keep:
                continue
            prompts = [prompts[i] for i in keep]

            # Base embeds (no adapter)
            adapter_bank.set_adapter(None)
            base_embeds, attn = opensora_encode_prompt(tokenizer, adapter_bank.base_t5, prompts, device)

            # Adapted embeds
            adapter_bank.set_adapter(spec.name, scale=1.0)
            adapted_embeds, attn2 = opensora_encode_prompt(tokenizer, adapter_bank.peft_model, prompts, device)

            # Toxic score for this category index
            y_idx = HARM_CATEGORIES.index(spec.name)
            logits = classifier.forward_text_embeds(adapted_embeds, attn2)
            # higher logit => more toxic; minimize it
            tox_loss = logits[:, y_idx].mean()

            preserve_loss = F.mse_loss(adapted_embeds, base_embeds)

            loss = lambda_tox * tox_loss + lambda_preserve * preserve_loss

            optim.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(adapter_bank.trainable_parameters(), grad_clip)
            optim.step()

        print(f"[{spec.name}] epoch {ep+1}/{epochs} done")

    out_dir = os.path.join(spec.out_dir, spec.name)
    adapter_bank.save_adapter(spec.name, out_dir)
    print(f"[{spec.name}] saved adapter to: {out_dir}")


# ---------------------------
# Inference
# ---------------------------
@torch.no_grad()
def infer_with_dynamic_adapter(
    components: dict,
    adapter_bank: T5AdapterBank,
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
    tokenizer: T5Tokenizer = components["tokenizer"]

    if force_category is None:
        pred_labels, pred_probs = classifier.predict_label([prompt])
        category = pred_labels[0]
        score = pred_probs[0]
    else:
        category, score = force_category, 1.0

    if category not in HARM_CATEGORIES:
        category = "disturbing"

    adapter_bank.set_adapter(category, scale=defense_scale)

    # Encode with adapted T5
    prompt_embeds, prompt_mask = opensora_encode_prompt(tokenizer, adapter_bank.peft_model, [prompt], adapter_bank.device)

    opensora_sample(
        components=components,
        prompt_embeds=prompt_embeds,
        prompt_mask=prompt_mask,
        num_frames=num_frames,
        height=height,
        width=width,
        num_steps=num_steps,
        guidance=guidance,
        seed=seed,
        out_path=out_path,
    )
    print(f"[infer] category={category}, score={score:.4f}, scale={defense_scale}, saved={out_path}")


# ---------------------------
# CLI
# ---------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["train", "infer"], required=True)

    # Open-Sora paths
    ap.add_argument("--opensora_repo", type=str, required=True, help="path to your local Open-Sora checkout")
    ap.add_argument("--opensora_ckpt", type=str, required=True)

    # Data/adapters
    ap.add_argument("--train_jsonl", type=str, default="")
    ap.add_argument("--adapter_root", type=str, required=True)

    # Training hyperparams
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lambda_tox", type=float, default=1.0)
    ap.add_argument("--lambda_preserve", type=float, default=0.1)

    # Inference
    ap.add_argument("--prompt", type=str, default="")
    ap.add_argument("--out", type=str, default="out.mp4")
    ap.add_argument("--defense_scale", type=float, default=1.0)
    ap.add_argument("--force_category", type=str, default="")

    # Video gen params
    ap.add_argument("--num_frames", type=int, default=49)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--width", type=int, default=848)
    ap.add_argument("--num_steps", type=int, default=30)
    ap.add_argument("--guidance", type=float, default=7.5)
    ap.add_argument("--seed", type=int, default=0)

    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) Load Open-Sora + T5
    components = load_opensora_components(args.opensora_repo, args.opensora_ckpt, device=device)
    tokenizer = components["tokenizer"]
    text_encoder = components["text_encoder"]
    text_encoder.to(device).eval()

    # 2) Build adapter bank on T5
    bank = T5AdapterBank(base_t5=text_encoder, adapter_root=args.adapter_root, device=device)

    # 3) Load classifier (replace with your real one)
    # IMPORTANT: change hidden size in ToxicClassifier if your T5 dim != 4096.
    classifier = ToxicClassifier(num_classes=len(HARM_CATEGORIES)).to(device).eval()

    if args.mode == "train":
        assert args.train_jsonl, "--train_jsonl required in train mode"
        ds = PromptLabelDataset(args.train_jsonl)
        dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_batch, num_workers=2)

        os.makedirs(args.adapter_root, exist_ok=True)

        for cat in HARM_CATEGORIES:
            spec = AdapterSpec(name=cat, out_dir=args.adapter_root)
            train_one_adapter(
                components=components,
                adapter_bank=bank,
                classifier=classifier,
                spec=spec,
                train_loader=dl,
                device=device,
                epochs=args.epochs,
                lr=args.lr,
                lambda_tox=args.lambda_tox,
                lambda_preserve=args.lambda_preserve,
            )

    else:  # infer
        # load all adapters
        for cat in HARM_CATEGORIES:
            spec = AdapterSpec(name=cat, out_dir=args.adapter_root)
            bank.add_adapter(spec)
            adapter_dir = os.path.join(args.adapter_root, cat)
            if os.path.isdir(adapter_dir):
                bank.load_adapter(cat, adapter_dir, spec)
            else:
                print(f"[warn] adapter dir not found: {adapter_dir} (skip load)")

        force = args.force_category.strip() or None
        infer_with_dynamic_adapter(
            components=components,
            adapter_bank=bank,
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
