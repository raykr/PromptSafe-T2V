# compare_text_encoder.py
import os
import argparse
import json
import hashlib
from typing import Dict, Tuple, List

import torch
from transformers import AutoTokenizer, T5EncoderModel


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def tensor_sha256(t: torch.Tensor) -> str:
    """
    对 tensor 做稳定 hash：转 CPU、contiguous、转 uint8 bytes
    注意：这里对原始 dtype 的 bytes hash；不同 dtype 会导致 hash 不同。
    """
    t = t.detach().cpu().contiguous()
    return sha256_bytes(t.numpy().tobytes())


def state_dict_fingerprint(
    state: Dict[str, torch.Tensor],
    max_tensors: int = 0,
    key_filter: str = "",
) -> Tuple[str, List[Tuple[str, Tuple[int, ...], str]]]:
    """
    返回:
      - overall_hash: 对( name + shape + tensor_bytes_hash ) 串联后的 sha256
      - details: [(name, shape, tensor_hash)]，可选抽样输出
    参数:
      - max_tensors=0 表示全部纳入 overall hash；>0 表示只抽样前 max_tensors 个参与（不建议用于严格判定）
      - key_filter: 只对包含该子串的参数做 fingerprint（例如 "shared" 或 "encoder.block.0"）
    """
    items = []
    for k in sorted(state.keys()):
        if key_filter and (key_filter not in k):
            continue
        v = state[k]
        items.append((k, v))

    if max_tensors > 0:
        items = items[:max_tensors]

    details = []
    h = hashlib.sha256()
    for k, v in items:
        v_hash = tensor_sha256(v)
        shape = tuple(v.shape)
        details.append((k, shape, v_hash))
        h.update(k.encode("utf-8"))
        h.update(str(shape).encode("utf-8"))
        h.update(v_hash.encode("utf-8"))

    overall = h.hexdigest()
    return overall, details


def compare_tokenizers(tok_a, tok_b, prompts: List[str]) -> Dict:
    info = {}

    # 基本属性
    info["name_or_path"] = (str(tok_a.name_or_path), str(tok_b.name_or_path))
    info["vocab_size"] = (tok_a.vocab_size, tok_b.vocab_size)
    info["model_max_length"] = (int(tok_a.model_max_length), int(tok_b.model_max_length))

    # special tokens
    info["special_tokens_map"] = (tok_a.special_tokens_map, tok_b.special_tokens_map)

    # 同一批 prompts 的 input_ids 是否一致
    enc_a = tok_a(prompts, padding=True, truncation=True, return_tensors="pt")
    enc_b = tok_b(prompts, padding=True, truncation=True, return_tensors="pt")

    # 检查形状是否相同
    shape_match = enc_a["input_ids"].shape == enc_b["input_ids"].shape
    
    if shape_match:
        same_input_ids = torch.equal(enc_a["input_ids"], enc_b["input_ids"])
        same_attention = torch.equal(enc_a["attention_mask"], enc_b["attention_mask"])
    else:
        # 形状不同，逐个比较每个prompt
        same_input_ids = False
        same_attention = False
        info["shape_mismatch"] = {
            "shape_a": list(enc_a["input_ids"].shape),
            "shape_b": list(enc_b["input_ids"].shape)
        }
        # 逐个prompt比较
        per_prompt_match = []
        for i in range(len(prompts)):
            # 分别编码单个prompt（不padding，看实际长度）
            enc_a_single = tok_a(prompts[i], padding=False, truncation=True, return_tensors="pt")
            enc_b_single = tok_b(prompts[i], padding=False, truncation=True, return_tensors="pt")
            ids_a = enc_a_single["input_ids"][0]
            ids_b = enc_b_single["input_ids"][0]
            match = (ids_a.shape == ids_b.shape) and torch.equal(ids_a, ids_b)
            per_prompt_match.append({
                "prompt_idx": i,
                "prompt": prompts[i],
                "match": bool(match),
                "len_a": int(ids_a.shape[0]),
                "len_b": int(ids_b.shape[0]),
                "ids_a": ids_a.tolist() if len(ids_a) <= 20 else ids_a.tolist()[:20],  # 只显示前20个
                "ids_b": ids_b.tolist() if len(ids_b) <= 20 else ids_b.tolist()[:20],
            })
        info["per_prompt_comparison"] = per_prompt_match
    
    info["same_input_ids"] = bool(same_input_ids) if shape_match else False
    info["same_attention_mask"] = bool(same_attention) if shape_match else False

    # 若不一致且形状相同，给出首个差异位置
    if shape_match and not same_input_ids:
        diff = (enc_a["input_ids"] != enc_b["input_ids"]).nonzero(as_tuple=False)
        if len(diff) > 0:
            first = diff[0].tolist()
            b, pos = first[0], first[1]
            info["first_diff_input_ids"] = {
                "batch_idx": b,
                "pos": pos,
                "tokA": int(enc_a["input_ids"][b, pos].item()),
                "tokB": int(enc_b["input_ids"][b, pos].item()),
                "prompt": prompts[b],
            }

    return info


@torch.no_grad()
def compare_encoder_outputs(
    enc_a: torch.nn.Module,
    enc_b: torch.nn.Module,
    tok_a,
    tok_b,
    prompts: List[str],
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> Dict:
    """
    将两个 encoder 放到同一 device/dtype 下比较输出（last_hidden_state）
    """
    enc_a = enc_a.to(device=device, dtype=dtype).eval()
    enc_b = enc_b.to(device=device, dtype=dtype).eval()

    batch_a = tok_a(prompts, padding=True, truncation=True, return_tensors="pt").to(device)
    batch_b = tok_b(prompts, padding=True, truncation=True, return_tensors="pt").to(device)

    out_a = enc_a(input_ids=batch_a["input_ids"], attention_mask=batch_a["attention_mask"]).last_hidden_state
    out_b = enc_b(input_ids=batch_b["input_ids"], attention_mask=batch_b["attention_mask"]).last_hidden_state

    # 要比较输出，前提是 input_ids 相同，否则没有可比性
    same_inputs = torch.equal(batch_a["input_ids"], batch_b["input_ids"]) and torch.equal(batch_a["attention_mask"], batch_b["attention_mask"])

    stats = {
        "same_inputs_for_output_compare": bool(same_inputs),
        "shape_a": tuple(out_a.shape),
        "shape_b": tuple(out_b.shape),
    }

    if same_inputs and out_a.shape == out_b.shape:
        diff = (out_a - out_b).abs()
        stats["max_abs_diff"] = float(diff.max().item())
        stats["mean_abs_diff"] = float(diff.mean().item())
        # 一个相对误差量纲（防止数值整体很小/很大）
        denom = out_a.abs().mean().clamp_min(1e-8)
        stats["mean_rel_diff"] = float((diff.mean() / denom).item())
    else:
        stats["max_abs_diff"] = None
        stats["mean_abs_diff"] = None
        stats["mean_rel_diff"] = None

    return stats


def load_tok_enc(model_path: str):
    tok = AutoTokenizer.from_pretrained(model_path, subfolder="tokenizer")
    enc = T5EncoderModel.from_pretrained(model_path, subfolder="text_encoder")
    return tok, enc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cog_path", type=str, default="/home/raykr/models/zai-org/CogVideoX-2b")
    parser.add_argument("--wan_path", type=str, default="/home/raykr/models/Wan-AI/Wan2.1-T2V-1.3B-Diffusers")
    parser.add_argument("--device", type=str, default="cpu", help="用于输出一致性比较的设备(cpu/cuda)")
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--skip_output_compare", action="store_true", help="只比较权重与tokenizer，不跑encoder输出")
    args = parser.parse_args()

    dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    dtype = dtype_map[args.dtype]

    prompts = [
        "A cat playing piano in a cozy room.",
        "A violent scene with blood and injury.",  # 故意放一个敏感prompt，但这里只做编码对比，不生成视频
        "A political leader giving a speech at a rally.",
        "A surreal horror scene with grotesque creatures.",
    ]

    print("=" * 80)
    print("Loading CogVideoX tokenizer/text_encoder...")
    tok_cog, enc_cog = load_tok_enc(args.cog_path)

    print("Loading Wan2.1 tokenizer/text_encoder...")
    tok_wan, enc_wan = load_tok_enc(args.wan_path)

    print("=" * 80)
    print("[1] Tokenizer comparison")
    tok_info = compare_tokenizers(tok_cog, tok_wan, prompts)
    print(json.dumps(tok_info, ensure_ascii=False, indent=2))

    print("=" * 80)
    print("[2] text_encoder config comparison (key fields)")
    cfg_cog = enc_cog.config.to_dict()
    cfg_wan = enc_wan.config.to_dict()

    # 只输出一些最关键字段
    keys = [
        "model_type", "d_model", "num_layers", "num_heads",
        "d_ff", "vocab_size", "dropout_rate",
        "is_encoder_decoder", "tie_word_embeddings",
    ]
    cfg_cmp = {}
    for k in keys:
        cfg_cmp[k] = (cfg_cog.get(k, None), cfg_wan.get(k, None))
    print(json.dumps(cfg_cmp, ensure_ascii=False, indent=2))

    print("=" * 80)
    print("[3] text_encoder state_dict name/shape quick check")
    sd_cog = enc_cog.state_dict()
    sd_wan = enc_wan.state_dict()

    keys_cog = sorted(sd_cog.keys())
    keys_wan = sorted(sd_wan.keys())

    same_keys = (keys_cog == keys_wan)
    print(f"same_param_keys = {same_keys} (cog={len(keys_cog)}, wan={len(keys_wan)})")

    # shape 对齐检查
    shape_mismatch = []
    if same_keys:
        for k in keys_cog:
            if tuple(sd_cog[k].shape) != tuple(sd_wan[k].shape):
                shape_mismatch.append((k, tuple(sd_cog[k].shape), tuple(sd_wan[k].shape)))
    else:
        # 如果 keys 不同，也做一个交集上的 shape 检查，便于诊断
        common = sorted(set(keys_cog).intersection(set(keys_wan)))
        for k in common:
            if tuple(sd_cog[k].shape) != tuple(sd_wan[k].shape):
                shape_mismatch.append((k, tuple(sd_cog[k].shape), tuple(sd_wan[k].shape)))

    print(f"num_shape_mismatch = {len(shape_mismatch)}")
    if shape_mismatch:
        print("first 10 shape mismatches:")
        for item in shape_mismatch[:10]:
            print("  ", item)

    print("=" * 80)
    print("[4] text_encoder weight content hash (STRICT)")
    # 严格判定：参与全部参数、同时包含 name/shape/bytes_hash 的整体 fingerprint
    fp_cog, _ = state_dict_fingerprint(sd_cog, max_tensors=0)
    fp_wan, _ = state_dict_fingerprint(sd_wan, max_tensors=0)
    print(f"cog_fingerprint = {fp_cog}")
    print(f"wan_fingerprint = {fp_wan}")
    print(f"same_weight_fingerprint = {fp_cog == fp_wan}")

    if not args.skip_output_compare:
        print("=" * 80)
        print("[5] Encoder output comparison (final confirmation)")
        out_stats = compare_encoder_outputs(
            enc_cog, enc_wan, tok_cog, tok_wan, prompts,
            device=args.device, dtype=dtype
        )
        print(json.dumps(out_stats, ensure_ascii=False, indent=2))

    print("=" * 80)
    # 最终结论
    same_tokenizer = tok_info["same_input_ids"] and tok_info["same_attention_mask"] and \
                     (tok_info["vocab_size"][0] == tok_info["vocab_size"][1]) and \
                     (tok_info["model_max_length"][0] == tok_info["model_max_length"][1])

    same_encoder = (fp_cog == fp_wan)

    if same_tokenizer and same_encoder:
        verdict = "✅ 结论：CogVideoX 与 Wan2.1 的 tokenizer + text_encoder 权重完全相同（可视为同一个 encoder）。"
    elif same_encoder and (not same_tokenizer):
        verdict = "⚠️ 结论：text_encoder 权重相同，但 tokenizer 行为/配置不同（需要谨慎，建议统一 tokenizer）。"
    elif (not same_encoder) and same_tokenizer:
        verdict = "⚠️ 结论：tokenizer 相同，但 text_encoder 权重不同（可跑但属于跨表征空间迁移，效果不保证）。"
    else:
        verdict = "❌ 结论：tokenizer 与 text_encoder 权重都不同（属于跨 tokenizer+跨 encoder 迁移）。"

    print(verdict)


if __name__ == "__main__":
    main()
