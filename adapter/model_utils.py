import torch
import torch.nn as nn
import math


# class SafeAdapter(nn.Module):
#     """LoRA-like Adapter for T5 Encoder"""
#     def __init__(self, hidden_size=4096, rank=256):
#         super().__init__()
#         self.down = nn.Linear(hidden_size, rank, bias=False)
#         self.up = nn.Linear(rank, hidden_size, bias=False)
#         nn.init.normal_(self.down.weight, std=1e-4)
#         nn.init.zeros_(self.up.weight)

#     def forward(self, x):
#         return x + self.up(self.down(x))

class SafeAdapter(nn.Module):
    """
    冻结 T5 输出后，做一个小瓶颈 + 残差的安全映射层
    H_safe = H + gate * MLP(LN(H))
    """
    def __init__(self, hidden_size: int, rank: int = 256, init_gate: float = 0.5):
        super().__init__()
        self.ln = nn.LayerNorm(hidden_size)
        self.down = nn.Linear(hidden_size, rank, bias=False)
        self.up = nn.Linear(rank, hidden_size, bias=False)
        self.act = nn.GELU()
        # 可学习 gate（也可以后续做成动态门控）
        self.gate = nn.Parameter(torch.tensor(init_gate))

        # 小初始化，避免一开始破坏分布
        nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.up.weight)

    def forward(self, hidden_states: torch.Tensor):
        x = self.ln(hidden_states)
        delta = self.up(self.act(self.down(x)))
        return hidden_states + self.gate * delta


class WrappedTextEncoder(nn.Module):
    """封装T5Encoder + Adapter"""
    def __init__(self, t5_encoder: nn.Module, adapter: SafeAdapter):
        super().__init__()
        self.t5 = t5_encoder
        for p in self.t5.parameters():
            p.requires_grad_(False)
        self.adapter = adapter

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
        hs = outputs.last_hidden_state
        # 确保数据类型匹配
        if hs.dtype != self.adapter.ln.weight.dtype:
            hs = hs.to(self.adapter.ln.weight.dtype)
        hs_safe = self.adapter(hs)
        outputs.last_hidden_state = hs_safe
        return outputs
