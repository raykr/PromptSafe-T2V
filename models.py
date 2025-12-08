import torch
import torch.nn as nn
import math


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


class AdapterRouter(nn.Module):
    """
    多 adapter 路由器：
      - 内部有若干个 SafeAdapter，每个对应一个有害类别
      - 推理时通过 set_route(category, scale) 选择一个 adapter + 防御强度
    """
    def __init__(self, adapters: dict[str, SafeAdapter]):
        super().__init__()
        # {'sexual': adapter_sexual, 'violent': adapter_violent, ...}
        self.adapters = nn.ModuleDict(adapters)
        self.current_category: str | None = None
        self.current_scale: torch.Tensor | float = 1.0

    def set_route(self, category: str | None, scale: torch.Tensor | float = 1.0):
        """
        设置当前使用哪一个 adapter，以及防御强度 scale
        category:
          - 为 None 时表示不使用任何 adapter（完全关闭防御）
          - 为某个 key 时使用对应的 adapter
        scale:
          - float 或者 [B] Tensor
        """
        self.current_category = category
        self.current_scale = scale

    def forward(self, hidden_states: torch.Tensor):
        if self.current_category is None:
            # 不进行任何防御
            return hidden_states
        if self.current_category not in self.adapters:
            # 防御类别未注册，退化为 no-op
            return hidden_states

        adapter = self.adapters[self.current_category]
        return adapter(hidden_states, scale=self.current_scale)


class WrappedTextEncoderRouter(nn.Module):
    def __init__(self, t5_encoder: nn.Module, router: AdapterRouter):
        super().__init__()
        self.t5 = t5_encoder
        for p in self.t5.parameters():
            p.requires_grad_(False)
        self.router = router

    # 对外暴露设置路由的接口
    def set_adapter_route(self, category: str | None, scale: torch.Tensor | float = 1.0):
        self.router.set_route(category, scale)

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
        hs = outputs.last_hidden_state  # [B,L,D]
        hs_safe = self.router(hs)       # 根据当前 route 选择一个 adapter + scale
        outputs.last_hidden_state = hs_safe
        return outputs
