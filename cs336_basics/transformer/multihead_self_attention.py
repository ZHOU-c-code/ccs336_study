import torch
import torch.nn as nn
from cs336_basics.transformer.linear import Linear
from cs336_basics.transformer.rope import RoPE
from cs336_basics.transformer.attention import Attention
from einops import rearrange

class CausalMultiHeadSelfAttention(nn.Module):
    """
    因果多头自注意力模块，遵循 Vaswani et al. (2017) 的设计。
    使用已有的 RoPE 和缩放点积注意力函数。
    """
    def __init__(self, d_model: int, num_heads: int, theta: float= 10000.0, max_seq_len: int= 2048, use_rope: bool = True):
        """
        Args:
            d_model: 输入/输出特征维度。
            num_heads: 注意力头数。
            bias: 线性层是否使用偏置。
        """
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads   # 每个头的维度，d_k = d_v
        self.use_rope = use_rope

        # 定义 Q、K、V 的投影层
        self.combined_QKV = Linear(d_model, 3 * d_model)
        # 输出投影层
        self.W_O = Linear(d_model, d_model)
        if use_rope:
            self.rope = RoPE(theta, self.d_k, max_seq_len)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor = None) -> torch.Tensor:
        *leading_dims, seq_len, _ = x.shape

        # 1. 线性投影得到 Q、K、V
        combined = self.combined_QKV(x)  # (batch, seq_len, 3*d_model)

        qkv = rearrange(combined, '... seq (three d) -> ... seq three d', three=3, d=self.d_model)
        Q, K, V = qkv[..., 0, :], qkv[..., 1, :], qkv[..., 2, :]

        # 3. 重塑为多头格式: (batch, seq_len, num_heads, d_k) -> (batch, num_heads, seq_len, d_k)
        Q = rearrange(Q, '... seq (h d) -> ... h seq d', h=self.num_heads, d=self.d_k)
        K = rearrange(K, '... seq (h d) -> ... h seq d', h=self.num_heads, d=self.d_k)
        V = rearrange(V, '... seq (h d) -> ... h seq d', h=self.num_heads, d=self.d_k)

        # 4. 应用 RoPE
        if self.use_rope:
            if token_positions is None:
                positions = torch.arange(seq_len, device=x.device)
                if leading_dims:
                    # 扩展为 (1,...,1, seq_len) 然后广播到 (leading_dims..., seq_len)
                    view_shape = [1] * len(leading_dims) + [seq_len]
                    expand_shape = leading_dims + [seq_len]
                    token_positions = positions.view(*view_shape).expand(*expand_shape)
                else:
                    token_positions = positions

            token_positions = token_positions[..., None, :].expand(*leading_dims, self.num_heads, seq_len)
            token_positions_flat = token_positions.reshape(-1, seq_len)
            Q_flat = Q.reshape(-1, seq_len, self.d_k)   # (total_heads, seq, d_k)
            K_flat = K.reshape(-1, seq_len, self.d_k)

            Q_rot = self.rope(Q_flat, token_positions_flat)
            K_rot = self.rope(K_flat, token_positions_flat)

            Q = Q_rot.view(*leading_dims, self.num_heads, seq_len, self.d_k)
            K = K_rot.view(*leading_dims, self.num_heads, seq_len, self.d_k)
        
        # 4. 构建因果掩码（下三角，对角线及以下为 True 表示允许注意力）
        mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=0)

        # 5. 调用缩放点积注意力（假设函数接受 Q、K、V 和 mask，返回输出）
        # 注意：scaled_dot_product_attention 应能处理 mask 广播，形状 (batch, num_heads, seq_len, seq_len)
        attn_out = Attention(Q, K, V, mask=mask)  # (batch, num_heads, seq_len, d_k)

        # 6. 将多头输出拼接回原始维度
        attn_out = rearrange(attn_out, '... h seq d -> ... seq (h d)')

        # 7. 输出投影
        output = self.W_O(attn_out)  # (batch, seq_len, d_model)

        return output