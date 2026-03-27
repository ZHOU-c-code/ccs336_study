import torch.nn as nn
from cs336_basics.transformer.rmsnorm import RMSNorm
from cs336_basics.transformer.multihead_self_attention import CausalMultiHeadSelfAttention
from cs336_basics.transformer.pw_ff import SwiGLU

class TransformerBlock(nn.Module):
    """
    Pre-Norm Transformer block consisting of:
        1. Causal multi-head self-attention sublayer with RMSNorm and residual connection.
        2. Feed-forward sublayer (SwiGLU) with RMSNorm and residual connection.
    """
    def __init__(self, d_model: int, num_heads: int, d_ff: int,theta: float, max_seq_len: int):
        super().__init__()
        # 第一个子层：RMSNorm + CausalMultiHeadSelfAttention
        self.norm1 = RMSNorm(d_model)
        self.attn = CausalMultiHeadSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            theta=theta,
            max_seq_len=max_seq_len,
            use_rope=True          # 默认使用 RoPE
        )
        
        # 第二个子层：RMSNorm + SwiGLU
        self.norm2 = RMSNorm(d_model)
        self.ff = SwiGLU(
            d_model=d_model,
            d_ff=d_ff              # 显式传入前馈中间层维度
        )

    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, d_model) 输入张量

        Returns:
            (batch_size, seq_len, d_model) 经过两个子层处理后的输出
        """
        # 第一个子层：残差连接 + 因果多头自注意力（Pre-Norm）
        x = x + self.attn(self.norm1(x))

        # 第二个子层：残差连接 + 前馈网络（Pre-Norm）
        x = x + self.ff(self.norm2(x))

        return x