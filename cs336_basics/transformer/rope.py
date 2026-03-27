import torch
import torch.nn as nn
from einops import rearrange

class RoPE(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        
        assert d_k % 2 == 0, f"d_k must be even, got {d_k}"
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        
        # 预计算所有位置的cos和sin
        # 但保持维度为 (max_seq_len, d_k/2, 2)
        # 这样可以直接用于旋转
        freqs = 1.0 / (theta ** (torch.arange(0, d_k, 2, device=device).float() / d_k))
        positions = torch.arange(max_seq_len, device=device).float()
        angles = torch.outer(positions, freqs)  # (max_seq_len, d_k//2)

        self.register_buffer('cos', torch.cos(angles), persistent=False)  # (max_seq_len, d_k//2)
        self.register_buffer('sin', torch.sin(angles), persistent=False)
    
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        x1, x2 = rearrange(x, '... seq (dim2 two) -> ... seq dim2 two', two=2).unbind(dim=-1)

        cos = self.cos[token_positions]  # (..., seq, dim2)
        sin = self.sin[token_positions]  # (..., seq, dim2)

        # 旋转
        y1 = x1 * cos - x2 * sin
        y2 = x2 * cos + x1 * sin

        # 重新组合并展平
        return rearrange([y1, y2], 'two ... seq dim2 -> ... seq (dim2 two)')