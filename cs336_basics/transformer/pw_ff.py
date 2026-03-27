import torch
import torch.nn as nn
import math
from cs336_basics.transformer.linear import Linear
from einops import einsum
from typing import Optional

class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: Optional[int] = None, device=None, dtype=None):
        """
        SwiGLU Feed-Forward Network
        
        Args:
            d_model: 模型维度
            device: 参数存储设备
            dtype: 参数数据类型
        """
        super().__init__()
        
        if d_ff is None:
            # 自动计算并调整为 64 的倍数（原逻辑）
            theoretical_ff = (8/3) * d_model
            d_ff = int(math.ceil(theoretical_ff / 64) * 64)
        
        self.d_ff = d_ff
        self.d_model = d_model
        
        # SwiGLU 需要三个线性变换：
        # 1. W1: 将输入从 d_model 投影到 d_ff（用于门控机制的主路径）
        # 2. W2: 将输入从 d_model 投影到 d_ff（用于门控机制的sigmoid路径）
        # 3. W3: 将门控后的结果从 d_ff 投影回 d_model
        
        self.W1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.W3 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.W2 = Linear(d_ff, d_model, device=device, dtype=dtype)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        SwiGLU前向传播
        
        Args:
            x: 输入张量，形状为 (batch_size, sequence_length, d_model)
               或任意形状，只要最后一维是 d_model
        
        Returns:
            输出张量，形状与输入相同 (..., d_model)
        """
        # SwiGLU = SiLU(x @ W1) * (x @ W3) 然后投影回 d_model
        

        main_path = torch.nn.functional.silu(self.W1(x))
        
        gate_path = self.W3(x)
        
        # 门控机制：主路径 * 门控路径
        gated = einsum(main_path, gate_path, "... d_ff, ... d_ff -> ... d_ff")
        
        # 投影回 d_model
        output = self.W2(gated)
        
        return output