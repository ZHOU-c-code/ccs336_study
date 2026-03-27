import torch
import torch.nn as nn
from einops import einsum

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        """
        Root Mean Square Layer Normalization (RMSNorm)
        
        Args:
            d_model: 模型的隐藏维度
            eps: 数值稳定性用的极小值
            device: 参数存储设备
            dtype: 参数数据类型
        """
        super().__init__()
        
        # 可学习的增益参数 g_i，形状为 (d_model,)
        self.gain = nn.Parameter(
            torch.ones(d_model, device=device, dtype=dtype)
        )
        
        # 保存超参数
        self.d_model = d_model
        self.eps = eps
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        RMSNorm前向传播
        
        Args:
            x: 输入张量，形状为 (batch_size, sequence_length, d_model)
               或任意形状，只要最后一维是 d_model
        
        Returns:
            归一化后的张量，形状与输入相同
        """
        # 保存原始数据类型，用于最后的恢复
        original_dtype = x.dtype
        
        # 上采样到float32防止计算平方时溢出
        x = x.to(torch.float32)
        
        # 使用 einsum 计算每个位置所有特征维度的平方和
        # 输入形状: (..., d_model) -> 输出形状: (...,)
        sum_sq = einsum(x, x, '... d_model, ... d_model -> ...',)  # 等价于 (x * x).sum(dim=-1)

        # 计算 RMS = sqrt(mean_sq + eps)
        mean_sq = sum_sq / self.d_model
        rms = torch.sqrt(mean_sq + self.eps)

        # 扩展维度以便广播
        rms = rms.unsqueeze(-1)  # 形状: (..., 1)

        # 使用 einsum 进行归一化并乘以增益
        # 归一化: x / rms，这里直接使用除法，因为 einsum 不适合逐元素除
        normalized = x / rms

        # 用 einsum 乘以增益: normalized * gain
        # 增益形状 (d_model,) -> 广播到 normalized 的最后一维
        result = einsum(normalized, self.gain, '... d_model, d_model -> ... d_model')

        
        # 恢复原始数据类型并返回
        return result.to(original_dtype)