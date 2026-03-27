import torch
import torch.nn as nn
import math
from einops import einsum

class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        """
        线性变换模块，不带偏置项
        
        Args:
            in_features: 输入特征维度
            out_features: 输出特征维度
            device: 参数存储设备
            dtype: 参数数据类型
        """
        super().__init__()
        
        # 创建权重参数（注意：存储的是W，不是W^T）
        self.W = nn.Parameter(
            torch.empty(out_features, in_features, device=device, dtype=dtype)
        )
        std=math.sqrt(2.0/(in_features + out_features))
        # 使用截断正态分布初始化权重
        nn.init.trunc_normal_(self.W, mean=0.0, std=std, a=-3*std, b=3*std)
        
        # 保存输入输出维度（可选，用于信息记录）
        self.in_features = in_features
        self.out_features = out_features
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        return einsum(self.W, x, "out_features in_features, ... in_features -> ... out_features")