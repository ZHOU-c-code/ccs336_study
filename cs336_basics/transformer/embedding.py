import torch
import torch.nn as nn

class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        """
        嵌入层模块：将整数token ID映射为稠密向量
        
        Args:
            num_embeddings: 词汇表大小（token ID的数量）
            embedding_dim: 嵌入向量的维度 (d_model)
            device: 参数存储设备
            dtype: 参数数据类型
        """
        super().__init__()
        
        # 创建嵌入矩阵，形状为 (vocab_size, d_model)
        # 注意：d_model 是最后一个维度，符合要求
        self.embedding_matrix = nn.Parameter(
            torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype)
        )
        
        # 使用截断正态分布初始化，均值为0，标准差为1，截断范围[-3, 3]
        nn.init.trunc_normal_(self.embedding_matrix, mean=0.0, std=1.0, a=-3.0, b=3.0)
        
        # 保存参数（可选）
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        通过索引查找token对应的嵌入向量
        
        Args:
            token_ids: 整数token ID张量，形状为 (batch_size, sequence_length)
                      或任意形状，但必须包含整数类型的token ID
        
        Returns:
            嵌入向量张量，形状为 (batch_size, sequence_length, embedding_dim)
            或 (*token_ids.shape, embedding_dim)
        """
        # 使用整数索引从嵌入矩阵中查找对应的向量
        # token_ids 中的每个整数都会从 embedding_matrix 中取出对应行
        embeddings = self.embedding_matrix[token_ids]
        
        return embeddings