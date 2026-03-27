import torch
from cs336_basics.transformer.softmax import Softmax
from einops import einsum

def Attention(query, key, value, mask=None):
    """
    实现缩放点积注意力机制。

    参数:
        query: 形状 (batch_size, ..., seq_len, d_k) 的张量
        key:   形状 (batch_size, ..., seq_len, d_k) 的张量
        value: 形状 (batch_size, ..., seq_len, d_v) 的张量
        mask:  可选，形状 (seq_len, seq_len) 的布尔张量
               True 表示允许注意力，False 表示掩码（注意力设为 0）

    返回:
        输出: 形状 (batch_size, ..., seq_len, d_v) 的张量
    """
    d_k = query.size(-1)

    # 1. 计算 Q 与 K 的点积 (Q @ K^T)
    # 利用矩阵乘法，key 的最后两维转置
    scores = einsum(query, key, "... seq_len_q d_k, ... seq_len_k d_k ->... seq_len_q seq_len_k")  # (..., seq_len_q, seq_len_k)

    # 2. 缩放
    scores = scores / (d_k ** 0.5)

    # 3. 应用掩码（如果提供）
    if mask is not None:
        # mask 形状为 (seq_len_q, seq_len_k)，需要广播到 scores 的形状
        # 将 mask 中为 False 的位置设为负无穷，这样 softmax 后对应概率为 0
        scores = scores.masked_fill(~mask, float('-inf'))

    # 4. 在最后一个维度（key 的序列长度）上做 softmax
    attention_weights = Softmax(scores, dim=-1)  # (..., seq_len_q, seq_len_k)

    # 5. 用注意力权重对 value 加权求和
    output = einsum(attention_weights, value, "... seq_len_q seq_len_k, ... seq_len_k d_v -> ... seq_len_q d_v")  # (..., seq_len_q, d_v)

    return output