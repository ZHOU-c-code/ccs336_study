import torch

def Softmax(tensor, dim):
    """
    对 PyTorch 张量的指定维度应用数值稳定的 softmax。
    """
    # 沿 dim 取最大值，keepdim=True 保持维度以便广播
    max_vals, _ = torch.max(tensor, dim=dim, keepdim=True)
    # 减去最大值（数值稳定）
    shifted = tensor - max_vals
    # 指数运算
    exp_vals = torch.exp(shifted)
    # 求和并归一化
    sum_vals = torch.sum(exp_vals, dim=dim, keepdim=True)
    return exp_vals / sum_vals