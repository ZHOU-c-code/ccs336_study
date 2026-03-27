import torch

# ---------- 模型配置（GPT‑2 风格） ----------
models = {
    "small":  {"layers": 12, "d_model": 768,  "d_ff": 3072, "vocab": 50257, "heads": 12},
    "medium": {"layers": 24, "d_model": 1024, "d_ff": 4096, "vocab": 50257, "heads": 16},
    "large":  {"layers": 36, "d_model": 1280, "d_ff": 5120, "vocab": 50257, "heads": 20},
    "xl":     {"layers": 48, "d_model": 1600, "d_ff": 6400, "vocab": 50257, "heads": 25},   # head_dim = 64
}

# ---------- 参数计数 ----------
def compute_params(d_model, num_layers, d_ff, vocab_size):
    # 词嵌入
    token_emb = vocab_size * d_model
    # 无位置嵌入参数
    pos_emb = 0

    # 每层参数
    per_layer_attn = 4 * d_model * d_model          # QKV (3*d_model*d_model) + W_O (d_model*d_model)
    per_layer_ffn  = 3 * d_model * d_ff             # SwiGLU: W1, W3, W2
    per_layer_norm = 2 * d_model                    # 两个 RMSNorm，每个 d_model
    per_layer_params = per_layer_attn + per_layer_ffn + per_layer_norm
    total_layer_params = per_layer_params * num_layers

    # 最终 RMSNorm
    final_norm = d_model

    # 输出投影（独立于词嵌入）
    out_proj = d_model * vocab_size

    total_params = token_emb + pos_emb + total_layer_params + final_norm + out_proj
    return total_params

# ---------- FLOPs 计算 ----------
def compute_flops(batch_size, seq_len, d_model, d_ff, num_heads, num_layers, vocab_size):
    """
    返回总 FLOPs 及按组件分解的字典（每层或整体）
    """
    # 注意：head_dim = d_model // num_heads
    head_dim = d_model // num_heads

    # 1. 每层自注意力部分
    # QKV 投影（一个 Linear 从 d_model 到 3*d_model）
    qkv_flops = 2 * batch_size * seq_len * d_model * (3 * d_model)   # = 6 B S d_model^2
    # 注意力分数 Q @ K^T
    attn_score_flops = 2 * batch_size * seq_len * seq_len * d_model
    # 注意力加权和 attn @ V
    attn_weighted_flops = 2 * batch_size * seq_len * seq_len * d_model
    # 输出投影 W_O
    out_proj_attn_flops = 2 * batch_size * seq_len * d_model * d_model

    per_layer_attn_flops = qkv_flops + attn_score_flops + attn_weighted_flops + out_proj_attn_flops

    # 2. 每层 FFN (SwiGLU)
    # W1, W3, W2 各一次矩阵乘
    ffn_flops = 3 * (2 * batch_size * seq_len * d_model * d_ff)   # = 6 B S d_model d_ff

    per_layer_total = per_layer_attn_flops + ffn_flops
    total_layers_flops = per_layer_total * num_layers

    # 3. 最终输出投影
    final_out_flops = 2 * batch_size * seq_len * d_model * vocab_size

    total_flops = total_layers_flops + final_out_flops

    # 分解字典（每层值均为该层内的 FLOPs）
    breakdown = {
        "QKV投影 (每层)": qkv_flops,
        "注意力分数 (每层)": attn_score_flops,
        "注意力加权和 (每层)": attn_weighted_flops,
        "注意力输出投影 (每层)": out_proj_attn_flops,
        "FFN (每层)": ffn_flops,
        "最终输出投影": final_out_flops,
    }
    return total_flops, breakdown

def format_flops(flops):
    if flops >= 1e12:
        return f"{flops / 1e12:.2f} TFLOPs"
    elif flops >= 1e9:
        return f"{flops / 1e9:.2f} GFLOPs"
    elif flops >= 1e6:
        return f"{flops / 1e6:.2f} MFLOPs"
    else:
        return f"{flops:.0f} FLOPs"

# ---------- 执行分析 ----------
BATCH_SIZE = 1
SEQ_LEN = 1024            # 默认上下文长度
cfg = models["xl"]
d_model = cfg["d_model"]
num_layers = cfg["layers"]
d_ff = cfg["d_ff"]
vocab_size = cfg["vocab"]
num_heads = cfg["heads"]

# (a) 参数量与内存
params = compute_params(d_model, num_layers, d_ff, vocab_size)
mem_bytes = params * 4
mem_gb = mem_bytes / (1024**3)
print(f"(a) GPT-2 XL 可训练参数: {params:,}")
print(f"    内存占用 (单精度): {mem_gb:.2f} GB\n")

# (b) FLOPs 分解
total_flops, breakdown = compute_flops(BATCH_SIZE, SEQ_LEN, d_model, d_ff, num_heads, num_layers, vocab_size)
print("(b) 前向传播 FLOPs (seq_len=1024):")
for name, flops in breakdown.items():
    print(f"    {name}: {format_flops(flops)}")
print(f"    总 FLOPs: {format_flops(total_flops)}\n")

# (c) 最耗 FLOPs 组件
attn_layers_flops = (breakdown["QKV投影 (每层)"] + breakdown["注意力分数 (每层)"] +
                     breakdown["注意力加权和 (每层)"] + breakdown["注意力输出投影 (每层)"]) * num_layers
ffn_layers_flops = breakdown["FFN (每层)"] * num_layers
final_flops = breakdown["最终输出投影"]
attn_ratio = attn_layers_flops / total_flops * 100
ffn_ratio = ffn_layers_flops / total_flops * 100
final_ratio = final_flops / total_flops * 100
print("(c) 最耗 FLOPs 的组件:")
print(f"    自注意力部分 (所有层): {format_flops(attn_layers_flops)} ({attn_ratio:.1f}%)")
print(f"    FFN 部分 (所有层): {format_flops(ffn_layers_flops)} ({ffn_ratio:.1f}%)")
print(f"    最终输出投影: {format_flops(final_flops)} ({final_ratio:.1f}%)")
print("    结论：前馈网络（SwiGLU）占用了最多的 FLOPs。\n")

# (d) 不同尺寸占比变化
print("(d) 不同模型尺寸 FLOPs 占比:")
print(f"{'模型':<10} {'自注意力占比 (%)':<20} {'FFN占比 (%)':<20} {'最终投影占比 (%)':<20}")
for name, cfg in models.items():
    d_model = cfg["d_model"]
    num_layers = cfg["layers"]
    d_ff = cfg["d_ff"]
    vocab_size = cfg["vocab"]
    num_heads = cfg["heads"]
    total, _ = compute_flops(BATCH_SIZE, SEQ_LEN, d_model, d_ff, num_heads, num_layers, vocab_size)
    # 重新计算各部分 FLOPs（避免重复调用）
    qkv = 2 * BATCH_SIZE * SEQ_LEN * d_model * (3 * d_model)
    score = 2 * BATCH_SIZE * SEQ_LEN * SEQ_LEN * d_model
    weighted = 2 * BATCH_SIZE * SEQ_LEN * SEQ_LEN * d_model
    out = 2 * BATCH_SIZE * SEQ_LEN * d_model * d_model
    attn_per_layer = qkv + score + weighted + out
    ffn_per_layer = 3 * (2 * BATCH_SIZE * SEQ_LEN * d_model * d_ff)
    attn_total = attn_per_layer * num_layers
    ffn_total = ffn_per_layer * num_layers
    final = 2 * BATCH_SIZE * SEQ_LEN * d_model * vocab_size
    attn_ratio = attn_total / total * 100
    ffn_ratio = ffn_total / total * 100
    final_ratio = final / total * 100
    print(f"{name:<10} {attn_ratio:<20.1f} {ffn_ratio:<20.1f} {final_ratio:<20.1f}")
print("\n随着模型增大，FFN 占比略微上升，自注意力占比略微下降，最终投影占比明显下降。\n")

# (e) 增加上下文长度 (GPT-2 XL, seq_len=16384)
SEQ_LEN_LONG = 16384
total_long, _ = compute_flops(BATCH_SIZE, SEQ_LEN_LONG, d_model, d_ff, num_heads, num_layers, vocab_size)
print(f"(e) 增加上下文长度至 {SEQ_LEN_LONG}:")
print(f"    原 FLOPs (seq_len=1024): {format_flops(total_flops)}")
print(f"    新 FLOPs: {format_flops(total_long)}")
print(f"    增长倍数: {total_long / total_flops:.2f}")
# 重新计算长上下文下的占比
qkv_long = 2 * BATCH_SIZE * SEQ_LEN_LONG * d_model * (3 * d_model)
score_long = 2 * BATCH_SIZE * SEQ_LEN_LONG * SEQ_LEN_LONG * d_model
weighted_long = 2 * BATCH_SIZE * SEQ_LEN_LONG * SEQ_LEN_LONG * d_model
out_long = 2 * BATCH_SIZE * SEQ_LEN_LONG * d_model * d_model
attn_per_layer_long = qkv_long + score_long + weighted_long + out_long
ffn_per_layer_long = 3 * (2 * BATCH_SIZE * SEQ_LEN_LONG * d_model * d_ff)
attn_total_long = attn_per_layer_long * num_layers
ffn_total_long = ffn_per_layer_long * num_layers
final_long = 2 * BATCH_SIZE * SEQ_LEN_LONG * d_model * vocab_size
total_long = attn_total_long + ffn_total_long + final_long
attn_ratio_long = attn_total_long / total_long * 100
ffn_ratio_long = ffn_total_long / total_long * 100
final_ratio_long = final_long / total_long * 100
print(f"    原占比: 自注意力 {attn_ratio:.1f}%, FFN {ffn_ratio:.1f}%, 最终投影 {final_ratio:.1f}%")
print(f"    新占比: 自注意力 {attn_ratio_long:.1f}%, FFN {ffn_ratio_long:.1f}%, 最终投影 {final_ratio_long:.1f}%")
print("    长上下文下，自注意力部分 FLOPs 随 L² 增长，成为绝对主导。\n")

# (f) 含义
print("(f) 含义总结")
print("   • 模型尺寸增大时，FFN 的 FLOPs 占比上升，自注意力占比下降，最终投影占比显著下降。")
print("   • 上下文长度增加时，自注意力成为性能瓶颈，需采用稀疏/线性注意力优化。")
print("   • 大模型训练/推理需结合 FLOPs 分析指导硬件选型与优化方向。")