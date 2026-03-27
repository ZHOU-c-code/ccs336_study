import torch
import torch.nn as nn
from cs336_basics.transformer.rmsnorm import RMSNorm
from cs336_basics.transformer.transformer_block import TransformerBlock
from cs336_basics.transformer.embedding import Embedding
from cs336_basics.transformer.linear import Linear
from cs336_basics.transformer.softmax import Softmax

class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        num_layers: int,
        rope_theta: float = 10000.0,
    ):
        super().__init__()
        # 嵌入层
        self.token_embedding = Embedding(vocab_size, d_model)

        # 堆叠 Transformer 块
        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                theta=rope_theta,
                max_seq_len=context_length,
            )
            for _ in range(num_layers)
        ])

        # 最大上下文长度
        self.context_length = context_length

        # 最终归一化和输出投影
        self.final_norm = RMSNorm(d_model)
        self.lm_head = Linear(d_model, vocab_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: (batch_size, seq_len)  token indices

        Returns:
            logits: (batch_size, seq_len, vocab_size)  未归一化的词表分布
        """
        seq_len = input_ids.size(1)
        # 序列长度不能超过预定义的最大上下文长度
        assert seq_len <= self.context_length, \
            f"Sequence length {seq_len} exceeds context length {self.context_length}"

        # 2. 嵌入
        x = self.token_embedding(input_ids)                     # (B, S, d_model)


        # 3. 依次经过所有 Transformer 块
        for block in self.blocks:
            x = block(x)

        # 3. 最终归一化并投影到词表
        x = self.final_norm(x)
        logits = self.lm_head(x)                                # (B, S, vocab_size)
        return logits  #Softmax(logits, dim=-1)