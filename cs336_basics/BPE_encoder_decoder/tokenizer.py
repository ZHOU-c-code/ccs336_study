import regex as re
from typing import Dict, List, Tuple, Optional, Iterable, Iterator, Set
import ast
import heapq
import os
from tests.common import FIXTURES_PATH, gpt2_bytes_to_unicode
import resource
import psutil
import json

VOCAB_PATH = FIXTURES_PATH / "gpt2_vocab.json"
MERGES_PATH = FIXTURES_PATH / "gpt2_merges.txt"



class Tokenizer:
    def __init__(
        self,
        vocab: Dict[int, bytes],
        merges: List[Tuple[bytes, bytes]],
        special_tokens: Optional[List[str]] = None,
    ):
        self.word_pattern = re.compile(
            r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""",
            flags=re.UNICODE
        )

        self.word_cache = {}

        # 确保词汇表包含所有单字节（0-255），否则后续无法编码未知字符
        self.vocab = vocab.copy()
        self.bytes_to_id = {b: i for i, b in self.vocab.items()}

        # 构建合并规则映射：字节对 -> 合并后的 token id
        self.merges = merges
        self.pair_to_id = {}
        for b1, b2 in merges:
            merged = b1 + b2
            if merged not in self.bytes_to_id:
                raise ValueError(f"Merged bytes {merged!r} not in vocab")
            self.pair_to_id[(b1, b2)] = self.bytes_to_id[merged]

        # 处理特殊 token
        self.special_tokens = special_tokens if special_tokens else []
        self.special_to_id = {}
        self.special_pattern = None
        if self.special_tokens:
            # 按长度降序排序，确保最长匹配优先
            sorted_special = sorted(self.special_tokens, key=len, reverse=True)
            self.special_pattern = re.compile(
                "|".join(re.escape(tok) for tok in sorted_special)
            )
            # 为每个特殊 token 分配 id（若已存在则复用）
            next_id = max(self.vocab.keys()) + 1 if self.vocab else 0
            for tok in self.special_tokens:
                tok_bytes = tok.encode("utf-8")
                if tok_bytes in self.bytes_to_id:
                    tid = self.bytes_to_id[tok_bytes]
                else:
                    tid = next_id
                    self.vocab[tid] = tok_bytes
                    self.bytes_to_id[tok_bytes] = tid
                    next_id += 1
                self.special_to_id[tok] = tid

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: Optional[List[str]] = None,
    ) -> "Tokenizer":
        vocab = {}
        with open(vocab_filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(maxsplit=1)
                if len(parts) != 2:
                    raise ValueError(f"Invalid vocab line: {line}")
                idx_str, bytes_repr = parts
                idx = int(idx_str)
                b = ast.literal_eval(bytes_repr)
                if not isinstance(b, bytes):
                    raise ValueError(f"Expected bytes, got {type(b)}: {bytes_repr}")
                vocab[idx] = b

        merges = []
        with open(merges_filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) != 2:
                    raise ValueError(f"Invalid merges line: {line}")
                b1 = ast.literal_eval(parts[0])
                b2 = ast.literal_eval(parts[1])
                if not isinstance(b1, bytes) or not isinstance(b2, bytes):
                    raise ValueError(f"Expected bytes, got {type(b1)} and {type(b2)}")
                merges.append((b1, b2))

        return cls(vocab, merges, special_tokens)

    def encode(self, text: str) -> List[int]:
        """将文本编码为 token id 列表，正确处理特殊 token。"""
        if not text:
            return []

        # 如果没有特殊 token，直接编码全部文本
        if not self.special_tokens:
            return self._encode_ordinary(text)

        # 分割特殊 token 和普通文本
        parts = self._split_by_special(text)
        result = []
        for part, is_special in parts:
            if is_special:
                # 特殊 token 直接映射
                result.append(self.special_to_id[part])
            else:
                # 普通部分进行 BPE 编码
                result.extend(self._encode_ordinary(part))
        return result

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """惰性编码，逐块处理，保证内存效率。"""
        for text_chunk in iterable:
            yield from self.encode(text_chunk)

    def decode(self, ids: List[int]) -> str:
        """将 token id 列表解码为文本，无效 UTF-8 用 � 替换。"""
        all_bytes = b"".join(self.vocab[i] for i in ids)
        return all_bytes.decode("utf-8", errors="replace")

    # ---------- 辅助方法 ----------
    def _encode_ordinary(self, text: str) -> List[int]:
        """将普通文本（不含特殊 token）编码为 token id 列表，遵循预分词 + 词内合并的逻辑。"""
        if not text:
            return []
        pattern = self.word_pattern
        pos = 0
        result = []
        for match in pattern.finditer(text):
            word = match.group()
            if word in self.word_cache:
                ids = self.word_cache[word]
            else:
                word_bytes = word.encode('utf-8')
                ids = self._bpe_encode_word(word_bytes)
                self.word_cache[word] = ids
            result.extend(ids)
            pos = match.end()

        # 处理剩余部分（通常是尾部空白）
        if pos < len(text):
            remaining = text[pos:]
            if remaining in self.word_cache:
                ids = self.word_cache[remaining]
            else:
                remaining_bytes = remaining.encode('utf-8')
                ids = self._bpe_encode_word(remaining_bytes)
                self.word_cache[remaining] = ids
            result.extend(ids)

        return result

    def _bpe_encode_word(self, word_bytes: bytes) -> List[int]:
        """对单个词的字节序列执行 BPE 合并，返回该词最终的 token id 列表。"""
        vocab = self.vocab          # 局部变量
        bytes_to_id = self.bytes_to_id
        pair_to_id = self.pair_to_id
        
        # 初始 id：每个字节对应一个单字节 token
        ids = [bytes_to_id[bytes([b])] for b in word_bytes]
        n = len(ids)
        if n < 2:
            return ids

        # 构建双向链表（数组模拟）
        prev = [-1] * n
        nxt = [-1] * n
        for i in range(n):
            if i > 0:
                prev[i] = i - 1
            if i < n - 1:
                nxt[i] = i + 1

        # 标记每个位置是否仍在链表中（初始全部存活）
        alive = [True] * n

        # 合并规则优先级：索引越小优先级越高
        rank = {pair: idx for idx, pair in enumerate(self.merges)}

        # 初始化优先队列，只放入在合并规则中的相邻对
        heap = []
        for i in range(n - 1):
            pair = (vocab[ids[i]], vocab[ids[i + 1]])
            if pair in rank:
                heapq.heappush(heap, (rank[pair], i, pair[0], pair[1]))

        while heap:
            r, i, left_bytes, right_bytes = heapq.heappop(heap)

            # 检查该对是否仍然有效
            if not alive[i]:
                continue
            j = nxt[i]
            if j == -1 or not alive[j]:
                continue
            if ids[i] != bytes_to_id[left_bytes] or ids[j] != bytes_to_id[right_bytes]:
                continue

            # 执行合并
            merged_id = pair_to_id[(left_bytes, right_bytes)]
            ids[i] = merged_id

            # 更新链表：删除 j
            nxt[i] = nxt[j]
            if nxt[j] != -1:
                prev[nxt[j]] = i
            alive[j] = False  # 标记 j 已删除

            # 生成新的相邻对并加入堆
            if prev[i] != -1:
                p = prev[i]
                pair_p = (vocab[ids[p]], vocab[ids[i]])
                if pair_p in rank:
                    heapq.heappush(heap, (rank[pair_p], p, pair_p[0], pair_p[1]))

            if nxt[i] != -1:
                k = nxt[i]
                pair_k = (vocab[ids[i]], vocab[ids[k]])
                if pair_k in rank:
                    heapq.heappush(heap, (rank[pair_k], i, pair_k[0], pair_k[1]))

        # 按链表顺序收集最终 token
        result = []
        i = 0
        while i != -1:
            result.append(ids[i])
            i = nxt[i]
        return result

    def _split_by_special(self, text: str):
        """
        将文本按特殊 token 分割，返回 (片段, 是否为特殊) 的生成器。
        使用预编译的正则表达式进行最长匹配。
        """
        if not self.special_pattern:
            yield (text, False)
            return

        pos = 0
        for match in self.special_pattern.finditer(text):
            start, end = match.span()
            if start > pos:
                yield (text[pos:start], False)
            yield (text[start:end], True)
            pos = end
        if pos < len(text):
            yield (text[pos:], False)


def memory_limit(max_mem):
    def decorator(f):
        def wrapper(*args, **kwargs):
            process = psutil.Process(os.getpid())
            prev_limits = resource.getrlimit(resource.RLIMIT_AS)
            resource.setrlimit(resource.RLIMIT_AS, (process.memory_info().rss + max_mem, -1))
            try:
                result = f(*args, **kwargs)
                return result
            finally:
                # Even if the function above fails (e.g., it exceeds the
                # memory limit), reset the memory limit back to the
                # previous limit so other tests aren't affected.
                resource.setrlimit(resource.RLIMIT_AS, prev_limits)

        return wrapper

    return decorator

def get_tokenizer_from_vocab_merges_path(
    vocab_path: str | os.PathLike,
    merges_path: str | os.PathLike,
    special_tokens: list[str] | None = None,
):
    gpt2_byte_decoder = {v: k for k, v in gpt2_bytes_to_unicode().items()}
    with open(vocab_path) as vocab_f:
        gpt2_vocab = json.load(vocab_f)
    gpt2_bpe_merges = []
    with open(merges_path) as f:
        for line in f:
            cleaned_line = line.rstrip()
            if cleaned_line and len(cleaned_line.split(" ")) == 2:
                gpt2_bpe_merges.append(tuple(cleaned_line.split(" ")))
    # The GPT-2 tokenizer uses a remapped unicode encoding for bytes. Let's
    # just return the original bytes, so we don't force students to use
    # any particular encoding scheme.
    vocab = {
        gpt2_vocab_index: bytes([gpt2_byte_decoder[token] for token in gpt2_vocab_item])
        for gpt2_vocab_item, gpt2_vocab_index in gpt2_vocab.items()
    }
    # If any of the special tokens don't exist in the vocab, append them to the vocab.
    if special_tokens:
        for special_token in special_tokens:
            byte_encoded_special_token = special_token.encode("utf-8")
            if byte_encoded_special_token not in set(vocab.values()):
                vocab[len(vocab)] = byte_encoded_special_token

    merges = [
        (
            bytes([gpt2_byte_decoder[token] for token in merge_token_1]),
            bytes([gpt2_byte_decoder[token] for token in merge_token_2]),
        )
        for merge_token_1, merge_token_2 in gpt2_bpe_merges
    ]
    return Tokenizer(vocab, merges, special_tokens)


def test_encode_iterable_memory_usage():
    tokenizer = get_tokenizer_from_vocab_merges_path(
        vocab_path=VOCAB_PATH,
        merges_path=MERGES_PATH,
    )
    with open(FIXTURES_PATH / "tinystories_sample_5M.txt") as f:
        ids = []
        for _id in _encode_iterable(tokenizer, f):
            ids.append(_id)
            print(_id)

@memory_limit(int(1e6))
def _encode_iterable(tokenizer, iterable):
    """
    We place tokenizer.encode_iterable into a separate function so we can limit memory
    for just this function. We set the memory limit to 1MB.
    """
    yield from tokenizer.encode_iterable(iterable)

if __name__ == "__main__":
    test_encode_iterable_memory_usage()