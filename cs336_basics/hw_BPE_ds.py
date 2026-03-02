"""
Optimized BPE Tokenizer Training for Large Datasets
Function-based implementation to match test interface
"""

import os
import multiprocessing as mp
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Optional, Set, Union
import regex as re
import heapq
import mmap
import time
from functools import partial

# 预编译的正则表达式模式
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
TOKEN_PATTERN = re.compile(PAT, flags=re.UNICODE)

# ============ 模块级别函数（用于multiprocessing）============

def process_chunk_worker(args):
    """
    处理单个数据块的worker函数（模块级别，可pickle）
    
    Args:
        args: (file_path, start, end, chunk_id, special_tokens)
    
    Returns:
        该chunk中的token计数
    """
    file_path, start, end, chunk_id, special_tokens = args
    local_counts = defaultdict(int)
    buffer_size = 1024 * 1024  # 1MB缓冲区
    
    # 预编译分割模式
    if special_tokens:
        escaped = [re.escape(t) for t in special_tokens]
        delimiter_pattern = re.compile("|".join(escaped))
    else:
        delimiter_pattern = None
    
    with open(file_path, 'rb') as f:
        f.seek(start)
        remaining = end - start
        overflow = b''
        
        while remaining > 0:
            read_size = min(buffer_size, remaining)
            data = f.read(read_size)
            
            # 处理可能的UTF-8截断
            try:
                text = (overflow + data).decode('utf-8')
                overflow = b''
            except UnicodeDecodeError:
                # 如果解码失败，可能是字符被截断
                for i in range(1, min(5, len(data))):
                    try:
                        text = (overflow + data[:-i]).decode('utf-8')
                        overflow = data[-i:]
                        f.seek(f.tell() - i)
                        remaining += i
                        break
                    except UnicodeDecodeError:
                        continue
                else:
                    text = (overflow + data).decode('utf-8', errors='ignore')
                    overflow = b''
            
            # 分割并tokenize
            if delimiter_pattern:
                segments = delimiter_pattern.split(text)
                for segment in segments:
                    if segment:
                        for match in TOKEN_PATTERN.finditer(segment):
                            token = match.group()
                            token_bytes = token.encode('utf-8')
                            local_counts[token_bytes] += 1
            else:
                for match in TOKEN_PATTERN.finditer(text):
                    token = match.group()
                    token_bytes = token.encode('utf-8')
                    local_counts[token_bytes] += 1
            
            remaining -= read_size
    
    return dict(local_counts)

def compute_pairs_chunk_worker(chunk):
    """
    计算pair频率的worker函数（模块级别，可pickle）
    
    Args:
        chunk: (word_bytes, freq) 的列表
    
    Returns:
        该chunk的pair频率字典
    """
    local_pairs = defaultdict(int)
    for word_bytes, freq in chunk:
        if len(word_bytes) <= 1:
            continue
        # 使用memoryview避免创建新bytes对象
        word_view = memoryview(word_bytes)
        for i in range(len(word_bytes) - 1):
            pair = (bytes(word_view[i:i+1]), bytes(word_view[i+1:i+2]))
            local_pairs[pair] += freq
    return dict(local_pairs)

def merge_counts_worker(chunk):
    """
    合并计数的worker函数
    
    Args:
        chunk: 计数结果列表
    
    Returns:
        合并后的计数
    """
    local_counter = Counter()
    for counts in chunk:
        local_counter.update(counts)
    return local_counter

# ============ 优先队列类 ============

class PriorityQueue:
    """
    优化的优先队列，支持快速更新和删除
    用于管理BPE中的pair频率
    """
    def __init__(self):
        self.heap = []
        self.entry_finder = {}
        self.REMOVED = '<removed>'
        self.counter = 0
    
    def _get_descending_key(self, pair: Tuple[bytes, bytes]) -> tuple:
        """将pair转换为降序排序的键"""
        # 将每个字节取负实现降序
        key1 = tuple(-b for b in pair[0]) + (0,) if pair[0] else (0,)
        key2 = tuple(-b for b in pair[1]) + (0,) if pair[1] else (0,)
        return (key1, key2)
    
    def add_or_update(self, pair: Tuple[bytes, bytes], count: int):
        """添加或更新pair的优先级"""
        if count <= 0:
            self.remove(pair)
            return
        desc_key = self._get_descending_key(pair)
        entry = [-count, desc_key, self.counter, pair]
        self.counter += 1
        
        if pair in self.entry_finder:
            # 标记旧条目为已删除
            self.entry_finder[pair][-1] = self.REMOVED
        
        self.entry_finder[pair] = entry
        heapq.heappush(self.heap, entry)
    
    def remove(self, pair: Tuple[bytes, bytes]):
        """移除pair"""
        if pair in self.entry_finder:
            entry = self.entry_finder.pop(pair)
            entry[-1] = self.REMOVED
    
    def pop_max(self) -> Tuple[Optional[Tuple[bytes, bytes]], int]:
        """弹出最大频率的pair"""
        while self.heap:
            neg_count, desc_key, counter, pair = heapq.heappop(self.heap)
            if pair != self.REMOVED:
                del self.entry_finder[pair]
                return pair, -neg_count
        return None, 0
    
    def peek_max(self) -> Tuple[Optional[Tuple[bytes, bytes]], int]:
        """查看最大频率的pair但不弹出"""
        while self.heap:
            neg_count, _, pair = self.heap[0]
            if pair != self.REMOVED:
                return pair, -neg_count
            heapq.heappop(self.heap)
        return None, 0
    
    def __len__(self) -> int:
        return len(self.entry_finder)

# ============ 辅助函数 ============

def find_chunk_boundaries(
    file_path: str,
    num_chunks: int,
    split_token: Optional[bytes] = None
) -> List[int]:
    """
    使用mmap高效查找chunk边界
    
    Args:
        file_path: 文件路径
        num_chunks: 目标chunk数量
        split_token: 用于分割的特殊token
    
    Returns:
        chunk边界位置列表
    """
    with open(file_path, 'rb') as f:
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            file_size = len(mm)
            chunk_size = file_size // num_chunks
            
            boundaries = [i * chunk_size for i in range(num_chunks + 1)]
            boundaries[-1] = file_size
            
            if split_token is None:
                return boundaries
            
            # 调整边界到split_token位置
            for i in range(1, len(boundaries) - 1):
                pos = boundaries[i]
                found = mm.find(split_token, pos)
                if found != -1:
                    boundaries[i] = found
            
            return sorted(set(boundaries))

def process_chunks_parallel(
    file_path: str,
    boundaries: List[int],
    special_tokens: List[str],
    num_processes: int
) -> List[Dict[bytes, int]]:
    """
    并行处理所有chunks
    
    Args:
        file_path: 文件路径
        boundaries: chunk边界
        special_tokens: 特殊token列表
        num_processes: 进程数
    
    Returns:
        所有chunk的处理结果
    """
    # 准备chunk参数
    chunk_args = []
    for i in range(len(boundaries) - 1):
        chunk_args.append((
            file_path,
            boundaries[i],
            boundaries[i+1],
            i,
            special_tokens
        ))
    
    # 并行处理
    with mp.Pool(processes=num_processes) as pool:
        # 使用imap_unordered实现负载均衡
        results = []
        for result in pool.imap_unordered(process_chunk_worker, chunk_args):
            results.append(result)
    
    return results

def compute_pairs_parallel(
    word_counts: Dict[bytes, int],
    num_workers: int
) -> Dict[Tuple[bytes, bytes], int]:
    """
    并行计算pair频率
    
    Args:
        word_counts: 单词计数
        num_workers: 工作进程数
    
    Returns:
        pair频率字典
    """
    items = list(word_counts.items())
    if not items:
        return {}
    
    # 将单词分块
    chunk_size = max(1, len(items) // (num_workers * 2))
    chunks = [items[i:i+chunk_size] for i in range(0, len(items), chunk_size)]
    
    # 并行计算
    with mp.Pool(processes=min(num_workers, len(chunks))) as pool:
        results = pool.map(compute_pairs_chunk_worker, chunks)
    
    # 合并结果
    pair_counts = defaultdict(int)
    for result in results:
        for pair, count in result.items():
            pair_counts[pair] += count
    
    return dict(pair_counts)

def merge_counts_parallel(
    counts_list: List[Dict[bytes, int]],
    num_workers: int
) -> Counter:
    """
    并行合并计数结果
    
    Args:
        counts_list: 计数结果列表
        num_workers: 工作进程数
    
    Returns:
        合并后的计数
    """
    if not counts_list:
        return Counter()
    
    # 将counts分块
    chunk_size = max(1, len(counts_list) // num_workers)
    chunks = [counts_list[i:i+chunk_size] for i in range(0, len(counts_list), chunk_size)]
    
    # 并行合并
    with mp.Pool(processes=min(num_workers, len(chunks))) as pool:
        partial_counts = pool.map(merge_counts_worker, chunks)
    
    # 最终合并
    final_counts = Counter()
    for pc in partial_counts:
        final_counts.update(pc)
    
    return final_counts

def initialize_vocab(special_tokens: List[str]) -> Tuple[Dict[int, bytes], Set[bytes], int]:
    """
    初始化词汇表
    
    Args:
        special_tokens: 特殊token列表
    
    Returns:
        (vocab, vocab_set, next_id)
    """
    vocab = {i: bytes([i]) for i in range(256)}
    vocab_set = set(vocab.values())
    next_id = 256
    
    # 添加特殊token
    for token in special_tokens:
        token_bytes = token.encode('utf-8')
        if token_bytes not in vocab_set:
            vocab[next_id] = token_bytes
            vocab_set.add(token_bytes)
            next_id += 1
    
    return vocab, vocab_set, next_id

# ============ 主训练函数 ============

def run_train_bpe(
    input_path: Union[str, os.PathLike],
    vocab_size: int,
    special_tokens: List[str],
    num_processes: int = None,
    verbose: bool = True,
    **kwargs,
) -> Tuple[Dict[int, bytes], List[Tuple[bytes, bytes]]]:
    """
    Given the path to an input corpus, run train a BPE tokenizer and
    output its vocabulary and merges.

    Args:
        input_path (str | os.PathLike): Path to BPE tokenizer training data.
        vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be
            kept as a single token. If these special tokens occur in the `input_path`,
            they are treated as any other string.
        num_processes (int, optional): Number of processes for parallel processing.
        verbose (bool): Whether to print progress information.
        **kwargs: Additional arguments (for compatibility).

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
        vocab:
            The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
            to bytes (token bytes)
        merges:
            BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
            representing that <token1> was merged with <token2>.
            Merges are ordered by order of creation.
    """
    # 设置进程数
    if num_processes is None:
        num_processes = mp.cpu_count()
    
    total_start = time.time()
    
    # Step 1: 初始化词汇表
    vocab, vocab_set, next_id = initialize_vocab(special_tokens)
    
    # Step 2: 分块
    if verbose:
        print(f"Phase 1: Chunking file with {num_processes} processes...")
    
    split_token = b"<|endoftext|>" if "<|endoftext|>" in special_tokens else None
    num_chunks = num_processes # 更多小块以便负载均衡
    
    boundaries = find_chunk_boundaries(str(input_path), num_chunks, split_token)
    
    if verbose:
        print(f"  Created {len(boundaries)-1} chunks")
    
    # Step 3: 并行处理chunks
    if verbose:
        phase_start = time.time()
        print("Phase 2: Processing chunks in parallel...")
    
    chunk_results = process_chunks_parallel(str(input_path), boundaries, special_tokens, num_processes)
    
    if verbose:
        print(f"  Completed in {time.time() - phase_start:.2f}s")
    
    # Step 4: 合并计数
    if verbose:
        phase_start = time.time()
        print("Phase 3: Merging counts...")
    
    word_counts = merge_counts_parallel(chunk_results, num_processes)
    
    if verbose:
        print(f"  Found {len(word_counts)} unique tokens")
        print(f"  Completed in {time.time() - phase_start:.2f}s")
    
    # Step 5: 计算pairs
    if verbose:
        phase_start = time.time()
        print("Phase 4: Computing pairs in parallel...")
    
    pair_counts = compute_pairs_parallel(word_counts, num_processes)
    
    if verbose:
        print(f"  Found {len(pair_counts)} unique pairs")
        print(f"  Completed in {time.time() - phase_start:.2f}s")
    
    # Step 6: 执行合并
    if verbose:
        phase_start = time.time()
        print("Phase 5: Performing merges...")
    
    # 创建优先队列
    pq = PriorityQueue()
    for pair, count in pair_counts.items():
        if count > 0:
            pq.add_or_update(pair, count)
    
    # 复制word_counts用于更新
    current_words_tuple = {}
    for word_bytes, freq in word_counts.items():
        # 将 bytes 转换为 tuple of single-byte bytes
        word_tuple = tuple(word_bytes[i:i+1] for i in range(len(word_bytes)))
        current_words_tuple[word_tuple] = freq
    merges = []
    merge_count = 0
    
    while len(vocab) < vocab_size:
        best_pair, count = pq.pop_max()
        if best_pair is None or count == 0:
            break
        
        token1, token2 = best_pair
        new_token = token1 + token2
        
        if new_token in vocab_set:
            continue
        
        # 添加到词汇表
        vocab[next_id] = new_token
        vocab_set.add(new_token)
        next_id += 1
        #merges.append((token1, token2,count))
        merges.append((token1, token2))
        merge_count += 1
        
        # 更新受影响的单词
        words_to_update = []
        for word_tuple, freq in current_words_tuple.items():
            if freq == 0:
                continue
            # 检查是否包含这个 pair
            for i in range(len(word_tuple) - 1):
                if word_tuple[i] == token1 and word_tuple[i+1] == token2:
                    words_to_update.append((word_tuple, freq))
                    break  # 找到一个即可
        
        # 如果没有词包含这个pair，继续下一个
        if not words_to_update:
            continue
        
        # 批量更新
        word_updates = defaultdict(int)  # {word_tuple: delta}
        pair_updates = defaultdict(int)  # {pair: delta}
        
        for word_tuple, freq in words_to_update:
            # 1. 记录旧词产生的所有pairs（要删除）
            for i in range(len(word_tuple) - 1):
                old_pair = (word_tuple[i], word_tuple[i+1])
                pair_updates[old_pair] -= freq
            
            # 2. 构建新词（在 token 层面合并）
            new_word = []
            i = 0
            word_len = len(word_tuple)
            
            while i < word_len:
                if i < word_len - 1 and word_tuple[i] == token1 and word_tuple[i+1] == token2:
                    new_word.append(new_token)
                    i += 2
                else:
                    new_word.append(word_tuple[i])
                    i += 1
            
            new_word_tuple = tuple(new_word)
            
            # 3. 记录新词产生的所有pairs（要添加）
            for i in range(len(new_word_tuple) - 1):
                new_pair = (new_word_tuple[i], new_word_tuple[i+1])
                pair_updates[new_pair] += freq
            
            # 4. 记录word_counts的更新
            word_updates[word_tuple] -= freq
            word_updates[new_word_tuple] += freq
        
        # 批量应用 word_counts 更新
        for word_tuple, delta in word_updates.items():
            new_freq = current_words_tuple.get(word_tuple, 0) + delta
            if new_freq > 0:
                current_words_tuple[word_tuple] = new_freq
            else:
                if word_tuple in current_words_tuple:
                    del current_words_tuple[word_tuple]
        
        # 批量应用 pair_counts 更新
        for pair, delta in pair_updates.items():
            new_count = pair_counts.get(pair, 0) + delta
            if new_count > 0:
                pair_counts[pair] = new_count
                pq.add_or_update(pair, new_count)
            else:
                if pair in pair_counts:
                    del pair_counts[pair]
                pq.remove(pair)      
    
    if verbose:
        print(f"  Completed {merge_count} merges in {time.time() - phase_start:.2f}s")
        print(f"Total training time: {time.time() - total_start:.2f}s")
        print(f"Final vocabulary size: {len(vocab)}")
    
    return vocab, merges


# ============ 兼容性包装函数 ============

def train_bpe_tokenizer(
    input_path: Union[str, os.PathLike],
    vocab_size: int,
    special_tokens: List[str],
    num_processes: int = None,
    **kwargs
) -> Tuple[Dict[int, bytes], List[Tuple[bytes, bytes]]]:
    """
    兼容原接口的包装函数
    """
    return run_train_bpe(input_path, vocab_size, special_tokens, num_processes, False, **kwargs)


# ============ 主函数 ============

def main():
    """主函数"""
    import cProfile
    import pstats
    
    # 配置
    #train_txt_path = "../data/TinyStoriesV2-GPT4-train.txt"
    train_txt_path = "../tests/fixtures/tinystories_sample_5M.txt"
    vocab_size = 1000
    special_tokens = ["<|endoftext|>"]
    
    # 性能分析
    pr = cProfile.Profile()
    pr.enable()
    
    # 训练 - 使用符合测试接口的函数
    vocab, merges = run_train_bpe(
        input_path=train_txt_path,
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        num_processes=8,
        verbose=True
    )
    
    pr.disable()
    
    # 保存profile结果
    pr.dump_stats('bpe_function_optimized.prof')
    
    # 打印性能统计
    print("\n" + "="*60)
    print("Performance Profile (Top 20 by cumulative time)")
    print("="*60)
    
    p = pstats.Stats('bpe_function_optimized.prof')
    p.strip_dirs().sort_stats('cumulative').print_stats(20)
    
    print("\n" + "="*60)
    print("Performance Profile (Top 20 by internal time)")
    print("="*60)
    
    p.strip_dirs().sort_stats('time').print_stats(20)
    
    # 验证结果
    print(f"\nFinal vocabulary size: {len(vocab)}")
    print(f"Number of merges: {len(merges)}")
    
    # 打印前10个合并操作作为示例
    print("\nFirst 10 merges:")
    for i, (t1, t2, count) in enumerate(merges[570:590]):
        try:
            t1_str = t1.decode('utf-8')
            t2_str = t2.decode('utf-8')
        except:
            t1_str = str(t1)
            t2_str = str(t2)
        print(f"  Merge {i+1}: {t1_str} + {t2_str}, {count}")


if __name__ == "__main__":
    # 重要：设置multiprocessing的启动方法
    mp.freeze_support()
    main()