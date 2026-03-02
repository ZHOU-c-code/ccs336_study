import os
import multiprocessing as mp
from collections import Counter
from typing import Dict, List, Tuple, Optional
import regex as re
import cProfile
import pstats
import heapq

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def preload_file(input_path, boundaries):
    """预先将整个文件读入内存"""
    with open(input_path, 'rb') as f:
        full_data = f.read()
    
    chunks_data = []
    for i in range(len(boundaries) - 1):
        start = boundaries[i]
        end = boundaries[i+1]
        chunks_data.append(full_data[start:end])
    
    return chunks_data

def find_chunk_boundaries(
    file,
    desired_num_chunks: int,
    split_special_token: Optional[bytes] = None,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes) or split_special_token is None, "Must represent special token as a bytestring"

    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size
    if split_special_token is None:
        return chunk_boundaries

    mini_chunk_size = 4096

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)
        while True:
            mini_chunk = file.read(mini_chunk_size)

            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    return sorted(set(chunk_boundaries))

def process_chunk(
    chunk_data: bytes,
    chunk_id: int,
    special_tokens: List[str],
    pat: str
) -> Dict:
    """
    Process a single chunk: decode, split on special tokens, pre-tokenize,
    and count pre-tokens as tuples of bytes.
    """
    if special_tokens:
        escaped_tokens = [re.escape(token) for token in special_tokens]
        delimiter_pattern = re.compile("|".join(escaped_tokens))
    else:
        delimiter_pattern = None
    token_pattern = re.compile(pat, flags=re.UNICODE)
    pretoken_counts = Counter()

    chunk_text = chunk_data.decode("utf-8", errors="ignore")

    if delimiter_pattern:
        segments = delimiter_pattern.split(chunk_text)
        full_segments = [seg for seg in segments if seg]
    else:
        full_segments = [chunk_text]

    for segment in full_segments:
        for match in token_pattern.finditer(segment):
            token_str = match.group()
            # Each character -> its UTF-8 bytes
            token_bytes = token_str.encode('utf-8')
            token_tuple = tuple(bytes([b]) for b in token_bytes)
            pretoken_counts[token_tuple] += 1

    return {
        'tokens': dict(pretoken_counts),
        'chunk_id': chunk_id
    }


def train_bpe_tokenizer(
    input_path: str,
    vocab_size: int,
    special_tokens: Optional[List[str]] = None,
    num_processes: int = None
) -> Tuple[Dict[int, bytes], List[Tuple[bytes, bytes]]]:
    """
    BPE training using bytes internally.
    Uses single-process pre-tokenization for correctness, then iterative merging.
    """
    if special_tokens is None:
        special_tokens = []
    
    if num_processes is None:
        num_processes = mp.cpu_count()

    # Step 1: Read and pre-tokenize the entire corpus
    use_special_chunking = False
    if "<|endoftext|>" in special_tokens:
        # 快速检查文件中是否有这个token
        with open(input_path, 'rb') as f:
            sample = f.read(1024 * 1024)  # 读1MB
            if b"<|endoftext|>" in sample:
                use_special_chunking = True

    with open(input_path, 'rb') as f:
        if use_special_chunking:
            boundaries = find_chunk_boundaries(f, 2*num_processes, b"<|endoftext|>")
        else:
            boundaries = find_chunk_boundaries(f, 2*num_processes)
    print(f"Created {len(boundaries)-1} chunks for {num_processes} processes")
    # Step 2: Initialize vocabulary
    vocab = {i: bytes([i]) for i in range(256)}
    next_id = 256
    vocab_set = set(vocab.values())
    for token in special_tokens:
        token_bytes = token.encode('utf-8')
        if token_bytes not in vocab_set:
            vocab[next_id] = token_bytes
            vocab_set.add(token_bytes)
            next_id += 1

    # Step 3: Initial parallel processing to get token positions        
    #print("Initial tokenization and pair counting...")
    # Process chunks to get initial token sequences and pair positions
    chunks_data = preload_file(input_path, boundaries)
    with mp.Pool(processes=num_processes) as pool:
        chunk_args = [(chunks_data[i], i, special_tokens, PAT)
                      for i in range(len(chunks_data))]
        results = pool.starmap(process_chunk, chunk_args)

    word_counts = Counter()
    for res in results:
        word_counts.update(res['tokens'])   
    pair_counts = Counter()
    for word, freq in word_counts.items():
        if len(word) == 1:
            continue
        for i in range(len(word) - 1):
            pair = (word[i], word[i+1])
            pair_counts[pair] += freq

    # Step 4: Perform merges
    merges = []
    pair_heap = [(-count, pair) for pair, count in pair_counts.items()]
    heapq.heapify(pair_heap)
    pair_counts_active = pair_counts.copy()
    merge_count = 0
    while len(vocab) < vocab_size and pair_heap:
        merge_count += 1
        while pair_heap:
            neg_count, best_pair = pair_heap[0]  # 只看不pop
            current_count = pair_counts_active.get(best_pair, 0)
            
            # 如果堆顶的值和当前实际值一致，说明有效
            if current_count == -neg_count and current_count > 0:
                break
            else:
                # 过期了，pop掉
                heapq.heappop(pair_heap)       
        # 如果没有有效元素了，退出
        if not pair_heap:
            break

        neg_count, best_pair = heapq.heappop(pair_heap)
        count = -neg_count

        token1, token2 = best_pair
        new_token = token1 + token2

        if new_token not in vocab_set:
            vocab[next_id] = new_token
            vocab_set.add(new_token)
            next_id += 1

        #merges.append((token1, token2,pair_counts[best_pair]))
        merges.append((token1, token2))

        # Find affected words
        pairs_to_update = set()
        words_to_process = list(word_counts.items())
        for word, freq in words_to_process:
            if freq == 0:
                continue
            
            # 快速检查这个词是否包含这个pair
            # 优化：如果token1不在word中，肯定不包含这个pair
            if token1 not in word:
                continue
            
            # 构建新词
            new_word = []
            i = 0
            changed = False
            word_len = len(word)  # 缓存长度
            
            while i < word_len:
                if i < word_len - 1 and word[i] == token1 and word[i+1] == token2:
                    new_word.append(new_token)
                    i += 2
                    changed = True
                else:
                    new_word.append(word[i])
                    i += 1
            
            if not changed:
                continue

            new_word_tuple = tuple(new_word)

            # Update word_counts
            word_counts[word] -= freq
            if word_counts[word] == 0:
                del word_counts[word]
            word_counts[new_word_tuple] += freq

            # Update pair_counts: remove old pairs, add new pairs
            # Old pairs
            for j in range(len(word) - 1):
                p = (word[j], word[j+1])
                old_count = pair_counts_active.get(p, 0)
                new_count = old_count - freq
                
                if new_count > 0:
                    pair_counts_active[p] = new_count
                    # 推入新值（旧值会留在堆中，后面会被跳过）
                    heapq.heappush(pair_heap, (-new_count, p))
                else:
                    # 如果count变为0，从active字典中删除
                    if p in pair_counts_active:
                        del pair_counts_active[p]
            # New pairs
            for j in range(len(new_word_tuple) - 1):
                p = (new_word_tuple[j], new_word_tuple[j+1])
                old_count = pair_counts_active.get(p, 0)
                new_count = old_count + freq
                pair_counts_active[p] = new_count
                heapq.heappush(pair_heap, (-new_count, p))
        
    return vocab, merges

if __name__ == "__main__":
    train_txt_path = "..\data\TinyStoriesV2-GPT4-train.txt"
    vocab_size = 500
    special_tokens = ["<|endoftext|>"]

    pr = cProfile.Profile()
    pr.enable()
    vocab, merges = train_bpe_tokenizer(train_txt_path, vocab_size, special_tokens, num_processes=8)
    pr.disable()

    pr.dump_stats('bpe_profile.prof')
    print("\n=== Top 20 functions by cumulative time ===\n")
    p = pstats.Stats('bpe_profile.prof')
    p.strip_dirs().sort_stats('cumulative').print_stats(20)
    
    print("\n=== Top 20 functions by internal time ===\n")
    p.strip_dirs().sort_stats('time').print_stats(20)
    #for i, merge in enumerate(merges[:95]):
        #print(f"Merge {i+1}: {merge}")