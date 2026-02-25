import os
import heapq
import multiprocessing as mp
from collections import Counter
from typing import Dict, List, Tuple, Optional
import regex as re

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def find_chunk_boundaries(
    file,
    desired_num_chunks: int,
    split_special_token: Optional[bytes] = None,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size
    if split_special_token is None:
        return chunk_boundaries

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

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
    file_path: str,
    start: int,
    end: int,
    chunk_id: int,
    special_tokens: List[str],
    pat: str
) -> Dict:
    """
    Process a single chunk: decode, split on special tokens, pre-tokenize,
    and count pre-tokens and byte pairs.
    """
    if special_tokens:
        # 转义特殊标记以用于正则表达式
        escaped_tokens = [re.escape(token) for token in special_tokens]
        delimiter_pattern = re.compile("|".join(escaped_tokens))
    else:
        delimiter_pattern = None
    token_pattern = re.compile(pat, flags=re.UNICODE)
    pretoken_counts = Counter()

    with open(file_path, 'rb') as f:
        f.seek(start)
        chunk_data = f.read(end - start).decode("utf-8", errors="ignore")
       
    if delimiter_pattern:
        # 按特殊标记分割，将它们作为单独的段
        segments = delimiter_pattern.split(chunk_data)
        # 将特殊标记作为它们自己的段添加回来
        special_matches = delimiter_pattern.findall(chunk_data)
        
        # 交织段和特殊标记
        full_segments = []
        for i, segment in enumerate(segments):
            if segment:  # 跳过空段
                full_segments.append(segment)
            if i < len(special_matches):
                full_segments.append(special_matches[i])
    else:
        full_segments = [chunk_data]

    for segment in full_segments:
        if segment in special_tokens:
            token_tuple = (segment,)
            pretoken_counts[token_tuple] += 1
        else:
            # 常规预标记化
            for match in token_pattern.finditer(segment):
                token_str = match.group()
                #if token_str.startswith(' '):
                    #token_str = token_str[1:] 
                token_tuple = tuple(token_str)
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
    Full parallel BPE training that maintains token positions for efficient updates.
    This is more complex but allows parallel processing during the merge phase.
    """
    if special_tokens is None:
        special_tokens = []
    
    if num_processes is None:
        num_processes = mp.cpu_count()
    
    # Step 1: Find chunk boundaries
    with open(input_path, 'rb') as f:
        split_token = special_tokens[0].encode('utf-8') if special_tokens else None
        boundaries = find_chunk_boundaries(f, num_processes, split_token)
    
    # Step 2: Initialize vocabulary
    vocab = {i: bytes([i]) for i in range(256)}
    next_id = 256
    vocab_set = set(vocab.values())
    special_token_bytes = {}
    for token in special_tokens:
        token_bytes = token.encode('utf-8')
        if token_bytes not in vocab_set:
            vocab[next_id] = token_bytes
            vocab_set.add(token_bytes)
            special_token_bytes[token] = token_bytes
            next_id += 1
    
    # Step 3: Initial parallel processing to get token positions
    print("Initial tokenization and pair counting...")
    # Process chunks to get initial token sequences and pair positions
    with mp.Pool(processes=num_processes) as pool:
        chunk_args = [(input_path, boundaries[i], boundaries[i+1], i, special_tokens, PAT)
                      for i in range(len(boundaries) - 1)]
        results = pool.starmap(process_chunk, chunk_args)
    
    # Merge results
    word_counts = Counter()
    for res in results:
        word_counts.update(res['tokens'])

    # Compute initial pair counts from word_counts
  
    pair_counts = Counter()
    for word, freq in word_counts.items():
        if len(word) == 1 and word[0] in special_tokens:
            continue
        for i in range(len(word) - 1):
            pair = (word[i], word[i+1])
            pair_counts[pair] += freq
    # Step 4: Perform merges
    merges = []
    # Create heap
    
    pair_heap = [(-count, pair) for pair, count in pair_counts.items()]
    heapq.heapify(pair_heap) #按元素比较建堆

    while len(vocab) < vocab_size and pair_heap:
        neg_count, best_pair = heapq.heappop(pair_heap)
        count = -neg_count
        if pair_counts.get(best_pair, 0) != count or count == 0:
            continue
        
        token1, token2 = best_pair
        new_token_str = token1 + token2
        new_token_bytes = new_token_str.encode('utf-8')
        
        if new_token_bytes not in vocab_set:
            vocab[next_id] = new_token_bytes
            vocab_set.add(new_token_bytes)
            next_id += 1
        
        merges.append((token1.encode('utf-8'), token2.encode('utf-8')))
        
        # 找出所有包含这个pair的词
        affected_words = []
        for word, freq in list(word_counts.items()):
            if len(word) == 1 and word[0] in special_tokens:
                continue
            
            # 检查是否包含pair
            for i in range(len(word) - 1):
                if word[i] == token1 and word[i+1] == token2:
                    affected_words.append((word, freq))
                    break
        
        # 先移除这些词的所有pair贡献
        for word, freq in affected_words:
            for i in range(len(word) - 1):
                p = (word[i], word[i+1])
                pair_counts[p] -= freq
                if pair_counts[p] <= 0:
                    del pair_counts[p]
        
        # 更新这些词并添加新词的pair贡献
        for word, freq in affected_words:
            # 创建新词
            new_word = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and word[i] == token1 and word[i+1] == token2:
                    new_word.append(new_token_str)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            
            new_word_tuple = tuple(new_word)
            
            # 更新word_counts
            word_counts[word] -= freq
            if word_counts[word] == 0:
                del word_counts[word]
            word_counts[new_word_tuple] = word_counts.get(new_word_tuple, 0) + freq
            
            # 添加新词的pair贡献
            for i in range(len(new_word_tuple) - 1):
                p = (new_word_tuple[i], new_word_tuple[i+1])
                pair_counts[p] = pair_counts.get(p, 0) + freq
        
        # 重建堆（只使用有效的pair）
        pair_heap = [(-c, p) for p, c in pair_counts.items() if c > 0]
        heapq.heapify(pair_heap)
    
    return vocab, merges


if __name__ == "__main__":
    train_txt_path=r"E:\桌面\cs336\assignment1-basics-main\data\TinyStoriesV2-GPT4-valid.txt"
    vocab_size=512
    special_tokens = ["<|endoftext|>"]
    vocab, merges = train_bpe_tokenizer(train_txt_path, vocab_size, special_tokens, num_processes=4)
    print(f"Vocabulary size: {len(vocab)}")
    print(f"Number of merges: {len(merges)}")
    # Optionally print first few merges
    for i, merge in enumerate(merges[:40]):
        print(f"Merge {i+1}: {merge}")
    #print(vocab)
    
    

'''
在每个块中进行pre-tokenization，然后构建dict{pretoken：count}，如（l，o，w）：5，顺便构建dict{pair：count}；
后续合并pair，并维护dict{pretoken：count}和{pair：count}；
遍历每一个块：
    对于块内每一个包含该pair的pretoken，直接类似(l，o，w):5变为(lo，w):5,对pretoken进行合并；
        在该pretoken内维护dict{pair：count}，将原paircount减去pretokencount，例如7-5，将原来的(前一个token，pair中的第一个token)和(pair中的第二个token，后一个token)的count减去pretokencount
        然后在字典中增加{（前一个token，newpair）：pretokencount}和{（newpair，后一个token）：pretokencount}
'''