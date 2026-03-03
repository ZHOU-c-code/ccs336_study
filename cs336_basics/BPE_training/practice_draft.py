'''2.1
print(ord('牛'))
print(chr(0).__repr__())
a="this is a test" + chr(0) + "string"
print(a.__repr__())
print("this is a test" + chr(0) + "string")
'''
'''2.2
print(ord('新'))
test_string = "hello! 新世界"
utf8_encoded = test_string.encode("utf-8")
print(utf8_encoded)
print(type(utf8_encoded))
# Get the byte values for the encoded string (integers from 0 to 255).
print(list(utf8_encoded))
print(len(test_string))
print(len(utf8_encoded))
print(utf8_encoded.decode("utf-8"))
def decode_utf8_bytes_to_str_wrong(bytestring: bytes):
    return "".join([bytes([b]).decode("utf-8") for b in bytestring])
print(decode_utf8_bytes_to_str_wrong("中".encode("utf-8")))
'''
'''2.4
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
#requires `regex` package
import regex as re
x=re.findall(PAT, "some text that i'll pre-tokenize")
print(x)
'''

"""
BPE (Byte Pair Encoding) 完整实现
从原始文本预处理到BPE训练的全过程
参考: Sennrich et al. 2016 "Neural Machine Translation of Rare Words with Subword Units"
"""

'''
from collections import defaultdict, Counter
import math

# ==================== 第一部分：文本预处理 ====================

def preprocess_text(text):
    """
    对原始文本进行预处理
    
    参数:
        text: 原始文本字符串
    
    返回:
        word_counts: 词频字典 {单词: 频率}
        stats: 统计信息
    """
    print("\n" + "="*60)
    print("第1步: 文本预处理")
    print("="*60)
    
    # 1. 原始文本展示
    print("\n1.1 原始语料:")
    print("-" * 40)
    lines = text.strip().split('\n')
    for i, line in enumerate(lines, 1):
        print(f"   行{i}: {line}")
    
    # 2. 分词（按空格分割）
    print("\n1.2 预分词（按空格分割）:")
    print("-" * 40)
    words = text.split()
    print(f"   分词结果: {words}")
    print(f"   总词数: {len(words)}")
    
    # 3. 统计词频
    print("\n1.3 词频统计:")
    print("-" * 40)
    word_counts = Counter(words)
    for word, count in sorted(word_counts.items()):
        print(f"   '{word}': {count}次")
    
    # 4. 统计信息
    stats = {
        'total_words': len(words),
        'unique_words': len(word_counts),
        'lines': len(lines)
    }
    
    print(f"\n1.4 统计摘要:")
    print(f"   总词数: {stats['total_words']}")
    print(f"   唯一词数: {stats['unique_words']}")
    print(f"   总行数: {stats['lines']}")
    
    return dict(word_counts), stats

# ==================== 第二部分：BPE算法实现 ====================

def initialize_vocab(word_counts, special_tokens=None):
    """
    初始化词汇表：将每个单词拆分为字符序列
    
    参数:
        word_counts: 词频字典 {单词: 频率}
        special_tokens: 特殊token列表
    
    返回:
        vocab: 词汇表字典 {单词: [字符列表, 频率]}
        base_vocab: 基础词汇集合
    """
    print("\n" + "="*60)
    print("第2步: 初始化词汇表")
    print("="*60)
    
    vocab = {}
    all_chars = set()
    
    # 添加特殊token
    if special_tokens:
        print(f"\n2.1 添加特殊token: {special_tokens}")
        for token in special_tokens:
            all_chars.add(token)
    
    # 处理每个单词
    print("\n2.2 将单词拆分为字符序列 (添加词尾符号 </w>):")
    print("-" * 40)
    
    for word, count in word_counts.items():
        # 将单词拆分为字符，并添加词尾符号
        chars = list(word) + ['</w>']
        vocab[word] = [chars, count]
        
        # 收集所有单字符
        for c in chars:
            all_chars.add(c)
        
        print(f"   '{word}' (频率:{count:2d}): {' '.join(chars)}")
    
    # 基础词汇
    base_vocab = sorted(all_chars)
    print(f"\n2.3 基础词汇表 (共{len(base_vocab)}个符号):")
    print(f"   {base_vocab}")
    
    return vocab, base_vocab

def get_pair_stats(vocab):
    """
    统计所有相邻字符对的频率
    
    参数:
        vocab: 词汇表字典 {单词: [字符列表, 频率]}
    
    返回:
        pairs: 字典 {(char1, char2): 总频率}
        detailed: 详细统计（用于展示）
    """
    pairs = defaultdict(int)
    detailed = defaultdict(list)
    
    for word, (chars, count) in vocab.items():
        for i in range(len(chars) - 1):
            pair = (chars[i], chars[i+1])
            pairs[pair] += count
            detailed[pair].append((word, count))
    
    return pairs, detailed

def merge_pair(pair, vocab):
    """
    合并指定的字符对
    
    参数:
        pair: 要合并的字符对 (char1, char2)
        vocab: 当前词汇表
    
    返回:
        new_vocab: 合并后的新词汇表
    """
    new_vocab = {}
    merged_token = ''.join(pair)
    
    for word, (chars, count) in vocab.items():
        new_chars = []
        i = 0
        while i < len(chars):
            if i < len(chars) - 1 and chars[i] == pair[0] and chars[i+1] == pair[1]:
                new_chars.append(merged_token)
                i += 2
            else:
                new_chars.append(chars[i])
                i += 1
        new_vocab[word] = [new_chars, count]
    
    return new_vocab

def train_bpe(word_counts, num_merges, special_tokens=None, verbose=True):
    """
    BPE训练主算法
    
    参数:
        word_counts: 词频字典 {单词: 频率}
        num_merges: 要执行的合并次数
        special_tokens: 特殊token列表
        verbose: 是否打印详细信息
    
    返回:
        final_vocab: 最终词汇表
        merges_history: 合并历史记录
        all_tokens: 所有生成的token
    """
    if verbose:
        print("\n" + "="*60)
        print("第3步: BPE训练")
        print("="*60)
    
    # 初始化
    vocab, base_vocab = initialize_vocab(word_counts, special_tokens)
    merges_history = []
    all_tokens = set(base_vocab)
    
    if verbose:
        print(f"\n3.1 开始训练 (目标合并次数: {num_merges})")
    
    # 执行合并
    for step in range(1, num_merges + 1):
        if verbose:
            print(f"\n{'─'*40}")
            print(f"合并步骤 #{step}")
            print(f"{'─'*40}")
        
        # 统计所有字符对
        pairs, detailed = get_pair_stats(vocab)
        
        if not pairs:
            if verbose:
                print("没有更多可合并的字符对")
            break
        
        # 找出频率最高的字符对
        best_pair = max(pairs, key=pairs.get)
        best_freq = pairs[best_pair]
        
        if verbose:
            print(f"\n当前所有字符对频率:")
            for pair, freq in sorted(pairs.items(), key=lambda x: -x[1])[:5]:  # 显示前5个
                print(f"   {pair[0]}+{pair[1]}: {freq}次")
                # 显示出现在哪些词中
                for word, wcount in detailed[pair]:
                    print(f"     出现在 '{word}' (频率:{wcount})")
        
        # 执行合并
        new_token = best_pair[0] + best_pair[1]
        vocab = merge_pair(best_pair, vocab)
        
        # 记录合并
        merges_history.append({
            'step': step,
            'pair': best_pair,
            'new_token': new_token,
            'frequency': best_freq
        })
        all_tokens.add(new_token)
        
        if verbose:
            print(f"\n✓ 选择合并: '{best_pair[0]}' + '{best_pair[1]}' → '{new_token}'")
            print(f"  合并频率: {best_freq}次")
            print(f"\n合并后的词汇表状态:")
            for word, (chars, count) in vocab.items():
                print(f"   '{word}': {' '.join(chars)}")
    
    return vocab, merges_history, sorted(all_tokens)

# ==================== 第三部分：结果展示 ====================

def print_bpe_results(vocab, merges_history, all_tokens, word_counts):
    """
    打印BPE训练结果
    """
    print("\n" + "="*60)
    print("第4步: BPE训练结果")
    print("="*60)
    
    # 4.1 合并历史
    print("\n4.1 合并操作历史:")
    print("-" * 40)
    for merge in merges_history:
        print(f"   步骤{merge['step']:2d}: {merge['pair'][0]}+{merge['pair'][1]:<3} "
              f"→ '{merge['new_token']:<5}' (频率: {merge['frequency']})")
    
    # 4.2 最终词汇表
    print("\n4.2 最终词汇表:")
    print("-" * 40)
    print(f"   基础字符 (共{len([t for t in all_tokens if len(t)==1])}个): "
          f"{[t for t in all_tokens if len(t)==1]}")
    
    merged_tokens = [t for t in all_tokens if len(t) > 1]
    print(f"   合并产生的token (共{len(merged_tokens)}个): {merged_tokens}")
    
    # 4.3 每个单词的最终分词结果
    print("\n4.3 各单词的最终分词结果:")
    print("-" * 40)
    for word, (chars, count) in vocab.items():
        print(f"   '{word}' (频率:{count:2d}): {' '.join(chars)}")
    
    # 4.4 词频加权统计
    print("\n4.4 词频加权统计:")
    print("-" * 40)
    token_usage = defaultdict(int)
    for word, (chars, count) in vocab.items():
        for token in chars:
            token_usage[token] += count
    
    print("   各token在语料中的总出现次数:")
    for token, freq in sorted(token_usage.items(), key=lambda x: -x[1]):
        if len(token) > 1:  # 只显示合并后的token
            print(f"   '{token}': {freq}次")
    
    # 4.5 压缩率计算
    original_chars = sum(len(word) * count for word, count in word_counts.items())
    final_tokens = sum(len(chars) * count for word, (chars, count) in vocab.items())
    compression_ratio = original_chars / final_tokens if final_tokens > 0 else 1
    
    print(f"\n4.5 压缩效果:")
    print(f"   原始字符数: {original_chars}")
    print(f"   最终token数: {final_tokens}")
    print(f"   压缩率: {compression_ratio:.2f}x")

# ==================== 第四部分：主程序 ====================

def main():
    """
    主程序：执行完整的BPE训练流程
    """
    print("="*60)
    print("BPE (Byte Pair Encoding) 完整实现")
    print("基于 Sennrich et al. 2016")
    print("="*60)
    
    # 1. 输入数据
    corpus = """low low low low low
lower lower widest widest widest
newest newest newest newest newest"""
    
    special_tokens = ["<|endoftext|>"]
    
    # 2. 预处理
    word_counts, stats = preprocess_text(corpus)
    
    # 3. BPE训练
    final_vocab, merges, all_tokens = train_bpe(
        word_counts, 
        num_merges=8,  # 合并8次，以便看到完整过程
        special_tokens=special_tokens,
        verbose=True
    )
    
    # 4. 展示结果
    print_bpe_results(final_vocab, merges, all_tokens, word_counts)
    
    # 5. 验证与预期结果对比
    print("\n" + "="*60)
    print("第5步: 验证与预期结果对比")
    print("="*60)
    
    expected_merges = [
        ('e', 's'),      # 应该先合并 'e' + 's' (出现在 newest, widest)
        ('es', 't'),     # 然后合并 'es' + 't' (出现在 newest, widest)
        ('n', 'e'),      # 可能合并 'n' + 'e' (出现在 newest)
        ('w', 'i'),      # 可能合并 'w' + 'i' (出现在 widest)
    ]
    
    print("\n预期的主要合并序列:")
    for i, (a, b) in enumerate(expected_merges, 1):
        print(f"   预期步骤{i}: {a}+{b}")
    
    print("\n实际合并序列:")
    for merge in merges[:4]:
        print(f"   实际步骤{merge['step']}: {merge['pair'][0]}+{merge['pair'][1]} → '{merge['new_token']}'")

# ==================== 第五部分：扩展功能 ====================

def encode_text(text, merges_history):
    """
    使用训练好的BPE模型对新文本进行编码
    
    参数:
        text: 待编码的文本
        merges_history: BPE训练得到的合并历史
    
    返回:
        tokens: 编码后的token序列
    """
    # 初始化：将文本拆分为字符（带词尾标记）
    words = text.split()
    result = []
    
    for word in words:
        chars = list(word) + ['</w>']
        
        # 按照合并历史的逆序应用合并
        for merge in reversed(merges_history):
            pair = merge['pair']
            merged = merge['new_token']
            
            i = 0
            while i < len(chars) - 1:
                if chars[i] == pair[0] and chars[i+1] == pair[1]:
                    chars[i:i+2] = [merged]
                else:
                    i += 1
        
        result.extend(chars)
    
    return result

def demo_encoding():
    """
    演示如何使用训练好的BPE模型编码新文本
    """
    print("\n" + "="*60)
    print("扩展演示: 使用BPE编码新文本")
    print("="*60)
    
    # 使用上面训练的结果演示
    corpus = """low low low low low
lower lower widest widest widest
newest newest newest newest newest"""
    
    word_counts, _ = preprocess_text(corpus)
    _, merges, _ = train_bpe(word_counts, num_merges=8, special_tokens=["<|endoftext|>"], verbose=False)
    
    test_texts = [
        "lowest",
        "wider",
        "new",
        "lower low"
    ]
    
    for text in test_texts:
        tokens = encode_text(text, merges)
        print(f"\n输入: '{text}'")
        print(f"编码: {' '.join(tokens)}")

# ==================== 运行主程序 ====================

if __name__ == "__main__":
    main()
    demo_encoding()
'''
vocab = {i: bytes([i]) for i in range(256)}
print(vocab)