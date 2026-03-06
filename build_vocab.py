import pandas as pd
import json
from collections import Counter
import os
import re

def build_poetry_vocabulary(input_file, output_dir="vocab", vocab_size=20000, min_freq=2):
    """
    构建诗词专用词汇表
    
    Args:
        input_file: 输入CSV文件路径
        output_dir: 输出目录
        vocab_size: 词汇表大小
        min_freq: 字符最小出现频率
    """
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 读取数据
    print("正在读取数据...")
    try:
        df = pd.read_csv(input_file, encoding='utf-8-sig')
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(input_file, encoding='gbk')
        except:
            df = pd.read_csv(input_file, encoding='utf-8')
    
    # 获取文本列
    if 'text1' in df.columns:
        texts = df['text1'].astype(str).tolist()
    else:
        texts = df.iloc[:, 0].astype(str).tolist()
    
    print(f"成功读取 {len(texts)} 条文本")
    
    # 数据清洗
    print("正在清洗数据...")
    cleaned_texts = []
    for text in texts:
        cleaned = clean_text(text)
        if len(cleaned) >= 10:  # 过滤太短的文本
            cleaned_texts.append(cleaned)
    
    print(f"清洗后保留 {len(cleaned_texts)} 条文本")
    
    # 合并所有文本
    all_text = ''.join(cleaned_texts)
    print(f"总字符数: {len(all_text)}")
    
    # 统计字符频率
    print("正在统计字符频率...")
    char_counter = Counter(all_text)
    
    # 输出字符统计信息
    print(f"总唯一字符数: {len(char_counter)}")
    print(f"前20个高频字符: {char_counter.most_common(20)}")
    
    # 构建词汇表
    print("正在构建词汇表...")
    vocabulary = build_vocab_from_counter(char_counter, vocab_size, min_freq)
    
    # 保存词汇表
    save_vocabulary(vocabulary, output_dir, char_counter)
    
    # 生成字符统计报告
    generate_char_report(all_text, output_dir)
    
    print("词汇表构建完成!")
    return vocabulary

def clean_text(text):
    """清洗文本"""
    if pd.isna(text):
        return ""
    
    # 移除特殊字符但保留中文标点和常见符号
    text = re.sub(r'[^\u4e00-\u9fa5，。！？；：、（）《》「」\s]', '', str(text))
    
    # 标准化标点
    text = text.replace(',', '，').replace('.', '。').replace('!', '！')
    text = text.replace('?', '？').replace(';', '；').replace(':', '：')
    
    # 移除多余空白
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def build_vocab_from_counter(char_counter, vocab_size, min_freq):
    """从字符计数器构建词汇表"""
    
    # 特殊标记
    special_tokens = [
        '[PAD]',    # 填充标记
        '[UNK]',    # 未知字符
        '[BOS]',    # 开始标记
        '[EOS]',    # 结束标记
        '[CLS]',    # 分类标记
        '[SEP]',    # 分隔标记
        '[MASK]',   # 掩码标记
    ]
    
    # 过滤低频字符
    filtered_chars = [(char, count) for char, count in char_counter.items() if count >= min_freq]
    
    # 按频率排序
    sorted_chars = sorted(filtered_chars, key=lambda x: x[1], reverse=True)
    
    # 构建词汇表
    vocabulary = special_tokens.copy()
    
    # 添加高频字符
    for char, count in sorted_chars[:vocab_size - len(special_tokens)]:
        vocabulary.append(char)
    
    print(f"词汇表大小: {len(vocabulary)}")
    print(f"包含特殊标记: {len(special_tokens)}")
    print(f"包含中文字符: {len(vocabulary) - len(special_tokens)}")
    
    return vocabulary

def save_vocabulary(vocabulary, output_dir, char_counter):
    """保存词汇表到文件"""
    
    # 保存词汇表文件
    vocab_file = os.path.join(output_dir, "vocab.txt")
    with open(vocab_file, 'w', encoding='utf-8') as f:
        for token in vocabulary:
            f.write(token + '\n')
    
    print(f"词汇表已保存到: {vocab_file}")
    
    # 保存字符频率文件
    freq_file = os.path.join(output_dir, "char_frequency.json")
    freq_data = {
        "vocab_size": len(vocabulary),
        "character_frequency": {char: count for char, count in char_counter.most_common(1000)}
    }
    
    with open(freq_file, 'w', encoding='utf-8') as f:
        json.dump(freq_data, f, ensure_ascii=False, indent=2)
    
    print(f"字符频率已保存到: {freq_file}")
    
    # 保存词汇表映射
    mapping_file = os.path.join(output_dir, "token_to_id.json")
    token_to_id = {token: idx for idx, token in enumerate(vocabulary)}
    
    with open(mapping_file, 'w', encoding='utf-8') as f:
        json.dump(token_to_id, f, ensure_ascii=False, indent=2)
    
    print(f"词汇表映射已保存到: {mapping_file}")

def generate_char_report(text, output_dir):
    """生成字符统计报告"""
    
    # 基本统计
    total_chars = len(text)
    unique_chars = len(set(text))
    
    # 字符类型统计
    chinese_chars = len(re.findall(r'[\u4e00-\u9fa5]', text))
    punctuation_chars = len(re.findall(r'[，。！？；：、（）《》「」]', text))
    space_chars = text.count(' ')
    
    # 字符频率分布
    char_counter = Counter(text)
    freq_ranges = {
        ">=1000": len([c for c, f in char_counter.items() if f >= 1000]),
        "100-999": len([c for c, f in char_counter.items() if 100 <= f < 1000]),
        "10-99": len([c for c, f in char_counter.items() if 10 <= f < 100]),
        "2-9": len([c for c, f in char_counter.items() if 2 <= f < 10]),
        "1": len([c for c, f in char_counter.items() if f == 1]),
    }
    
    # 生成报告
    report = {
        "basic_stats": {
            "total_characters": total_chars,
            "unique_characters": unique_chars,
            "character_diversity": unique_chars / total_chars if total_chars > 0 else 0
        },
        "character_types": {
            "chinese_characters": chinese_chars,
            "punctuation_marks": punctuation_chars,
            "spaces": space_chars,
            "other": total_chars - chinese_chars - punctuation_chars - space_chars
        },
        "frequency_distribution": freq_ranges,
        "top_50_characters": char_counter.most_common(50)
    }
    
    # 保存报告
    report_file = os.path.join(output_dir, "character_report.json")
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    print(f"字符统计报告已保存到: {report_file}")
    
    # 打印简要报告
    print("\n=== 字符统计报告 ===")
    print(f"总字符数: {total_chars}")
    print(f"唯一字符数: {unique_chars}")
    print(f"字符多样性: {report['basic_stats']['character_diversity']:.4f}")
    print(f"中文字符: {chinese_chars} ({chinese_chars/total_chars*100:.1f}%)")
    print(f"标点符号: {punctuation_chars} ({punctuation_chars/total_chars*100:.1f}%)")
    
    print("\n字符频率分布:")
    for range_name, count in freq_ranges.items():
        print(f"  {range_name}: {count} 个字符")

def analyze_vocabulary_coverage(vocabulary, texts, output_dir):
    """分析词汇表覆盖度"""
    
    all_chars = set(''.join(texts))
    vocab_chars = set(vocabulary[7:])  # 排除特殊标记
    
    covered_chars = all_chars.intersection(vocab_chars)
    uncovered_chars = all_chars - vocab_chars
    
    coverage = len(covered_chars) / len(all_chars) if len(all_chars) > 0 else 0
    
    coverage_report = {
        "total_unique_chars": len(all_chars),
        "vocab_chars": len(vocab_chars),
        "covered_chars": len(covered_chars),
        "uncovered_chars": len(uncovered_chars),
        "coverage_rate": coverage,
        "uncovered_characters": list(uncovered_chars)
    }
    
    # 保存覆盖度报告
    coverage_file = os.path.join(output_dir, "vocabulary_coverage.json")
    with open(coverage_file, 'w', encoding='utf-8') as f:
        json.dump(coverage_report, f, ensure_ascii=False, indent=2)
    
    print(f"\n词汇表覆盖度分析:")
    print(f"总唯一字符: {len(all_chars)}")
    print(f"词汇表字符: {len(vocab_chars)}")
    print(f"覆盖字符: {len(covered_chars)}")
    print(f"未覆盖字符: {len(uncovered_chars)}")
    print(f"覆盖度: {coverage*100:.2f}%")
    
    if uncovered_chars:
        print(f"未覆盖字符示例: {list(uncovered_chars)[:20]}")
    
    return coverage_report

# 使用示例
if __name__ == "__main__":
    input_file = "train.csv"  # 替换为您的文件路径
    output_dir = "poetry_vocab"
    
    # 构建词汇表
    vocabulary = build_poetry_vocabulary(
        input_file=input_file,
        output_dir=output_dir,
        vocab_size=20000,
        min_freq=1  # 对于诗词数据，可以设置为1以保留所有字符
    )
    
    # 重新读取数据用于覆盖度分析
    df = pd.read_csv(input_file, encoding='utf-8-sig')
    if 'text1' in df.columns:
        texts = df['text1'].astype(str).tolist()
    else:
        texts = df.iloc[:, 0].astype(str).tolist()
    
    cleaned_texts = [clean_text(text) for text in texts if len(clean_text(text)) >= 10]
    
    # 分析词汇表覆盖度
    coverage_report = analyze_vocabulary_coverage(vocabulary, cleaned_texts, output_dir)
    
    print(f"\n词汇表构建完成！所有文件保存在: {output_dir}/")
    print("生成的文件包括:")
    print("  - vocab.txt: 词汇表文件")
    print("  - token_to_id.json: 词汇表映射")
    print("  - char_frequency.json: 字符频率统计")
    print("  - character_report.json: 字符统计报告")
    print("  - vocabulary_coverage.json: 词汇表覆盖度分析")