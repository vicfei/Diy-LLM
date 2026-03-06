import torch
import torch.nn as nn
import math
import re
import pandas as pd

# 模型定义（与训练时保持一致）
class BalancedPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * 
            (-math.log(10000.0) / d_model)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        scaling_factor = math.sqrt(d_model) / 2
        pe = pe * scaling_factor
        
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x):
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len]

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)
        
    def forward(self, x, mask=None):
        batch_size, seq_len, d_model = x.shape
        
        Q = self.w_q(x)
        K = self.w_k(x)
        V = self.w_v(x)
        
        Q = Q.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        if mask is not None:
            scores.masked_fill_(mask, -1e9)
        
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        output = torch.matmul(attn_weights, V)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        output = self.w_o(output)
        
        return output, attn_weights

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()
        
    def forward(self, x):
        return self.linear2(self.dropout(self.activation(self.linear1(x))))

class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # Pre-LN设计
        attn_output, attn_weights = self.self_attn(self.norm1(x), mask)
        x = x + self.dropout(attn_output)
        
        ffn_output = self.ffn(self.norm2(x))
        x = x + self.dropout(ffn_output)
        
        return x, attn_weights

class DecoderOnlyTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, num_layers, d_ff, 
                 max_seq_len=5000, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = BalancedPositionalEncoding(d_model, max_seq_len)
        self.embed_dropout = nn.Dropout(dropout)
        
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, n_heads, d_ff, dropout) 
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)
        
    def forward(self, x, mask=None):
        # 词嵌入 + 缩放
        x = self.token_embedding(x) * math.sqrt(self.d_model)
        
        # 位置编码
        x = self.pos_encoding(x)
        x = self.embed_dropout(x)
        
        # 通过所有Decoder层
        attention_weights = []
        for layer in self.layers:
            x, attn_weights = layer(x, mask)
            attention_weights.append(attn_weights)
        
        # 最终输出
        x = self.norm(x)
        logits = self.lm_head(x)
        
        return logits, attention_weights

def causal_mask(seq_len):
    """创建因果掩码"""
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
    return mask.unsqueeze(0).unsqueeze(0)

def load_vocab(vocab_file):
    """加载词汇表"""
    with open(vocab_file, 'r', encoding='utf-8') as f:
        vocab = [line.strip() for line in f if line.strip()]
    
    token_to_id = {token: idx for idx, token in enumerate(vocab)}
    id_to_token = {idx: token for idx, token in enumerate(vocab)}
    
    print(f"词汇表加载完成，大小: {len(vocab)}")
    return vocab, token_to_id, id_to_token

class PPOPoetryGenerator:
    """PPO训练后的诗词生成器"""
    
    def __init__(self, model_path, vocab_file, device=None):
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
            
        print(f"使用设备: {self.device}")
        
        # 加载词汇表
        self.vocab, self.token_to_id, self.id_to_token = load_vocab(vocab_file)
        self.vocab_size = len(self.vocab)
        
        # 加载模型配置
        checkpoint = torch.load(model_path, map_location=self.device)
        self.config = checkpoint['config']
        
        # 初始化模型
        self.model = DecoderOnlyTransformer(
            vocab_size=self.vocab_size,
            d_model=self.config['d_model'],
            n_heads=self.config['n_heads'],
            num_layers=self.config['num_layers'],
            d_ff=self.config['d_ff'],
            max_seq_len=self.config['seq_len']
        ).to(self.device)
        
        # 加载模型权重
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        print(f"PPO模型加载成功! 训练奖励: {checkpoint.get('reward', '未知'):.4f}")
        
    def generate_poem(self, prompt="", max_len=50, temperature=0.9, show_progress=False):
        """根据prompt生成诗词"""
        # 初始化输入
        if prompt:
            input_ids = [self.token_to_id['[BOS]']]
            for char in prompt:
                if char in self.token_to_id:
                    input_ids.append(self.token_to_id[char])
                else:
                    input_ids.append(self.token_to_id['[UNK]'])
        else:
            input_ids = [self.token_to_id['[BOS]']]
        
        generated_tokens = []
        
        with torch.no_grad():
            for step in range(max_len):
                # 准备输入
                input_tensor = torch.tensor([input_ids], dtype=torch.long).to(self.device)
                mask = causal_mask(len(input_ids)).to(self.device)
                
                # 前向传播
                logits, _ = self.model(input_tensor, mask)
                next_token_logits = logits[0, -1, :] / temperature
                
                # 应用softmax获取概率
                probs = torch.softmax(next_token_logits, dim=-1)
                
                # 采样下一个token
                next_token_id = torch.multinomial(probs, num_samples=1).item()
                
                # 添加到序列
                input_ids.append(next_token_id)
                token_str = self.id_to_token[next_token_id]
                generated_tokens.append(token_str)
                
                if show_progress:
                    current_text = ''.join(generated_tokens)
                    print(f"步骤 {step+1}: {current_text}")
                
                # 如果生成结束标记或达到最大长度，停止生成
                if next_token_id == self.token_to_id['[EOS]'] or len(input_ids) >= max_len:
                    break
        
        # 转换为文本
        generated_poem = ''.join(generated_tokens)
        
        # 清理输出（移除特殊标记）
        generated_poem = generated_poem.replace('[UNK]', '').replace('[PAD]', '')
        
        return generated_poem
    
    def batch_generate(self, prompts, max_len=50, temperature=0.8):
        """批量生成诗词"""
        results = []
        for i, prompt in enumerate(prompts):
            print(f"生成中 ({i+1}/{len(prompts)}): '{prompt}'")
            poem = self.generate_poem(prompt, max_len, temperature)
            results.append(poem)
            print(f"  结果: {poem}\n")
        return results
    
    def interactive_generate(self):
        """交互式生成模式"""
        print("进入交互式诗词生成模式（输入'quit'退出）")
        print("=" * 50)
        
        while True:
            prompt = input("\n请输入提示词: ").strip()
            
            if prompt.lower() in ['quit', 'exit', '退出']:
                print("感谢使用!")
                break
                
            if not prompt:
                print("提示词不能为空，请重新输入")
                continue
                
            print("生成中...")
            poem = self.generate_poem(prompt, max_len=40, temperature=0.7)
            print(f"\n生成结果:")
            print(f"提示: '{prompt}'")
            print(f"诗词: {poem}")
            print("-" * 30)

def format_poem_display(poem, line_length=8):
    """格式化诗词显示，每行指定字数"""
    formatted_lines = []
    current_line = ""
    
    for char in poem:
        current_line += char
        if len(current_line) >= line_length and char in ['，', '。', '！', '？']:
            formatted_lines.append(current_line)
            current_line = ""
    
    if current_line:
        formatted_lines.append(current_line)
    
    return '\n'.join(formatted_lines)

# 使用示例
if __name__ == "__main__":
    # 配置路径
    MODEL_PATH = "stable_rl_model.pth"  # PPO训练保存的模型路径
    VOCAB_FILE = "vocab.txt"  # 词汇表文件路径
    
    # 初始化生成器
    print("正在加载PPO诗词生成模型...")
    generator = PPOPoetryGenerator(MODEL_PATH, VOCAB_FILE)
    
    # 示例1: 单次生成
    print("\n" + "="*50)
    print("示例生成:")
    print("="*50)
    
    test_prompts = ["春风", "明月", "相思", "山水", "花开"]
    for prompt in test_prompts:
        poem = generator.generate_poem(prompt, max_len=40, temperature=0.7)
        formatted_poem = format_poem_display(poem)
        print(f"提示: '{prompt}'")
        print(f"生成:")
        print(formatted_poem)
        print("-" * 30)
    
    # 示例2: 批量生成
    print("\n" + "="*50)
    print("批量生成示例:")
    print("="*50)
    
    batch_prompts = ["秋风", "饮酒", "登高", "夜雨"]
    results = generator.batch_generate(batch_prompts, max_len=35, temperature=0.8)
    
    # 示例3: 交互式生成（取消注释以启用）
    # print("\n" + "="*50)
    # print("交互式生成模式:")
    # print("="*50)
    # generator.interactive_generate()
    
    # 示例4: 自定义参数生成
    print("\n" + "="*50)
    print("自定义参数生成:")
    print("="*50)
    
    custom_prompts = [
        ("江南", 50, 0.6),   # (提示词, 最大长度, 温度)
        ("离别", 30, 0.9),
        ("雪", 40, 0.7)
    ]
    
    for prompt, max_len, temp in custom_prompts:
        poem = generator.generate_poem(prompt, max_len=max_len, temperature=temp)
        formatted_poem = format_poem_display(poem)
        print(f"提示: '{prompt}' (长度:{max_len}, 温度:{temp})")
        print(f"生成:")
        print(formatted_poem)
        print("-" * 30)

    print("\n诗词生成完成!")