import torch
import torch.nn as nn
import torch.optim as optim
import math
import random
import numpy as np
from torch.utils.data import Dataset, DataLoader
import json
import os
from tqdm import tqdm

class PoetryDataset(Dataset):
    """诗词数据集类"""
    
    def __init__(self, texts, token_to_id, seq_len=128):
        self.texts = texts
        self.token_to_id = token_to_id
        self.seq_len = seq_len
        self.pad_token_id = token_to_id['[PAD]']
        self.bos_token_id = token_to_id['[BOS]']
        self.eos_token_id = token_to_id['[EOS]']
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        
        # 将文本转换为token ID序列
        tokens = [self.bos_token_id]  # 添加开始标记
        for char in text:
            if char in self.token_to_id:
                tokens.append(self.token_to_id[char])
            else:
                tokens.append(self.token_to_id['[UNK]'])
        tokens.append(self.eos_token_id)  # 添加结束标记
        
        # 截断或填充序列
        if len(tokens) > self.seq_len:
            tokens = tokens[:self.seq_len]
            tokens[-1] = self.eos_token_id  # 确保最后一个token是EOS
        else:
            tokens.extend([self.pad_token_id] * (self.seq_len - len(tokens)))
        
        # 创建输入和目标
        input_ids = torch.tensor(tokens[:-1], dtype=torch.long)
        target_ids = torch.tensor(tokens[1:], dtype=torch.long)
        
        return input_ids, target_ids

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
        
        # 平衡缩放
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
    
    return vocab, token_to_id, id_to_token

def load_data(csv_file, token_to_id, max_samples=None):
    """加载和预处理数据"""
    import pandas as pd
    
    # 读取数据
    try:
        df = pd.read_csv(csv_file, encoding='utf-8-sig')
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(csv_file, encoding='gbk')
        except:
            df = pd.read_csv(csv_file, encoding='utf-8')
    
    # 获取文本列
    if 'text1' in df.columns:
        texts = df['text1'].astype(str).tolist()
    else:
        texts = df.iloc[:, 0].astype(str).tolist()
    
    # 数据清洗
    def clean_text(text):
        import re
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
    
    cleaned_texts = []
    for text in texts:
        cleaned = clean_text(text)
        if len(cleaned) >= 10:  # 过滤太短的文本
            cleaned_texts.append(cleaned)
    
    if max_samples:
        cleaned_texts = cleaned_texts[:max_samples]
    
    print(f"加载了 {len(cleaned_texts)} 条文本")
    return cleaned_texts

def train_decoder_only(config):
    """修复版的训练函数"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    try:
        # 加载词汇表
        vocab, token_to_id, id_to_token = load_vocab(config['vocab_file'])
        config['vocab_size'] = len(vocab)
        config['pad_token_id'] = token_to_id['[PAD]']
        print(f"词汇表大小: {config['vocab_size']}")
        
        # 加载数据
        texts = load_data(config['data_file'], token_to_id, config.get('max_samples'))
        print(f"加载了 {len(texts)} 条文本")
        
        # 创建数据集和数据加载器 - 强制设置num_workers=0避免多进程问题
        dataset = PoetryDataset(texts, token_to_id, config['seq_len'])
        dataloader = DataLoader(
            dataset, 
            batch_size=config['batch_size'], 
            shuffle=True,
            num_workers=0,  # 修复：设置为0避免多进程问题
            pin_memory=True
        )
        
        print(f"数据加载器创建成功，批次数量: {len(dataloader)}")
        
        # 初始化模型
        model = DecoderOnlyTransformer(
            vocab_size=config['vocab_size'],
            d_model=config['d_model'],
            n_heads=config['n_heads'],
            num_layers=config['num_layers'],
            d_ff=config['d_ff'],
            max_seq_len=config['seq_len'],
            dropout=config.get('dropout', 0.1)
        ).to(device)
        
        # 打印模型参数数量
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"模型总参数: {total_params:,}, 可训练参数: {trainable_params:,}")
        
        # 损失函数和优化器
        criterion = nn.CrossEntropyLoss(ignore_index=config['pad_token_id'])
        optimizer = optim.AdamW(
            model.parameters(), 
            lr=config['learning_rate'],
            weight_decay=config.get('weight_decay', 0.01)
        )
        
        # 学习率调度器
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=config['num_epochs'] * len(dataloader)
        )
        
        # 梯度累积
        accumulation_steps = config.get('accumulation_steps', 1)
        
        # 训练循环
        best_loss = float('inf')
        train_losses = []
        
        for epoch in range(config['num_epochs']):
            model.train()
            total_loss = 0
            progress_bar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{config["num_epochs"]}')
            
            for step, (x, y) in enumerate(progress_bar):
                try:
                    x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                    
                    # 创建因果掩码
                    mask = causal_mask(config['seq_len'] - 1).to(device)
                    
                    # 前向传播
                    logits, _ = model(x, mask)
                    loss = criterion(logits.reshape(-1, config['vocab_size']), y.reshape(-1))
                    
                    # 梯度累积
                    loss = loss / accumulation_steps
                    loss.backward()
                    
                    if (step + 1) % accumulation_steps == 0:
                        # 梯度裁剪
                        torch.nn.utils.clip_grad_norm_(model.parameters(), config['max_grad_norm'])
                        optimizer.step()
                        optimizer.zero_grad()
                        scheduler.step()
                    
                    total_loss += loss.item() * accumulation_steps
                    
                    # 更新进度条
                    progress_bar.set_postfix({
                        'loss': f'{loss.item() * accumulation_steps:.4f}',
                        'lr': f'{scheduler.get_last_lr()[0]:.6f}'
                    })
                    
                    # 每100步强制刷新输出
                    if step % 100 == 0:
                        progress_bar.refresh()
                        
                except Exception as e:
                    print(f"训练步骤 {step} 出错: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
            
            # 计算平均损失
            avg_loss = total_loss / len(dataloader)
            train_losses.append(avg_loss)
            
            print(f'Epoch {epoch+1}, 平均损失: {avg_loss:.4f}, 学习率: {scheduler.get_last_lr()[0]:.6f}')
            
            # 保存最佳模型
            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': best_loss,
                    'config': config
                }, config['model_save_path'])
                print(f"保存最佳模型，损失: {best_loss:.4f}")
        
        # 保存训练损失
        with open(config['loss_save_path'], 'w') as f:
            json.dump(train_losses, f)
        
        print(f"训练完成! 最佳损失: {best_loss:.4f}")
        return model, train_losses
        
    except Exception as e:
        print(f"训练过程出错: {e}")
        import traceback
        traceback.print_exc()
        return None, []

def generate_poetry(model, id_to_token, token_to_id, prompt="", max_len=50, temperature=0.8):
    """生成诗词"""
    device = next(model.parameters()).device
    model.eval()
    
    # 初始化输入
    if prompt:
        input_ids = [token_to_id['[BOS]']]
        for char in prompt:
            if char in token_to_id:
                input_ids.append(token_to_id[char])
            else:
                input_ids.append(token_to_id['[UNK]'])
    else:
        input_ids = [token_to_id['[BOS]']]
    
    with torch.no_grad():
        for _ in range(max_len):
            # 准备输入
            input_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)
            mask = causal_mask(len(input_ids)).to(device)
            
            # 前向传播
            logits, _ = model(input_tensor, mask)
            next_token_logits = logits[0, -1, :] / temperature
            
            # 应用softmax获取概率
            probs = torch.softmax(next_token_logits, dim=-1)
            
            # 采样下一个token
            next_token_id = torch.multinomial(probs, num_samples=1).item()
            
            # 添加到序列
            input_ids.append(next_token_id)
            
            # 如果生成结束标记或达到最大长度，停止生成
            if next_token_id == token_to_id['[EOS]'] or len(input_ids) >= max_len:
                break
    
    # 转换为文本
    generated_tokens = []
    for token_id in input_ids[1:]:  # 跳过BOS
        if token_id == token_to_id['[EOS]']:
            break
        generated_tokens.append(id_to_token[token_id])
    
    return ''.join(generated_tokens)

# 配置参数
config = {
    # 'vocab_file': 'vocab_part.txt',  # 词汇表文件路径
    # 'data_file': 'test_part.csv',    # 数据文件路径
    'vocab_file': './vocab.txt',  # 词汇表文件路径
    'data_file': './train.csv',    # 数据文件路径
    'model_save_path': 'best_poetry_model.pth',
    'loss_save_path': 'training_losses.json',
    
    # 模型参数
    'vocab_size': 20000,  # 将从词汇表文件获取实际大小
    'd_model': 512,
    'n_heads': 8,
    'num_layers': 6,
    'd_ff': 2048,
    'seq_len': 128,
    
    # 训练参数
    'batch_size': 32,
    'num_epochs': 50,
    'learning_rate': 1e-4,
    'max_grad_norm': 1.0,
    'dropout': 0.1,
    'weight_decay': 0.01,
    'accumulation_steps': 2,
    'num_workers': 2,
    'max_samples': None  # 设为None使用所有数据，或指定数量用于测试
}

if __name__ == "__main__":
    # 训练模型
    model, losses = train_decoder_only(config)
    
    # 加载词汇表用于生成
    vocab, token_to_id, id_to_token = load_vocab(config['vocab_file'])
    
    # 加载最佳模型进行生成
    checkpoint = torch.load(config['model_save_path'])
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 生成示例诗词
    print("\n=== 生成的诗词示例 ===")
    prompts = ["春风", "明月", "山水", "花开", "相思"]
    for prompt in prompts:
        generated = generate_poetry(model, id_to_token, token_to_id, prompt=prompt)
        print(f"提示: '{prompt}' -> 生成: {generated}")