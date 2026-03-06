import torch
import torch.nn as nn
import math
import re
import pandas as pd
from typing import List, Optional, Tuple
import time

# ==================== KV Cache实现 ====================
class KVCache:
    """KV Cache管理器，存储每一层的键值对"""
    def __init__(self, num_layers: int, batch_size: int, n_heads: int, d_k: int, device):
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.n_heads = n_heads
        self.d_k = d_k
        self.device = device
        
        # 初始化缓存
        self.key_cache = [torch.empty(0, device=device) for _ in range(num_layers)]
        self.value_cache = [torch.empty(0, device=device) for _ in range(num_layers)]
        
    def update(self, layer_idx: int, new_key: torch.Tensor, new_value: torch.Tensor):
        """更新指定层的KV缓存"""
        # new_key, new_value的形状: (batch_size, n_heads, seq_len, d_k)
        if self.key_cache[layer_idx].numel() == 0:
            # 第一次更新，直接赋值
            self.key_cache[layer_idx] = new_key
            self.value_cache[layer_idx] = new_value
        else:
            # 后续更新，拼接
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], new_key], dim=2)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], new_value], dim=2)
    
    def get_cache(self, layer_idx: int):
        """获取指定层的缓存"""
        return self.key_cache[layer_idx], self.value_cache[layer_idx]
    
    def clear(self):
        """清空缓存"""
        self.key_cache = [torch.empty(0, device=self.device) for _ in range(self.num_layers)]
        self.value_cache = [torch.empty(0, device=self.device) for _ in range(self.num_layers)]

# ==================== 保持与原始模型兼容的带KV Cache模型定义 ====================
class BalancedPositionalEncoding(nn.Module):
    """与原始模型完全一致的位置编码"""
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

class CachedMultiHeadAttention(nn.Module):
    """支持KV Cache的多头注意力，但与原始模型兼容"""
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
        
    def forward(self, x, mask=None, kv_cache=None, layer_idx=0, use_cache=False):
        batch_size, seq_len, d_model = x.shape
        
        # 查询向量
        Q = self.w_q(x)
        
        # 如果有缓存且使用缓存
        if use_cache and kv_cache is not None:
            # 获取缓存
            cache_key, cache_value = kv_cache.get_cache(layer_idx)
            
            # 计算当前token的K和V
            K_new = self.w_k(x)
            V_new = self.w_v(x)
            
            # 形状转换
            Q = Q.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
            K_new = K_new.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
            V_new = V_new.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
            
            # 如果有缓存，则拼接
            if cache_key.numel() > 0:
                K = torch.cat([cache_key, K_new], dim=2)
                V = torch.cat([cache_value, V_new], dim=2)
            else:
                K, V = K_new, V_new
            
            # 更新缓存
            kv_cache.update(layer_idx, K_new, V_new)
            
        else:
            # 无缓存模式（与原始模型完全一致）
            K = self.w_k(x)
            V = self.w_v(x)
            
            Q = Q.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
            K = K.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
            V = V.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        # 应用掩码
        if mask is not None:
            # 如果有缓存，需要调整掩码大小
            if use_cache and kv_cache is not None and cache_key.numel() > 0:
                cache_len = cache_key.size(2)
                # 只掩码当前位置
                mask = mask[:, :, :, cache_len:cache_len + seq_len]
            
            # 根据数据类型选择掩码值
            if scores.dtype == torch.float16:
                scores = scores.masked_fill(mask, -1e4)
            else:
                scores = scores.masked_fill(mask, -1e9)
        
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 注意力加权
        output = torch.matmul(attn_weights, V)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        output = self.w_o(output)
        
        return output, attn_weights

class FeedForward(nn.Module):
    """与原始模型完全一致的前馈网络"""
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()
        
    def forward(self, x):
        return self.linear2(self.dropout(self.activation(self.linear1(x))))

class CachedDecoderLayer(nn.Module):
    """支持KV Cache的解码器层，但与原始模型兼容"""
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = CachedMultiHeadAttention(d_model, n_heads, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None, kv_cache=None, layer_idx=0, use_cache=False):
        # 自注意力 + 残差连接
        attn_output, attn_weights = self.self_attn(
            self.norm1(x), mask, kv_cache, layer_idx, use_cache
        )
        x = x + self.dropout(attn_output)
        
        # 前馈网络 + 残差连接
        ffn_output = self.ffn(self.norm2(x))
        x = x + self.dropout(ffn_output)
        
        return x, attn_weights

class CachedDecoderOnlyTransformer(nn.Module):
    """支持KV Cache的Transformer解码器，但与原始模型兼容"""
    def __init__(self, vocab_size, d_model, n_heads, num_layers, d_ff, 
                 max_seq_len=5000, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = BalancedPositionalEncoding(d_model, max_seq_len)
        self.embed_dropout = nn.Dropout(dropout)
        
        self.layers = nn.ModuleList([
            CachedDecoderLayer(d_model, n_heads, d_ff, dropout) 
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)
        
    def forward(self, x, mask=None, kv_cache=None, use_cache=False):
        # 词嵌入 + 缩放
        x = self.token_embedding(x) * math.sqrt(self.d_model)
        
        # 位置编码
        x = self.pos_encoding(x)
        x = self.embed_dropout(x)
        
        # 通过所有解码器层
        attention_weights = []
        for layer_idx, layer in enumerate(self.layers):
            x, attn_weights = layer(x, mask, kv_cache, layer_idx, use_cache)
            attention_weights.append(attn_weights)
        
        # 最终输出
        x = self.norm(x)
        logits = self.lm_head(x)
        
        return logits, attention_weights

# ==================== 带KV Cache的生成器 ====================
class KVCachePoetryGenerator:
    """支持KV Cache的诗词生成器"""
    
    def __init__(self, model_path, vocab_file, device=None, use_half=False, max_batch_size=8):
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        self.use_half = use_half
        self.max_batch_size = max_batch_size
        
        print(f"使用设备: {self.device}, 半精度: {use_half}, 最大批大小: {max_batch_size}")
        
        # 加载词汇表
        self.vocab, self.token_to_id, self.id_to_token = self.load_vocab(vocab_file)
        self.vocab_size = len(self.vocab)
        
        # 加载模型配置
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        self.config = checkpoint['config']
        
        # 初始化模型
        self.model = CachedDecoderOnlyTransformer(
            vocab_size=self.vocab_size,
            d_model=self.config['d_model'],
            n_heads=self.config['n_heads'],
            num_layers=self.config['num_layers'],
            d_ff=self.config['d_ff'],
            max_seq_len=self.config['seq_len']
        ).to(self.device)
        
        # 加载模型权重
        # 先尝试直接加载
        try:
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print("模型权重加载成功!")
        except RuntimeError as e:
            print(f"直接加载失败，尝试兼容性加载: {e}")
            # 尝试兼容性加载
            self.load_compatible_state_dict(checkpoint['model_state_dict'])
        
        self.model.eval()
        
        # 转换为半精度
        if use_half and self.device.type == 'cuda':
            self.model.half()
        
        print(f"PPO模型加载成功! 训练奖励: {checkpoint.get('reward', '未知'):.4f}")
    
    def load_vocab(self, vocab_file):
        """加载词汇表"""
        with open(vocab_file, 'r', encoding='utf-8') as f:
            vocab = [line.strip() for line in f if line.strip()]
        
        token_to_id = {token: idx for idx, token in enumerate(vocab)}
        id_to_token = {idx: token for idx, token in enumerate(vocab)}
        
        print(f"词汇表加载完成，大小: {len(vocab)}")
        return vocab, token_to_id, id_to_token
    
    def load_compatible_state_dict(self, state_dict):
        """兼容性加载模型权重"""
        model_dict = self.model.state_dict()
        
        # 构建映射关系
        mapping = {
            'pos_encoding.pe': 'pos_encoding.pe',
        }
        
        # 添加FFN层的映射
        for i in range(self.config['num_layers']):
            mapping[f'layers.{i}.ffn.linear1.weight'] = f'layers.{i}.ffn.linear1.weight'
            mapping[f'layers.{i}.ffn.linear1.bias'] = f'layers.{i}.ffn.linear1.bias'
            mapping[f'layers.{i}.ffn.linear2.weight'] = f'layers.{i}.ffn.linear2.weight'
            mapping[f'layers.{i}.ffn.linear2.bias'] = f'layers.{i}.ffn.linear2.bias'
        
        # 加载权重
        for key in state_dict:
            if key in model_dict:
                model_dict[key] = state_dict[key]
            elif key in mapping and mapping[key] in model_dict:
                model_dict[mapping[key]] = state_dict[key]
            else:
                # 尝试模糊匹配
                found = False
                for model_key in model_dict:
                    if key.replace('ffn.', 'ffn.') == model_key:
                        model_dict[model_key] = state_dict[key]
                        found = True
                        break
                
                if not found:
                    print(f"警告: 无法加载权重 {key}")
        
        self.model.load_state_dict(model_dict)
        print("兼容性权重加载完成!")
    
    def generate_with_kv_cache(self, prompt="", max_len=50, temperature=0.9):
        """使用KV Cache生成单首诗词"""
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
        
        # 初始化KV Cache
        kv_cache = KVCache(
            num_layers=self.config['num_layers'],
            batch_size=1,
            n_heads=self.config['n_heads'],
            d_k=self.config['d_model'] // self.config['n_heads'],
            device=self.device
        )
        
        with torch.no_grad():
            # 第一次生成：处理整个prompt
            if len(input_ids) > 1:
                input_tensor = torch.tensor([input_ids], dtype=torch.long).to(self.device)
                seq_len = len(input_ids)
                
                # 创建因果掩码
                mask = torch.triu(torch.ones(seq_len, seq_len, device=self.device), diagonal=1).bool()
                mask = mask.unsqueeze(0).unsqueeze(0)
                
                # 前向传播（不使用缓存，但会填充缓存）
                if self.use_half and self.device.type == 'cuda':
                    with torch.amp.autocast('cuda', enabled=True):
                        logits, _ = self.model(input_tensor, mask, kv_cache, use_cache=True)
                else:
                    logits, _ = self.model(input_tensor, mask, kv_cache, use_cache=True)
                
                # 获取最后一个token的logits
                next_token_logits = logits[0, -1, :] / temperature
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token_id = torch.multinomial(probs, num_samples=1).item()
                
                # 添加到序列
                input_ids.append(next_token_id)
                token_str = self.id_to_token[next_token_id]
                generated_tokens.append(token_str)
            
            # 后续生成：每次只处理一个token，使用KV Cache
            for step in range(max_len - len(generated_tokens)):
                # 只使用最后一个token
                input_tensor = torch.tensor([[input_ids[-1]]], dtype=torch.long).to(self.device)
                
                # 注意：这里不需要掩码，因为KV Cache已经包含了因果信息
                if self.use_half and self.device.type == 'cuda':
                    with torch.amp.autocast('cuda', enabled=True):
                        logits, _ = self.model(input_tensor, mask=None, kv_cache=kv_cache, use_cache=True)
                else:
                    logits, _ = self.model(input_tensor, mask=None, kv_cache=kv_cache, use_cache=True)
                
                next_token_logits = logits[0, -1, :] / temperature
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token_id = torch.multinomial(probs, num_samples=1).item()
                
                # 添加到序列
                input_ids.append(next_token_id)
                token_str = self.id_to_token[next_token_id]
                generated_tokens.append(token_str)
                
                # 如果生成结束标记，停止生成
                if next_token_id == self.token_to_id['[EOS]']:
                    break
        
        # 转换为文本
        generated_poem = ''.join(generated_tokens)
        
        # 清理输出
        generated_poem = generated_poem.replace('[UNK]', '').replace('[PAD]', '')
        eos_pos = generated_poem.find('[EOS]')
        if eos_pos != -1:
            generated_poem = generated_poem[:eos_pos]
        
        return generated_poem
    
    def batch_generate_with_kv_cache(self, prompts, max_len=50, temperature=0.8):
        """使用KV Cache批量生成诗词"""
        batch_size = len(prompts)
        results = [""] * batch_size
        
        # 初始化KV Cache（每个序列独立）
        kv_caches = []
        for _ in range(batch_size):
            kv_cache = KVCache(
                num_layers=self.config['num_layers'],
                batch_size=1,
                n_heads=self.config['n_heads'],
                d_k=self.config['d_model'] // self.config['n_heads'],
                device=self.device
            )
            kv_caches.append(kv_cache)
        
        # 为每个序列初始化输入
        all_input_ids = []
        all_active = []
        
        for prompt in prompts:
            if prompt:
                input_ids = [self.token_to_id['[BOS]']]
                for char in prompt:
                    if char in self.token_to_id:
                        input_ids.append(self.token_to_id[char])
                    else:
                        input_ids.append(self.token_to_id['[UNK]'])
            else:
                input_ids = [self.token_to_id['[BOS]']]
            
            all_input_ids.append(input_ids)
            all_active.append(True)
        
        with torch.no_grad():
            # 处理所有序列的prompt部分
            for i in range(batch_size):
                if all_active[i]:
                    input_ids = all_input_ids[i]
                    if len(input_ids) > 1:
                        input_tensor = torch.tensor([input_ids], dtype=torch.long).to(self.device)
                        seq_len = len(input_ids)
                        
                        # 创建因果掩码
                        mask = torch.triu(torch.ones(seq_len, seq_len, device=self.device), diagonal=1).bool()
                        mask = mask.unsqueeze(0).unsqueeze(0)
                        
                        # 前向传播，填充KV Cache
                        if self.use_half and self.device.type == 'cuda':
                            with torch.amp.autocast('cuda', enabled=True):
                                logits, _ = self.model(
                                    input_tensor, 
                                    mask, 
                                    kv_caches[i], 
                                    use_cache=True
                                )
                        else:
                            logits, _ = self.model(
                                input_tensor, 
                                mask, 
                                kv_caches[i], 
                                use_cache=True
                            )
                        
                        # 获取下一个token
                        next_token_logits = logits[0, -1, :] / temperature
                        probs = torch.softmax(next_token_logits, dim=-1)
                        next_token_id = torch.multinomial(probs, num_samples=1).item()
                        
                        # 更新序列
                        all_input_ids[i].append(next_token_id)
                        token_str = self.id_to_token[next_token_id]
                        
                        # 保存结果
                        results[i] = token_str
                        
                        # 检查是否结束
                        if next_token_id == self.token_to_id['[EOS]']:
                            all_active[i] = False
            
            # 并行生成剩余token
            max_steps = max_len - 1  # 减去已经生成的一个token
            
            for step in range(max_steps):
                # 检查是否所有序列都已完成
                if not any(all_active):
                    break
                
                # 准备当前批次
                batch_indices = []
                
                for i in range(batch_size):
                    if all_active[i]:
                        batch_indices.append(i)
                
                if not batch_indices:
                    break
                
                # 逐个处理（因为KV Cache是独立的，不能批量处理）
                for idx in batch_indices:
                    # 只取最后一个token
                    input_tensor = torch.tensor([[all_input_ids[idx][-1]]], dtype=torch.long).to(self.device)
                    
                    # 使用KV Cache生成
                    if self.use_half and self.device.type == 'cuda':
                        with torch.amp.autocast('cuda', enabled=True):
                            logits, _ = self.model(
                                input_tensor, 
                                mask=None, 
                                kv_cache=kv_caches[idx], 
                                use_cache=True
                            )
                    else:
                        logits, _ = self.model(
                            input_tensor, 
                            mask=None, 
                            kv_cache=kv_caches[idx], 
                            use_cache=True
                        )
                    
                    # 获取下一个token
                    next_token_logits = logits[0, -1, :] / temperature
                    probs = torch.softmax(next_token_logits, dim=-1)
                    next_token_id = torch.multinomial(probs, num_samples=1).item()
                    
                    # 更新序列
                    all_input_ids[idx].append(next_token_id)
                    token_str = self.id_to_token[next_token_id]
                    
                    # 更新结果
                    results[idx] += token_str
                    
                    # 检查是否结束
                    if (next_token_id == self.token_to_id['[EOS]'] or 
                        len(all_input_ids[idx]) >= max_len + len(prompts[idx])):
                        all_active[idx] = False
        
        # 清理输出
        for i in range(batch_size):
            if results[i]:
                results[i] = results[i].replace('[UNK]', '').replace('[PAD]', '')
                eos_pos = results[i].find('[EOS]')
                if eos_pos != -1:
                    results[i] = results[i][:eos_pos]
        
        return results
    
    def fast_generate(self, prompts, max_len=50, temperature=0.8, batch_size=None, use_kv_cache=True):
        """快速生成接口"""
        if batch_size is None:
            batch_size = self.max_batch_size
        
        all_results = []
        
        # 分批处理
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i+batch_size]
            
            if use_kv_cache:
                if len(batch_prompts) == 1:
                    # 单序列使用单序列KV Cache生成
                    result = self.generate_with_kv_cache(batch_prompts[0], max_len, temperature)
                    all_results.append(result)
                else:
                    # 多序列使用批量KV Cache生成
                    batch_results = self.batch_generate_with_kv_cache(batch_prompts, max_len, temperature)
                    all_results.extend(batch_results)
            else:
                # 不使用KV Cache的回退方法
                batch_results = []
                for prompt in batch_prompts:
                    result = self.generate_without_cache(prompt, max_len, temperature)
                    batch_results.append(result)
                all_results.extend(batch_results)
        
        return all_results
    
    def generate_without_cache(self, prompt="", max_len=50, temperature=0.9):
        """不使用KV Cache的生成方法（兼容性）"""
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
                seq_len = len(input_ids)
                
                # 创建因果掩码
                mask = torch.triu(torch.ones(seq_len, seq_len, device=self.device), diagonal=1).bool()
                mask = mask.unsqueeze(0).unsqueeze(0)
                
                # 前向传播（不使用缓存）
                if self.use_half and self.device.type == 'cuda':
                    with torch.amp.autocast('cuda', enabled=True):
                        logits, _ = self.model(input_tensor, mask)
                else:
                    logits, _ = self.model(input_tensor, mask)
                
                next_token_logits = logits[0, -1, :] / temperature
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token_id = torch.multinomial(probs, num_samples=1).item()
                
                # 添加到序列
                input_ids.append(next_token_id)
                token_str = self.id_to_token[next_token_id]
                generated_tokens.append(token_str)
                
                # 如果生成结束标记或达到最大长度，停止生成
                if next_token_id == self.token_to_id['[EOS]'] or len(input_ids) >= max_len:
                    break
        
        # 转换为文本
        generated_poem = ''.join(generated_tokens)
        
        # 清理输出
        generated_poem = generated_poem.replace('[UNK]', '').replace('[PAD]', '')
        eos_pos = generated_poem.find('[EOS]')
        if eos_pos != -1:
            generated_poem = generated_poem[:eos_pos]
        
        return generated_poem

# ==================== 使用示例 ====================
if __name__ == "__main__":
    # 配置路径
    MODEL_PATH = "stable_rl_model.pth"
    VOCAB_FILE = "vocab.txt"
    
    # 测试数据
    test_prompts = [
        "春风", "明月", "相思", "山水", "花开",
        "秋风", "饮酒", "登高", "夜雨", "江南"
    ]
    
    print("="*60)
    print("KV Cache诗词生成器测试")
    print("="*60)
    
    # 创建生成器
    generator = KVCachePoetryGenerator(
        MODEL_PATH, 
        VOCAB_FILE,
        use_half=False,      # 使用全精度（更稳定）
        max_batch_size=4
    )
    
    # 测试单序列KV Cache生成
    print("\n" + "="*60)
    print("单序列KV Cache生成测试:")
    print("="*60)
    
    kv_result = generator.generate_with_kv_cache("春天", max_len=30, temperature=0.7)
    print(f"提示: '春天'")
    print(f"生成: {kv_result}")
    
    # 测试批量KV Cache生成
    print("\n" + "="*60)
    print("批量KV Cache生成测试:")
    print("="*60)
    
    batch_kv_results = generator.batch_generate_with_kv_cache(
        test_prompts[:3], 
        max_len=35, 
        temperature=0.7
    )
    
    for i, (prompt, poem) in enumerate(zip(test_prompts[:3], batch_kv_results)):
        print(f"{i+1}. 提示: '{prompt}'")
        print(f"   生成: {poem}")
        print()
    
    # 性能对比测试
    print("\n" + "="*60)
    print("KV Cache性能对比测试:")
    print("="*60)
    
    # 测试不同配置
    test_configs = [
        ("无缓存", False),
        ("有缓存", True),
    ]
    
    # 使用少量数据测试
    small_prompts = test_prompts[:3]
    
    for config_name, use_kv in test_configs:
        print(f"\n配置: {config_name}")
        
        # 计时
        start_time = time.time()
        
        results = generator.fast_generate(
            small_prompts,
            max_len=30,
            temperature=0.7,
            batch_size=len(small_prompts),
            use_kv_cache=use_kv
        )
        
        end_time = time.time()
        elapsed = end_time - start_time
        
        print(f"时间: {elapsed:.3f}秒")
        print(f"平均每首诗: {elapsed/len(small_prompts)*1000:.1f}毫秒")
        
        # 显示结果
        for i, (prompt, poem) in enumerate(zip(small_prompts, results)):
            print(f"  {i+1}. '{prompt}' -> {poem[:20]}...")
    
    # 保存结果
    print("\n" + "="*60)
    print("保存生成结果:")
    print("="*60)
    
    # 使用最佳配置生成所有测试数据
    final_results = generator.fast_generate(
        test_prompts,
        max_len=40,
        temperature=0.7,
        batch_size=4,
        use_kv_cache=True
    )
    
    # 创建结果DataFrame
    results_df = pd.DataFrame({
        '提示词': test_prompts,
        '生成诗词': final_results,
        '长度': [len(poem) for poem in final_results]
    })
    
    # 保存到CSV
    results_df.to_csv('poetry_kv_cache_results.csv', index=False, encoding='utf-8-sig')
    print(f"结果已保存到 poetry_kv_cache_results.csv")
    print("\n前5条结果:")
    print(results_df.head())
    
    print("\n诗词生成完成!")