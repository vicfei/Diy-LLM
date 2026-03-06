import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
import matplotlib.pyplot as plt
import pandas as pd
import re
import math
from tqdm import tqdm

class FixedPPOTrainer:
    def __init__(self, model, vocab_size, device, 
                 lr=1e-5, clip_epsilon=0.2, value_coef=0.5, entropy_coef=0.01):
        self.model = model
        self.vocab_size = vocab_size
        self.device = device
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        
        # 优化器 - 确保所有参数都要求梯度
        self.optimizer = optim.AdamW(
            [p for p in model.parameters() if p.requires_grad], 
            lr=lr
        )
        print(f"PPO训练器初始化完成，可训练参数: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    
    def compute_rewards(self, generated_poems, prompts):
        """计算奖励函数"""
        rewards = []
        
        for poem, prompt in zip(generated_poems, prompts):
            reward = 0
            
            # 1. 长度奖励
            length_score = min(len(poem) / 25.0, 0.3)
            reward += length_score
            
            # 2. 提示词相关性
            if prompt and any(word in poem for word in prompt):
                reward += 0.3
            
            # 3. 诗词结构
            if '。' in poem:
                reward += 0.2
                # 额外的结构完整性奖励
                lines = [line for line in poem.split('。') if line.strip()]
                if len(lines) >= 2:
                    reward += 0.1
            
            # 4. 多样性
            if len(poem) > 0:
                unique_ratio = len(set(poem)) / len(poem)
                reward += unique_ratio * 0.2
            
            # 5. 古典诗词特征奖励
            classical_words = {'风', '月', '花', '山', '水', '云', '雨', '春', '秋', '夜'}
            classical_count = sum(1 for char in poem if char in classical_words)
            if len(poem) > 0:
                classical_ratio = classical_count / len(poem)
                reward += classical_ratio * 0.1
            
            rewards.append(min(reward, 1.0))  # 限制最大奖励为1.0
        
        return torch.tensor(rewards, device=self.device, requires_grad=False)
    
    def generate_with_log_probs(self, prompts, max_len=30, temperature=0.8, require_grad=False):
        """生成诗词并记录log概率 - 修复梯度问题"""
        batch_size = len(prompts)
        
        # 初始化
        current_inputs = []
        for prompt in prompts:
            input_ids = [self.token_to_id['[BOS]']]
            for char in prompt:
                if char in self.token_to_id:
                    input_ids.append(self.token_to_id[char])
                else:
                    input_ids.append(self.token_to_id['[UNK]'])
            current_inputs.append(input_ids)
        
        log_probs = []
        entropies = []
        
        # 根据是否需要梯度选择上下文管理器
        context = torch.no_grad() if not require_grad else torch.enable_grad()
        
        with context:
            for step in range(max_len):
                # 准备当前输入
                max_len_current = max(len(seq) for seq in current_inputs)
                padded_inputs = []
                for seq in current_inputs:
                    padded = seq + [self.token_to_id['[PAD]']] * (max_len_current - len(seq))
                    padded_inputs.append(padded)
                
                input_tensor = torch.tensor(padded_inputs, dtype=torch.long).to(self.device)
                mask = causal_mask(max_len_current).to(self.device)
                
                # 前向传播
                logits, _ = self.model(input_tensor, mask)
                last_logits = logits[:, -1, :]
                
                # 计算概率分布
                probs = torch.softmax(last_logits / temperature, dim=-1)
                log_probs_step = torch.log_softmax(last_logits, dim=-1)
                
                # 采样下一个token
                actions = torch.multinomial(probs, num_samples=1).squeeze(-1)
                
                # 计算选择的log概率和熵
                action_log_probs = log_probs_step.gather(1, actions.unsqueeze(-1)).squeeze(-1)
                entropy = -torch.sum(probs * log_probs_step, dim=-1)
                
                # 更新序列
                for i in range(batch_size):
                    current_inputs[i].append(actions[i].item())
                
                log_probs.append(action_log_probs)
                entropies.append(entropy)
                
                # 检查是否所有序列都结束
                all_eos = all(actions[i].item() == self.token_to_id['[EOS]'] for i in range(batch_size))
                if all_eos:
                    break
        
        # 转换为诗词文本
        generated_poems = []
        for seq in current_inputs:
            tokens = []
            for token_id in seq[1:]:
                if token_id == self.token_to_id['[EOS]']:
                    break
                tokens.append(self.id_to_token.get(token_id, '[UNK]'))
            generated_poems.append(''.join(tokens))
        
        # 堆叠log概率和熵
        if log_probs:
            log_probs_tensor = torch.stack(log_probs, dim=1)
            entropies_tensor = torch.stack(entropies, dim=1)
        else:
            log_probs_tensor = torch.zeros(batch_size, 0, device=self.device)
            entropies_tensor = torch.zeros(batch_size, 0, device=self.device)
        
        return generated_poems, log_probs_tensor, entropies_tensor
    
    def ppo_update(self, old_log_probs, new_log_probs, rewards, entropies):
        """修复的PPO更新步骤"""
        # 确保所有张量都在计算图中
        if not old_log_probs.requires_grad:
            old_log_probs = old_log_probs.detach()
        
        # 计算概率比
        ratio = torch.exp(new_log_probs - old_log_probs)
        
        # 确保rewards有正确的维度
        if rewards.dim() == 1:
            rewards = rewards.unsqueeze(1)
        
        # PPO裁剪目标
        surr1 = ratio * rewards
        surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * rewards
        
        # 策略损失 - 使用负号因为我们要最大化奖励
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # 熵奖励（鼓励探索）
        entropy_bonus = -self.entropy_coef * entropies.mean()
        
        # 总损失
        total_loss = policy_loss + entropy_bonus
        
        # 反向传播
        self.optimizer.zero_grad()
        total_loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        return total_loss.item()
    
    def train_epoch(self, prompts, epochs_per_batch=2):
        """修复的训练epoch"""
        self.model.train()  # 确保模型在训练模式
        
        # 第一步：用旧策略生成（不需要梯度）
        with torch.no_grad():
            generated_poems, old_log_probs, _ = self.generate_with_log_probs(
                prompts, require_grad=False
            )
        
        # 计算奖励
        rewards = self.compute_rewards(generated_poems, prompts)
        
        # 多次PPO更新
        total_loss = 0
        for update_step in range(epochs_per_batch):
            # 第二步：用新策略生成（需要梯度）
            _, new_log_probs, entropies = self.generate_with_log_probs(
                prompts, require_grad=True
            )
            
            # PPO更新
            loss = self.ppo_update(old_log_probs, new_log_probs, rewards, entropies)
            total_loss += loss
        
        avg_loss = total_loss / epochs_per_batch
        avg_reward = rewards.mean().item()
        
        return generated_poems, avg_reward, avg_loss

    def set_vocab(self, token_to_id, id_to_token):
        """设置词汇表"""
        self.token_to_id = token_to_id
        self.id_to_token = id_to_token

def stable_rl_training(config):
    """稳定的强化学习训练"""
    print("=" * 50)
    print("开始稳定的强化学习训练")
    print("=" * 50)
    
    # 1. 加载词汇表
    vocab, token_to_id, id_to_token = load_vocab(config['vocab_file'])
    
    # 2. 加载数据
    training_texts = load_data_debug(config['data_file'], token_to_id, config.get('max_samples', 50))
    
    # 3. 生成提示词
    prompts = generate_prompts_simple(training_texts, config.get('num_prompts', 20))
    
    # 4. 加载模型
    print("加载基础模型...")
    try:
        checkpoint = torch.load(config['base_model_path'], map_location=config['device'])
        
        # 创建模型实例
        base_model = DecoderOnlyTransformer(
            vocab_size=len(vocab),
            d_model=config['d_model'],
            n_heads=config['n_heads'],
            num_layers=config['num_layers'],
            d_ff=config['d_ff'],
            max_seq_len=config['seq_len']
        ).to(config['device'])
        
        # 加载权重
        base_model.load_state_dict(checkpoint['model_state_dict'])
        
        # 确保所有参数都要求梯度
        for param in base_model.parameters():
            param.requires_grad = True
            
        print("模型加载成功!")
        
    except Exception as e:
        print(f"模型加载失败: {e}")
        import traceback
        traceback.print_exc()
        return None, []
    
    # 5. 设置PPO训练器
    ppo_trainer = FixedPPOTrainer(
        model=base_model,
        vocab_size=len(vocab),
        device=config['device'],
        lr=config.get('rl_lr', 1e-5),
        clip_epsilon=0.1,  # 减小裁剪参数
        entropy_coef=0.001  # 减小熵系数
    )
    ppo_trainer.set_vocab(token_to_id, id_to_token)
    
    # 6. 动态调整批次大小
    actual_batch_size = min(config['rl_batch_size'], len(prompts))
    print(f"实际批次大小: {actual_batch_size}")
    
    # 7. 训练循环
    best_reward = -float('inf')
    reward_history = []
    
    for epoch in range(config['rl_epochs']):
        print(f"\n--- RL Epoch {epoch+1}/{config['rl_epochs']} ---")
        
        # 选择批次
        batch_prompts = random.sample(prompts, actual_batch_size)
        
        try:
            # 训练步骤
            generated_poems, avg_reward, loss = ppo_trainer.train_epoch(batch_prompts)
            reward_history.append(avg_reward)
            
            print(f"Epoch {epoch+1} 完成: 奖励={avg_reward:.4f}, 损失={loss:.4f}")
            
            # 显示一些生成示例
            if epoch % 2 == 0:
                print("生成示例:")
                for i in range(min(2, len(batch_prompts))):
                    print(f"  '{batch_prompts[i]}' -> '{generated_poems[i]}'")
            
            # 保存最佳模型
            if avg_reward > best_reward:
                best_reward = avg_reward
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': base_model.state_dict(),
                    'reward': best_reward,
                    'config': config
                }, config['rl_model_save_path'])
                print(f"保存最佳模型，奖励: {best_reward:.4f}")
                
        except Exception as e:
            print(f"Epoch {epoch+1} 训练失败: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # 绘制奖励曲线
    if reward_history:
        plt.figure(figsize=(10, 6))
        plt.plot(reward_history)
        plt.title('强化学习训练奖励曲线')
        plt.xlabel('训练轮次')
        plt.ylabel('平均奖励')
        plt.grid(True)
        plt.savefig('rl_training_rewards.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    final_reward = best_reward if best_reward > -float('inf') else (reward_history[-1] if reward_history else 0)
    print(f"\n强化学习训练完成! 最终奖励: {final_reward:.4f}")
    return base_model, reward_history

# 优化的配置
optimized_config = {
    'base_model_path': 'best_poetry_model.pth',
    'vocab_file': 'vocab.txt',
    'data_file': 'test.csv',
    
    # 模型参数
    'vocab_size': 9270,
    'd_model': 512,
    'n_heads': 8,
    'num_layers': 6,
    'd_ff': 2048,
    'seq_len': 128,
    
    # 训练参数
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'rl_lr': 5e-6,  # 降低学习率
    'rl_epochs': 10,      # 增加轮次
    'rl_batch_size': 4,   # 减小批次大小
    'num_prompts': 20,
    'max_samples': 50,    # 增加数据量
    
    # 保存路径
    'rl_model_save_path': 'stable_rl_model.pth'
}

# 辅助函数（需要从之前的代码中复制）
def load_vocab(vocab_file):
    """加载词汇表"""
    with open(vocab_file, 'r', encoding='utf-8') as f:
        vocab = [line.strip() for line in f if line.strip()]
    
    token_to_id = {token: idx for idx, token in enumerate(vocab)}
    id_to_token = {idx: token for idx, token in enumerate(vocab)}
    
    print(f"词汇表加载完成，大小: {len(vocab)}")
    return vocab, token_to_id, id_to_token

def load_data_debug(csv_file, token_to_id, max_samples=None):
    """数据加载函数"""
    try:
        df = pd.read_csv(csv_file, encoding='utf-8-sig')
    except UnicodeDecodeError:
        df = pd.read_csv(csv_file, encoding='utf-8')
    
    if 'text1' in df.columns:
        texts = df['text1'].astype(str).tolist()
    else:
        texts = df.iloc[:, 0].astype(str).tolist()
    
    def clean_text(text):
        if pd.isna(text):
            return ""
        text = str(text)
        text = re.sub(r'[^\u4e00-\u9fa5，。！？]', '', text)
        return text.strip()
    
    cleaned_texts = [clean_text(text) for text in texts if len(clean_text(text)) >= 2]
    
    if max_samples and len(cleaned_texts) > max_samples:
        cleaned_texts = cleaned_texts[:max_samples]
    
    print(f"加载了 {len(cleaned_texts)} 条文本")
    return cleaned_texts

def generate_prompts_simple(training_texts, num_prompts=50):
    """生成提示词"""
    prompts = []
    
    for text in training_texts:
        if len(text) >= 2:
            length = min(random.randint(2, 4), len(text))
            prompts.append(text[:length])
    
    fixed_prompts = [
        "春风", "明月", "山水", "花开", "相思",
        "江南", "秋风", "饮酒", "登高", "夜雨"
    ]
    
    prompts.extend(fixed_prompts)
    unique_prompts = list(set(prompts))
    
    while len(unique_prompts) < num_prompts:
        need = num_prompts - len(unique_prompts)
        unique_prompts.extend(unique_prompts[:need])
        unique_prompts = list(set(unique_prompts))
    
    return unique_prompts[:num_prompts]

def causal_mask(seq_len):
    """创建因果掩码"""
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
    return mask.unsqueeze(0).unsqueeze(0)

def generate_poetry_simple(model, id_to_token, token_to_id, prompt="", max_len=20):
    """简化生成函数"""
    device = next(model.parameters()).device
    model.eval()
    
    input_ids = [token_to_id['[BOS]']]
    for char in prompt:
        if char in token_to_id:
            input_ids.append(token_to_id[char])
    
    with torch.no_grad():
        for _ in range(max_len):
            input_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)
            mask = causal_mask(len(input_ids)).to(device)
            
            logits, _ = model(input_tensor, mask)
            next_token_logits = logits[0, -1, :] / 0.8
            
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token_id = torch.multinomial(probs, num_samples=1).item()
            
            input_ids.append(next_token_id)
            
            if next_token_id == token_to_id['[EOS]']:
                break
    
    tokens = []
    for token_id in input_ids[1:]:
        if token_id == token_to_id['[EOS]']:
            break
        tokens.append(id_to_token.get(token_id, '[UNK]'))
    
    return ''.join(tokens)

if __name__ == "__main__":
    # 导入模型类
    try:
        from decoder_only import DecoderOnlyTransformer
    except ImportError:
        # 简化模型定义
        class DecoderOnlyTransformer(nn.Module):
            def __init__(self, vocab_size, d_model, n_heads, num_layers, d_ff, max_seq_len=5000, dropout=0.1):
                super().__init__()
                self.d_model = d_model
                self.token_embedding = nn.Embedding(vocab_size, d_model)
                self.lm_head = nn.Linear(d_model, vocab_size)
                
            def forward(self, x, mask=None):
                x = self.token_embedding(x) * math.sqrt(self.d_model)
                logits = self.lm_head(x)
                return logits, None
    
    print("启动稳定的强化学习训练...")
    model, rewards = stable_rl_training(optimized_config)
    
    if model is not None and rewards:
        print(f"\n训练完成! 最终奖励: {rewards[-1]:.4f}")
        print("生成测试:")
        vocab, token_to_id, id_to_token = load_vocab(optimized_config['vocab_file'])
        test_prompts = ["春风", "明月", "相思"]
        for prompt in test_prompts:
            generated = generate_poetry_simple(model, id_to_token, token_to_id, prompt)
            print(f"  '{prompt}' -> '{generated}'")