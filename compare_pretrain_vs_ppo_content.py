import torch
import numpy as np
import pandas as pd
from collections import Counter
import jieba
import matplotlib.pyplot as plt
from tqdm import tqdm

class PoetryModelComparator:
    def __init__(self, model1_path, model2_path, vocab_file, device='cuda'):
        self.device = device
        self.vocab, self.token_to_id, self.id_to_token = self.load_vocab(vocab_file)
        
        # 加载两个模型
        self.model1 = self.load_model(model1_path, "基础模型")
        self.model2 = self.load_model(model2_path, "RL优化模型")
        
    def load_vocab(self, vocab_file):
        """加载词汇表"""
        with open(vocab_file, 'r', encoding='utf-8') as f:
            vocab = [line.strip() for line in f if line.strip()]
        token_to_id = {token: idx for idx, token in enumerate(vocab)}
        id_to_token = {idx: token for idx, token in enumerate(vocab)}
        return vocab, token_to_id, id_to_token
    
    def load_model(self, model_path, model_name):
        """加载模型"""
        print(f"加载{model_name}: {model_path}")
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # 创建模型实例
        model = DecoderOnlyTransformer(
            vocab_size=len(self.vocab),
            d_model=512,
            n_heads=8,
            num_layers=6,
            d_ff=2048,
            max_seq_len=128
        ).to(self.device)
        
        # 加载权重
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
            
        model.eval()
        print(f"{model_name}加载完成")
        return model
    
    def generate_poetry(self, model, prompt, max_len=50, temperature=0.8):
        """生成诗词"""
        device = next(model.parameters()).device
        
        input_ids = [self.token_to_id['[BOS]']]
        for char in prompt:
            if char in self.token_to_id:
                input_ids.append(self.token_to_id[char])
        
        with torch.no_grad():
            for _ in range(max_len):
                input_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)
                mask = causal_mask(len(input_ids)).to(device)
                
                logits, _ = model(input_tensor, mask)
                next_token_logits = logits[0, -1, :] / temperature
                
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token_id = torch.multinomial(probs, num_samples=1).item()
                
                input_ids.append(next_token_id)
                
                if next_token_id == self.token_to_id['[EOS]']:
                    break
        
        # 转换为文本
        tokens = []
        for token_id in input_ids[1:]:
            if token_id == self.token_to_id['[EOS]']:
                break
            tokens.append(self.id_to_token.get(token_id, '[UNK]'))
        
        return ''.join(tokens)
    
    def compare_models(self, test_prompts, num_samples_per_prompt=3):
        """全面比较两个模型"""
        print("开始模型比较...")
        
        results = {
            'prompt': [],
            'model1_output': [],
            'model2_output': [],
            'model1_metrics': [],
            'model2_metrics': []
        }
        
        all_model1_poems = []
        all_model2_poems = []
        
        for prompt in tqdm(test_prompts, desc="生成诗词"):
            for i in range(num_samples_per_prompt):
                # 生成诗词
                poem1 = self.generate_poetry(self.model1, prompt)
                poem2 = self.generate_poetry(self.model2, prompt)
                
                # 计算指标
                metrics1 = self.calculate_metrics(poem1, prompt)
                metrics2 = self.calculate_metrics(poem2, prompt)
                
                results['prompt'].append(prompt)
                results['model1_output'].append(poem1)
                results['model2_output'].append(poem2)
                results['model1_metrics'].append(metrics1)
                results['model2_metrics'].append(metrics2)
                
                all_model1_poems.append(poem1)
                all_model2_poems.append(poem2)
        
        # 总体统计
        overall_comparison = self.overall_comparison(all_model1_poems, all_model2_poems)
        
        return pd.DataFrame(results), overall_comparison
    
    def calculate_metrics(self, poem, prompt):
        """计算单个诗词的指标"""
        metrics = {}
        
        # 1. 长度指标
        metrics['length'] = len(poem)
        
        # 2. 结构完整性
        lines = [line for line in poem.split('。') if line.strip()]
        metrics['line_count'] = len(lines)
        metrics['has_ending_punctuation'] = 1 if poem.endswith('。') else 0
        
        # 3. 词汇多样性
        unique_chars = len(set(poem))
        metrics['diversity'] = unique_chars / len(poem) if len(poem) > 0 else 0
        
        # 4. 古典特征
        classical_words = {'风', '月', '花', '山', '水', '云', '雨', '春', '秋', '夜', '江', '天'}
        classical_count = sum(1 for char in poem if char in classical_words)
        metrics['classical_ratio'] = classical_count / len(poem) if len(poem) > 0 else 0
        
        # 5. 与提示词相关性
        if prompt:
            prompt_in_poem = any(word in poem for word in prompt)
            metrics['prompt_relevance'] = 1 if prompt_in_poem else 0
        else:
            metrics['prompt_relevance'] = 0
            
        # 6. 重复度惩罚
        char_counts = Counter(poem)
        max_repeat = max(char_counts.values()) if char_counts else 0
        metrics['repetition_penalty'] = max(0, 1 - (max_repeat / len(poem))) if len(poem) > 0 else 1
        
        # 7. 综合得分
        metrics['overall_score'] = (
            metrics['diversity'] * 0.3 +
            metrics['classical_ratio'] * 0.2 +
            metrics['prompt_relevance'] * 0.2 +
            metrics['repetition_penalty'] * 0.3
        )
        
        return metrics
    
    def overall_comparison(self, poems1, poems2):
        """总体比较"""
        comparison = {}
        
        # 计算平均指标
        metrics1 = [self.calculate_metrics(poem, '') for poem in poems1]
        metrics2 = [self.calculate_metrics(poem, '') for poem in poems2]
        
        for key in metrics1[0].keys():
            avg1 = np.mean([m[key] for m in metrics1])
            avg2 = np.mean([m[key] for m in metrics2])
            comparison[key] = {'model1': avg1, 'model2': avg2, 'difference': avg2 - avg1}
        
        return comparison
    
    def human_evaluation(self, comparison_df, num_examples=10):
        """人工评估界面"""
        print("\n=== 人工评估 ===")
        print("请为每对诗词评分 (1-5分)，5分为最佳")
        print("评分标准:")
        print("1分: 质量很差，语义不通")
        print("2分: 质量较差，有较多问题") 
        print("3分: 质量一般，基本通顺")
        print("4分: 质量良好，有意境")
        print("5分: 质量优秀，有意境且优美")
        
        human_scores = []
        
        # 随机选择示例进行评估
        sample_df = comparison_df.sample(min(num_examples, len(comparison_df)))
        
        for idx, row in sample_df.iterrows():
            print(f"\n提示词: '{row['prompt']}'")
            print(f"基础模型: {row['model1_output']}")
            print(f"RL优化模型: {row['model2_output']}")
            
            while True:
                try:
                    score1 = int(input("基础模型评分 (1-5): "))
                    score2 = int(input("RL优化模型评分 (1-5): "))
                    if 1 <= score1 <= 5 and 1 <= score2 <= 5:
                        break
                    else:
                        print("请输入1-5之间的整数")
                except ValueError:
                    print("请输入有效的整数")
            
            human_scores.append({
                'prompt': row['prompt'],
                'model1_score': score1,
                'model2_score': score2,
                'preference': 'model1' if score1 > score2 else 'model2' if score2 > score1 else 'tie'
            })
        
        return pd.DataFrame(human_scores)
    
    def visualize_comparison(self, comparison_df, overall_stats):
        """可视化比较结果"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 指标对比
        metrics_to_plot = ['diversity', 'classical_ratio', 'overall_score', 
                          'prompt_relevance', 'repetition_penalty']
        
        for i, metric in enumerate(metrics_to_plot):
            ax = axes[i//3, i%3]
            model1_scores = [m[metric] for m in comparison_df['model1_metrics']]
            model2_scores = [m[metric] for m in comparison_df['model2_metrics']]
            
            positions = [1, 2]
            ax.boxplot([model1_scores, model2_scores], positions=positions)
            ax.set_xticks(positions)
            ax.set_xticklabels(['基础模型', 'RL优化模型'])
            ax.set_title(f'{metric} 对比')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 打印总体统计
        print("\n=== 总体统计 ===")
        stats_df = pd.DataFrame(overall_stats).T
        print(stats_df)

# 使用示例
def main():
    # 测试提示词
    test_prompts = [
        "春风", "明月", "山水", "花开", "相思",
        "江南", "秋风", "饮酒", "登高", "夜雨",
        "孤舟", "白云", "青山", "流水", "落花"
    ]
    
    # 初始化比较器
    comparator = PoetryModelComparator(
        model1_path='best_poetry_model.pth',
        model2_path='stable_rl_model.pth', 
        vocab_file='vocab.txt'
    )
    
    # 进行比较
    comparison_df, overall_stats = comparator.compare_models(test_prompts)
    
    # 保存结果
    comparison_df.to_csv('model_comparison_results.csv', index=False, encoding='utf-8-sig')
    
    # 可视化
    comparator.visualize_comparison(comparison_df, overall_stats)
    
    # 人工评估
    human_scores = comparator.human_evaluation(comparison_df, num_examples=8)
    human_scores.to_csv('human_evaluation.csv', index=False, encoding='utf-8-sig')
    
    # 分析人工评估结果
    model1_avg = human_scores['model1_score'].mean()
    model2_avg = human_scores['model2_score'].mean()
    preference_counts = human_scores['preference'].value_counts()
    
    print(f"\n=== 人工评估总结 ===")
    print(f"基础模型平均分: {model1_avg:.2f}")
    print(f"RL优化模型平均分: {model2_avg:.2f}")
    print(f"偏好统计:")
    for pref, count in preference_counts.items():
        model_name = '基础模型' if pref == 'model1' else 'RL优化模型' if pref == 'model2' else '平局'
        print(f"  {model_name}: {count}次")

if __name__ == "__main__":
    # 需要导入模型类
    try:
        from decoder_only import DecoderOnlyTransformer, causal_mask
    except ImportError:
        # 简化定义
        class DecoderOnlyTransformer:
            def __init__(self, vocab_size, d_model, n_heads, num_layers, d_ff, max_seq_len=5000, dropout=0.1):
                pass
        
        def causal_mask(seq_len):
            return torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
    
    main()