# Diy-LLM：从零实现的古诗词生成模型全链路项目(原 PoetryLLM / Poetry-PPO)

一个教学级的小型语言模型项目：从零手写 **字符级 Decoder-Only Transformer（约 5000 万参数）**，在古诗词语料上完成 **预训练 → PPO 强化学习优化 → KV Cache 推理加速 → 多 Agent 应用落地 → 自动 + 人工双轨评估** 的完整闭环。

本项目的核心价值不在于生成质量，而在于一条腿走完现代 LLM 的标准生命周期：**自建词表 → 预训练 → RLHF（规则奖励版）→ 推理优化 → 评估对比 → Agent 化落地**，适合作为深度学习/大模型方向的学习与面试项目。

---

## 目录

- [项目特性](#项目特性)
- [系统架构](#系统架构)
- [文件结构](#文件结构)
- [环境依赖](#环境依赖)
- [快速开始](#快速开始)
- [模型配置](#模型配置)
- [PPO 奖励函数设计](#ppo-奖励函数设计)
- [多 Agent 系统](#多-agent-系统)
- [评估结果](#评估结果)
- [已知问题与改进方向](#已知问题与改进方向)
- [常见问题](#常见问题)

---

## 项目特性

- **手搓 Transformer**：完整的 Pre-LN Decoder-Only 架构（多头注意力、GELU FFN、残差连接、因果掩码），不依赖 `nn.Transformer` 或 HuggingFace；
- **自定义位置编码**：`BalancedPositionalEncoding` 对正弦位置编码做 `√d_model/2` 缩放，平衡 token embedding（×√d_model）与位置信号的幅值量级；
- **规则奖励 PPO**：无需训练奖励模型，用长度、相关性、结构、多样性、古典意象五维规则奖励驱动策略优化；
- **KV Cache 推理**：增量解码避免 O(L²) 重复计算，支持半精度（fp16）推理与批量生成接口；
- **大小模型分工的多 Agent 系统**：本地 PPO 小模型负责诗词生成，云端大模型（阿里云 DashScope qwen-max）负责赏析与修改，LangGraph 状态图编排工作流；
- **双轨评估**：7 项自动指标 + 人工盲评对比预训练模型与 PPO 模型。

---

## 系统架构

```
train.csv（古诗词语料，text1 列）
   │
build_vocab.py ──→ vocab.txt（字符级词表，9270 个 token）
   │
decoder_only.py / pretrain.py ──→ best_poetry_model.pth（预训练基座）
   │
ppo_train.py ──→ stable_rl_model.pth（PPO 规则奖励微调）
   │
   ├─ predict_ppo.py                    基础推理（逐 token 全量前向）
   ├─ kvcahe_ppo.py                     KV Cache 高效推理（增量解码）
   ├─ agent_kvcache_ppo.py              LangGraph 三 Agent 交互系统
   └─ compare_pretrain_vs_ppo_content.py   模型对比评估
```

---

## 文件结构

| 文件 | 作用 |
|---|---|
| `build_vocab.py` | 数据清洗、字符频率统计、构建词表（`vocab.txt`、`token_to_id.json`、`char_frequency.json`、`character_report.json`、覆盖度分析） |
| `decoder_only.py` | 模型定义 + 预训练（`PoetryDataset`、`DecoderOnlyTransformer`、训练循环、生成函数） |
| `pretrain.py` | 与 `decoder_only.py` 基本相同的训练脚本（历史副本，见[已知问题](#已知问题与改进方向)） |
| `ppo_train.py` | PPO 强化学习训练（`FixedPPOTrainer`：规则奖励、clip 裁剪、熵奖励） |
| `predict_ppo.py` | 基础推理器（`PPOPoetryGenerator`：单次/批量/交互式生成） |
| `kvcahe_ppo.py` | KV Cache 推理器（`KVCachePoetryGenerator` + `CachedDecoderOnlyTransformer`，兼容预训练权重） |
| `agent_kvcache_ppo.py` | 多 Agent 系统（生成/赏析/修改三个 Agent + LangGraph 工作流 + 规则路由） |
| `compare_pretrain_vs_ppo_content.py` | 模型对比评估（`PoetryModelComparator`：自动指标、可视化、人工评估） |
| `test_part.csv` | 测试用古诗词数据样例（`text1` 列） |
| `vocab_part.txt` | 部分词表样例（前 7 行为特殊标记 `[PAD]/[UNK]/[BOS]/[EOS]/[CLS]/[SEP]/[MASK]`） |
| `agent_content.txt` | 多 Agent 系统一次完整交互的运行日志 |
| `compare_pretrain_ppo_content.txt` | 模型对比评估的运行日志 |
| `文件关系分析.md` / `文件关系图.md` | 项目文件依赖关系说明与 Mermaid 可视化 |

---

## 模型权重

| 平台 | 地址 |
|---|---|
| Hugging Face | https://huggingface.co/cnfeichen/Diy-LLM |
| ModelScope | https://modelscope.cn/models/cnfeichen/Diy-LLM |

含三个检查点:`best_poetry_model.pth`(预训练基座)、`stable_rl_model.pth`(PPO)、`grpo_model_v2.pth`(GRPO,对比实测最佳 RL 版本)。

## 环境依赖

```bash
pip install torch pandas numpy matplotlib tqdm jieba
# 多 Agent 系统额外依赖：
pip install langgraph langchain-core dashscope
```

硬件建议：预训练/PPO 训练需要 CUDA GPU；纯推理可在 CPU 上运行。

---

## 快速开始

### 1. 构建词表

```bash
python build_vocab.py
# 输入: train.csv（text1 列为古诗词文本）
# 输出: poetry_vocab/vocab.txt、token_to_id.json、char_frequency.json 等
```

### 2. 预训练

```bash
python decoder_only.py
# 输出: best_poetry_model.pth（含 config，供下游重建模型）
#      training_losses.json
```

### 3. PPO 强化学习优化

```bash
python ppo_train.py
# 输入: best_poetry_model.pth + vocab.txt + test.csv（提取提示词）
# 输出: stable_rl_model.pth、rl_training_rewards.png
```

### 4. 模型对比评估

```bash
python compare_pretrain_vs_ppo_content.py
# 输出: model_comparison_results.csv、human_evaluation.csv、model_comparison.png
```

### 5. 推理生成

```bash
# 方式一：基础推理
python predict_ppo.py

# 方式二：KV Cache 高效推理（含性能对比测试）
python kvcahe_ppo.py
```

### 6. 多 Agent 交互系统

```bash
# 先设置 API 密钥（见下方"安全提示"）
export DASHSCOPE_API_KEY="your-api-key"
python agent_kvcache_ppo.py
```

交互命令：

| 输入 | 行为 |
|---|---|
| `春天` | 以"春天"为提示词生成诗词（默认随后自动赏析） |
| `赏析` | 对当前诗词进行赏析（qwen-max） |
| `修改 格式规整` | 按指定要求修改当前诗词（qwen-max） |
| `设置 max_len=50` / `设置 temperature=0.8` | 调整生成参数 |
| `帮助` / `结束` | 查看命令 / 退出 |

> **安全提示**：请勿将真实 API 密钥硬编码进代码。`agent_kvcache_ppo.py` 中的 `DASHSCOPE_API_KEY` 应改为 `os.environ["DASHSCOPE_API_KEY"]` 读取，且任何已提交到仓库/外发的密钥应立即在阿里云控制台吊销轮换。

---

## 模型配置

| 超参数 | 值 |
|---|---|
| 词表大小 | 9270（字符级，含 7 个特殊标记） |
| d_model / 注意力头数 | 512 / 8 |
| 层数 / d_ff | 6 / 2048 |
| 最大序列长度 | 128 |
| 总参数量 | 约 5000 万 |
| 预训练优化 | AdamW（lr=1e-4, weight_decay=0.01）+ CosineAnnealingLR（按 step），梯度累积 2 步，梯度裁剪 1.0，batch_size=32，50 epochs |
| PPO 优化 | lr=5e-6，clip_epsilon=0.1，entropy_coef=0.001，10 epochs，batch_size=4 |

---

## PPO 奖励函数设计

纯规则打分（无需奖励模型），单首诗词满分 1.0：

| 奖励项 | 权重 | 目的 |
|---|---|---|
| 长度得分（25 字封顶 0.3） | ≤ 0.3 | 防止生成过短 |
| 提示词相关性（提示字出现在诗中） | +0.3 | 保证扣题 |
| 结构完整性（含「。」且 ≥2 句） | +0.3 | 保证诗体完整 |
| 字符多样性（unique 比例） | × 0.2 | 抑制复读 |
| 古典意象字比例（风花雪月山水云雨春秋夜） | × 0.1 | 引导古典风格 |

训练流程：旧策略 `no_grad` 采样记录 `old_log_probs` → 规则奖励 → 新策略前向 → `ratio = exp(new − old)` → clip 裁剪 → 加熵奖励 → 反向更新。

---

## 多 Agent 系统

**架构**（`agent_kvcache_ppo.py`）：

```
用户输入
   │
generation（PoetryGenerationAgent，本地 PPO 模型 + KV Cache）
   │ 条件路由（规则关键词匹配）
   ├─→ appreciation（PoetryAppreciationAgent，qwen-max 赏析）
   │        │ 条件路由
   │        └─→ revision ──→ final_response ──→ END
   ├─→ revision（PoetryRevisionAgent，qwen-max 修改）──→ final_response ──→ END
   └─→ final_response ──→ END
```

- 状态管理：`AgentState`（TypedDict + `add_messages` reducer）+ `MemorySaver` checkpoint；
- 路由：`Router.should_appreciate / should_revise` 基于用户消息关键词（"赏析/修改/优化…"）的规则路由；
- 分工理念：**本地小模型做生成（低成本、可离线），云端大模型做理解与润色（赏析、修改）**，与工业界端云协同思路一致。

---

## 评估结果

基于 15 个提示词 × 3 次采样的对比（详见 `compare_pretrain_ppo_content.txt`）：

| 指标 | 预训练模型 | PPO 模型 | 差值 |
|---|---|---|---|
| 字符多样性 diversity | 0.844 | 0.853 | +0.009 |
| 古典意象比 classical_ratio | 0.100 | 0.116 | +0.016 |
| 综合得分 overall_score | 0.550 | 0.555 | +0.005 |
| **人工盲评平均分（1-5）** | **2.38** | **3.12** | **+0.74** |

人工盲评 8 组：PPO 模型 5 胜、预训练模型 2 胜、平局 1 组。结论：规则奖励 PPO 带来**可感知但有限**的提升——这与 RLHF 领域普遍经验一致（规则奖励天花板低、易被 hack）。

---

## 已知问题与改进方向

本项目是学习/练手性质，保留了若干"够用的简化"，同时也是很好的进阶改进切入点：

1. **PPO 实现简化**：
   - 无 Critic / 优势函数（reward 直接当 advantage，无 GAE 减方差）；
   - 无参考模型 KL 惩罚（仅靠小学习率和 clip 防策略跑偏）；
   - new/old log_prob 对应两条不同轨迹——标准 PPO 应缓存采样 actions 后对同一轨迹重新打分；
2. **KV Cache 位置编码 bug**：增量解码时 `pos_encoding` 取 `pe[:, :seq_len]`，seq_len=1 导致每个新 token 都拿到位置 0 的编码，应传入 offset 取 `pe[:, offset:offset+seq_len]`；
3. **批量生成未真正并行**：`batch_generate_with_kv_cache` 实际逐序列循环，可实现真正的 batched KV Cache；
4. **多 Agent 状态隔离**：`process_request` 固定 `thread_id="1"`，多轮请求共享 checkpoint，路由可能扫描到历史消息误触发，应为每轮会话生成独立 thread_id；
5. **代码组织**：`pretrain.py` 与 `decoder_only.py` 重复、`load_vocab` 在 5 个文件中复制粘贴，应抽取公共模块（`models.py`、`utils.py`）；`kvcahe_ppo.py` 文件名拼写应为 `kvcache_ppo.py`；
6. **安全**：API 密钥不得硬编码进源码（见"安全提示"）。

---

## 常见问题

**Q：为什么用字符级分词而不是 BPE？**
中文做 char 级词表小（<1 万）、无 OOV、诗句短，对这个规模的项目是合理选择；代价是序列语义粒度粗，长文本任务不适用。

**Q：为什么 PPO 提升有限？**
规则奖励只覆盖了长度、结构、意象等浅层特征，无法评价"诗意"这种深层质量，这正是需要训练奖励模型（RM）的原因——本项目用规则奖励换取了流程的简洁可跑通。

**Q：为什么赏析和修改要调云端大模型？**
小模型（50M）能学诗词的"形"（字词搭配、格律表面特征），但做不了语义级的赏析与定向修改。让本地小模型负责生成、云端大模型负责理解与润色，是成本与能力的合理分工。

---

## License

本项目仅供学习与研究使用。
