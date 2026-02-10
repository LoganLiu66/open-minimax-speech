# ConditionalCFM Training for MiniMax-Speech

本目录包含基于 [MiniMax-Speech 论文](https://arxiv.org/abs/2505.07916) 的 Conditional Flow Matching (CFM) 训练实现。

## 概述

MiniMax-Speech 是一个基于自回归 Transformer 的 TTS 模型，使用 Flow Matching 技术生成高质量语音。本实现提供了 ConditionalCFM 模型的训练框架。

## 主要特性

根据论文实现的关键特性：

1. **Conditional Flow Matching**: 使用条件流匹配进行语音合成
2. **Classifier-Free Guidance (CFG)**: 支持无分类器引导，平衡模态覆盖和样本保真度
3. **Cosine Time Scheduler**: 支持余弦时间调度器，改善训练稳定性
4. **多说话人支持**: 可配置说话人嵌入维度

## 核心修改

相比基础 CFM 实现，根据论文做了以下修改：

### 1. 时间调度器 (Time Scheduler)
```python
if self.t_scheduler == 'cosine':
    t = 1 - torch.cos(t * 0.5 * torch.pi)
```
- 支持线性和余弦两种时间调度策略
- 余弦调度器可以改善训练稳定性和采样质量

### 2. Classifier-Free Guidance
```python
# Training时随机dropout条件
if self.training_cfg_rate > 0:
    cfg_mask = torch.rand(b, device=x1.device) > self.training_cfg_rate
    mu = mu * cfg_mask.view(-1, 1, 1)
    spks = spks * cfg_mask.view(-1, 1)
    cond = cond * cfg_mask.view(-1, 1, 1)

# Inference时结合有条件和无条件预测
dphi_dt = ((1.0 + self.inference_cfg_rate) * dphi_dt - 
           self.inference_cfg_rate * cfg_dphi_dt)
```
- 训练时以一定概率随机丢弃条件信息
- 推理时结合条件和无条件预测，提高生成质量

### 3. 说话人编码器集成
- 支持可学习的说话人编码器特征
- 实现零样本 TTS 和单样本语音克隆

## 文件结构

```
flow_matching/
├── flow_matching.py           # ConditionalCFM 模型定义
├── flow_matching_trainer.py   # 训练脚本
├── estimator.py               # 条件解码器（速度估计器）
└── README.md                  # 本文档
```

## 使用方法

### 1. 准备数据

准备数据列表文件（JSON Lines格式）：

```json
{"audio_file": "/path/to/audio1.wav", "sid": 0, "lang": "en", "text": "Hello world"}
{"audio_file": "/path/to/audio2.wav", "sid": 1, "lang": "en", "text": "This is a test"}
```

### 2. 配置文件

参考 `configs/flow_matching_config.yaml`:

```yaml
trainer:
  learning_rate: 2e-4
  epochs: 200
  batch_size: 16

model:
  cfm_params:
    t_scheduler: "cosine"
    training_cfg_rate: 0.1
    inference_cfg_rate: 1.0
```

关键参数说明：
- `t_scheduler`: 时间调度器类型 ("linear" 或 "cosine")
- `training_cfg_rate`: CFG 训练丢弃率（推荐 0.1-0.3）
- `inference_cfg_rate`: CFG 推理强度（推荐 0.5-1.5）
- `sigma_min`: ODE 求解器的最小噪声标准差

### 3. 训练

**单卡训练：**
```bash
python minimaxspeech/modules/flow_matching/flow_matching_trainer.py \
    --config configs/flow_matching_config.yaml
```

**多卡训练（DDP）：**
```bash
export CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7'
torchrun --nproc_per_node=8 \
    minimaxspeech/modules/flow_matching/flow_matching_trainer.py \
    --config configs/flow_matching_config.yaml
```

### 4. 推理

训练完成后，可以使用 ConditionalCFM 模型进行推理：

```python
from minimaxspeech.modules.flow_matching.flow_matching import ConditionalCFM
import torch

# 加载模型
model = ConditionalCFM(in_channels=80, cfm_params=cfm_params)
checkpoint = torch.load('output/flow_matching/checkpoint_best.pth')
model.load_state_dict(checkpoint['model'])
model.eval()

# 生成语音
with torch.no_grad():
    generated_mel = model(
        mu=condition_features,  # 来自编码器的条件特征
        mask=mask,
        n_timesteps=10,
        temperature=1.0,
        spks=speaker_embedding,
        cond=additional_condition
    )
```

## 实现细节

### 损失函数

使用 MSE 损失训练速度估计器：

```python
loss = F.mse_loss(pred * mask, u * mask, reduction="sum") / (torch.sum(mask) * u.shape[1])
```

其中：
- `pred`: 估计器预测的速度场
- `u`: 真实速度场 (x1 - (1 - sigma_min) * z)
- `mask`: 用于处理变长序列

### ODE 求解器

使用 Euler 方法求解 ODE：

```python
for step in range(1, len(t_span)):
    dphi_dt = self.estimator(x, mask, mu, t, spks, cond)
    x = x + dt * dphi_dt
    t = t + dt
```

## 论文参考

**MiniMax-Speech: Intrinsic Zero-Shot Text-to-Speech with a Learnable Speaker Encoder**

- arXiv: https://arxiv.org/abs/2505.07916
- Authors: Bowen Zhang, Congchao Guo, et al.
- 提交日期: 2025-05-12

主要创新点：
1. 可学习的说话人编码器，无需参考音频转录
2. Flow-VAE 增强音频质量
3. 支持 32 种语言
4. 在客观和主观评测中达到 SOTA 性能

## 扩展应用

基于训练好的模型，可以实现：

1. **情感控制**: 通过 LoRA 微调实现任意语音情感控制
2. **Text-to-Voice (T2V)**: 直接从文本描述合成音色特征
3. **专业语音克隆 (PVC)**: 使用额外数据微调音色特征

## 注意事项

1. **数据质量**: 训练数据质量直接影响生成效果，建议使用高质量、多样化的数据
2. **CFG 调优**: `training_cfg_rate` 和 `inference_cfg_rate` 需要根据具体任务调整
3. **计算资源**: Flow Matching 训练相对高效，但仍建议使用多 GPU 加速
4. **时间步数**: 推理时 `n_timesteps` 影响质量和速度的平衡，推荐 5-20 步

## 引用

如果使用本实现，请引用 MiniMax-Speech 论文：

```bibtex
@article{zhang2025minimax,
  title={MiniMax-Speech: Intrinsic Zero-Shot Text-to-Speech with a Learnable Speaker Encoder},
  author={Zhang, Bowen and Guo, Congchao and Yang, Geng and Yu, Hang and Zhang, Haozhe and Lei, Heidi and Mai, Jialong and Yan, Junjie and Yang, Kaiyue and Yang, Mingqi and others},
  journal={arXiv preprint arXiv:2505.07916},
  year={2025}
}
```

## 许可证

请遵循项目根目录的许可证要求。
