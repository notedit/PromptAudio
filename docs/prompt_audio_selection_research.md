# TTS Prompt Audio 选取策略调研报告

## 一、项目目标

从一段 5 分钟以内的音频中，自动选取 10 秒以内的最优片段作为 CosyVoice3 的 prompt audio，需满足：
1. 不切断字/词
2. 优先选质量高、情感韵律变化大的部分
3. 选出的音频首尾各有 ~150ms 静音
4. 支持多段拼接策略（有条件拼接）及多 prompt 独立融合

---

## 二、CosyVoice3 对 Prompt Audio 的要求

### 2.1 官方技术规格

| 参数 | 要求 |
|------|------|
| 采样率 | >= 16kHz（推理使用 `load_wav('prompt.wav', 16000)`） |
| 格式 | WAV（最佳）、MP3、M4A |
| 声道 | 单声道 |
| 位深 | 16 bit |
| 文件大小 | <= 10MB |
| 推荐时长 | 3-10 秒（零样本克隆），阿里云 API 推荐 10-20 秒 |
| 有效语音占比 | >= 60% |
| 说话人 | 单说话人 |
| 音量标准化 | `normalized_wav = raw_wav / max(raw_wav) * 0.6` |

### 2.2 韵律敏感性（核心）

CosyVoice3 的 speech tokenizer 以 **25 tokens/秒** 编码，捕获了情感、韵律、口音等副语言学信息。**prompt 的韵律模式会直接传递到合成音频**：
- 平淡 prompt → 平淡合成
- 高情感 prompt → 高情感合成
- 论文指出 "emotion expression can adversely affect pronunciation"，风格和内容在优化过程中存在竞争关系

### 2.3 CosyVoice3 训练数据预处理流水线（参考）

论文描述的六步预处理：
1. 说话人分离 + VAD + 音频事件检测 → 生成 <30s 的说话人级别片段
2. MossFormer2 降噪，筛除首尾不完整词的片段
3. Faster-Whisper Large-V3 + 多模型交叉验证（pair-wise WER < 15%）
4. Montreal Forced Aligner 对齐 → 按阈值调整标点
5. 音量标准化 `raw_wav / max(raw_wav) * 0.6`
6. 过滤：丢弃音频-文本长度比最小 1% 和最大 5%

---

## 三、推荐的端到端 Pipeline 架构

```
5分钟音频输入
    |
    v
[Step 1: 预处理]
    ├── 重采样至 16kHz 单声道
    ├── 音量标准化: wav / max(wav) * 0.6
    └── 降噪（可选）: MossFormer2 或 Demucs
    |
    v
[Step 2: ASR 转录 + 强制对齐]
    ├── WhisperX 自动转录 + 词级强制对齐
    ├── 获得词级时间戳: [{word, start, end}, ...]
    └── 按句子/标点重新分组为句级时间戳
    |
    v
[Step 3: 候选片段生成] — 三条路径并行
    |
    ├─ [路径A: 单段连续]
    |   ├── 遍历连续句子组合，找 8-10 秒的候选段
    |   ├── 边界对齐到词级时间戳（不切断任何词）
    |   └── 150ms 静音填充（优先自然静音，不足则补充）
    |
    ├─ [路径B: 相邻句拼接]
    |   ├── 选 2-3 段高质量短句（各 >= 3 秒）拼至 8-10 秒
    |   ├── 拼接约束：原始间距 < 3s, F0 均值差 < 15%, 能量差 < 20%
    |   └── 拼接处插入 150-300ms 静音模拟自然停顿
    |
    └─ [路径C: 多 prompt 独立]
        ├── 选 3-5 段各 3-5 秒高质量片段，不拼接
        └── 用于 speaker embedding 加权融合（Multi-Prompt Fusion）
    |
    v
[Step 4: 质量门控]  (淘汰劣质片段，路径 A/B/C 共用)
    ├── DNSMOS OVRL >= 3.5
    ├── 削波率 < 1%
    ├── HNR >= 15 dB
    └── 有效语音占比 >= 60%
    |
    v
[Step 5: 综合评分排序]
    ├── 质量分 Q (权重 0.6) + 韵律丰富度分 P (权重 0.4)
    ├── 路径 B 额外扣除拼接罚分
    └── 路径 A/B 取 Final_Score 最高者作为单 prompt 输出
    |
    v
[输出]
    ├── 单 prompt 最优段（路径 A 或 B 最高分）
    └── 多 prompt 候选集（路径 C，用于 embedding 融合场景）
```

---

## 四、各环节技术方案详解

### 4.1 词/句边界检测 — 避免切断词语

#### 推荐方案：WhisperX（端到端，首选）

WhisperX 在 Whisper 转录基础上使用 Wav2Vec2 CTC 强制对齐，输出精确的词级时间戳。

```python
import whisperx

# 转录
model = whisperx.load_model("large-v3", device="cuda")
audio = whisperx.load_audio("input.wav")
result = model.transcribe(audio, language="zh")

# 强制对齐
model_a, metadata = whisperx.load_align_model(language_code="zh", device="cuda")
result = whisperx.align(
    result["segments"], model_a, metadata, audio, device="cuda",
    return_char_alignments=True  # 中文建议开启字符级对齐
)
# result["segments"][i]["words"] = [{"word": "你好", "start": 0.5, "end": 0.8}, ...]
```

**中文支持**：使用 `jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn` 对齐模型，中文被归类为 `LANGUAGES_WITHOUT_SPACES`，WhisperX 做了特殊处理。

#### 备选方案：MFA（精度最高）

适合 WhisperX 对齐不准时二次精修：

```bash
conda install -c conda-forge montreal-forced-aligner
mfa model download acoustic mandarin_mfa
mfa model download dictionary mandarin_china_mfa
mfa align /path/to/corpus mandarin_china_mfa.dict mandarin_mfa /path/to/output
```

输出 TextGrid 文件，包含词级和音素级时间戳。

#### 工具对比

| 工具 | 中文词级精度 | 易用性 | 需要转写文本 | 推荐场景 |
|------|------------|--------|-------------|---------|
| WhisperX | 良好 | 高（端到端） | 否 | **首选** |
| MFA | 最高 | 中 | 是 | 精度优先 |
| torchaudio CTC | 良好 | 中 | 是 | PyTorch 生态 |
| stable-ts | 中等 | 高 | 否 | 辅助验证 |

### 4.2 静音检测与 150ms 填充

#### 方案 A：利用自然静音（推荐）

使用 Silero VAD，设置 `speech_pad_ms=150` 自动在语音段前后扩展 150ms：

```python
import torch

model, utils = torch.hub.load('snakers4/silero-vad', 'silero_vad')
get_speech_timestamps, _, read_audio, _, _ = utils

wav = read_audio('input.wav', sampling_rate=16000)
speech_timestamps = get_speech_timestamps(
    wav, model,
    sampling_rate=16000,
    threshold=0.5,
    min_speech_duration_ms=250,
    min_silence_duration_ms=100,
    speech_pad_ms=150,  # 前后各扩展 150ms
    return_seconds=True,
)
```

#### 方案 B：人工补充静音（自然静音不足时）

```python
from pydub import AudioSegment

clip = AudioSegment.from_file("segment.wav")
silence_150ms = AudioSegment.silent(duration=150, frame_rate=16000)

# 检查首尾静音是否已足够
# 不足时补充差额
padded = silence_150ms + clip + silence_150ms
padded.export("padded_segment.wav", format="wav")
```

**推荐策略**：优先方案 A（自然静音过渡更适合 TTS 模型学习），仅当自然静音 < 150ms 时用方案 B 补足差额。

### 4.3 音频质量评估

使用**无参考（Non-Intrusive）指标**，因为没有干净的参考信号。

#### DNSMOS — 整体质量评估（最推荐）

```python
# pip install speechmos
import speechmos
# 输出: SIG(语音质量 1-5), BAK(背景噪声 1-5), OVRL(整体 1-5)
```

或使用 `torchmetrics.audio.DeepNoiseSuppressionMeanOpinionScore`

#### NISQA — 细粒度质量分析

```python
# pip install nisqa 或 torchmetrics
# 输出: MOS_pred, noisiness, coloration, discontinuity, loudness
# discontinuity 可检测削波/失真
```

#### TorchAudio SQUIM — 无参考估算传统指标

```python
import torchaudio
objective_model = torchaudio.pipelines.SQUIM_OBJECTIVE.get_model()
stoi, pesq, si_sdr = objective_model(waveform)  # 无需参考信号
```

#### HNR（谐波噪声比）— 声音纯净度

```python
import parselmouth
snd = parselmouth.Sound("audio.wav")
harmonicity = snd.to_harmonicity()
hnr = harmonicity.values[harmonicity.values != -200].mean()
# HNR > 20dB 为优质语音
```

#### 削波/失真检测

```python
import numpy as np
clipping_ratio = np.sum(np.abs(samples) >= 0.99) / len(samples)
# clipping_ratio < 0.01 为合格
```

### 4.4 韵律丰富度与情感分析

#### F0（基频/音高）变化 — 最重要的韵律指标

推荐 `parselmouth`（Praat 的 Python 封装）：

```python
import parselmouth
import numpy as np

snd = parselmouth.Sound("audio.wav")
pitch = snd.to_pitch_ac()
f0_values = pitch.selected_array['frequency']
f0_values = f0_values[f0_values > 0]  # 去除无声段

f0_std = np.std(f0_values)           # F0 标准差
f0_range = np.max(f0_values) - np.min(f0_values)  # F0 范围
f0_cv = np.std(f0_values) / np.mean(f0_values)    # 变异系数（归一化，可跨说话人比较）
```

备选：`librosa.pyin(y, fmin, fmax)` — 纯 Python，易集成但精度略低。

#### 能量/强度变化

```python
# parselmouth
intensity = snd.to_intensity()
energy_cv = np.std(intensity.values[0]) / np.mean(intensity.values[0])

# 或 librosa
rms = librosa.feature.rms(y=y)[0]
energy_cv = np.std(rms) / np.mean(rms)
```

#### 情感维度分析

推荐 audeering 的 wav2vec2 模型（输出连续值，适合排序）：

```python
# pip install transformers
from transformers import Wav2Vec2ForSequenceClassification
# 模型: audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim
# 输出: arousal(激活度), valence(效价), dominance(支配度)
```

#### openSMILE eGeMAPS — 综合韵律特征向量

```python
import opensmile
smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.eGeMAPSv02,
    feature_level=opensmile.FeatureLevel.Functionals,
)
features = smile.process_file("audio.wav")  # 88 维特征
```

---

## 五、综合评分公式

### 5.1 质量门控（第一阶段，淘汰劣质片段）

```
quality_pass = (dnsmos_ovrl >= 3.5) AND (clipping_ratio < 0.01) AND (hnr >= 15)
```

### 5.2 综合评分（第二阶段，排序优选）

```
Final_Score = w1 * Q_norm + w2 * P_norm
```

**质量分 Q**：
```
Q = 0.5 * dnsmos_ovrl + 0.2 * nisqa_mos + 0.15 * squim_pesq_norm + 0.15 * snr_norm
```

**韵律丰富度分 P**：
```
P = 0.35 * f0_cv_norm + 0.25 * energy_cv_norm + 0.20 * speech_rate_var_norm + 0.20 * arousal_norm
```

所有子指标归一化到 [0, 1] 区间。

### 5.3 权重推荐

| 使用场景 | w1 (质量) | w2 (韵律) |
|----------|-----------|-----------|
| 通用/默认 | 0.6 | 0.4 |
| 播报/朗读 | 0.7 | 0.3 |
| 有声书/情感表达 | 0.4 | 0.6 |
| 嘈杂环境录制 | 0.8 | 0.2 |

---

## 六、候选片段生成策略

### 6.1 推荐：基于句子组合的分段

比滑动窗口更优，天然保证语义完整性：

1. WhisperX 转录 + 对齐 → 获得句级时间戳列表
2. 遍历连续句子的组合，找总时长在 **8-10 秒** 范围内的组合
3. 边界对齐到词级时间戳（不切断任何词）
4. 前后各留 150ms 自然静音

### 6.2 备选：重叠滑动窗口 + 边界修正

- 窗口 10 秒、步长 2-3 秒（70-80% 重叠）
- 候选数 ~116 个
- 生成后用词级时间戳修正边界到最近词间隙
- 用 NMS（IoU > 0.5 去重）去除重复候选

### 6.3 时长考量

- **下限 8 秒**：过短导致 speaker embedding 信息不足
- **上限 12 秒**：过长增加推理计算量
- **最优 10 秒**：250 个语音 token，足够捕捉说话人特征和韵律
- 宁可稍长也不要切断最后一个词，**完整的语句边界比精确时长更重要**

---

## 七、多段拼接策略分析

### 7.1 拼接 vs 单段连续的权衡

从 5 分钟音频中找到一段**连续的**高质量 10 秒并不容易（可能中间夹杂咳嗽、口误、噪声），多段拼接可以各取精华。但 CosyVoice3 的 speech tokenizer 以连续 25 tokens/秒编码，拼接引入的风险不可忽视：

| 风险 | 说明 |
|------|------|
| 韵律跳变 | 两段话的 F0 基线、语速、能量水平突变，模型会把"跳变"当作说话人风格 |
| 上下文断裂 | 句子间的自然衔接韵律丢失，模型学到碎片化韵律模式 |
| embedding 模糊 | 不同位置的微妙录音环境差异（混响、底噪）导致 speaker embedding 被平均模糊 |

### 7.2 拼接可行性分级

```
                推荐度
单段连续 10s     ★★★★★  首选，天然一致性最好
相邻句拼接       ★★★★   间隔 < 2s 的相邻句可拼，韵律连贯性高
跨段拼接         ★★     间隔 > 5s，需验证 F0/能量一致性
跨区域拼接       ★      不同段落风格差异大，不推荐
```

### 7.3 拼接约束条件

| 约束条件 | 阈值 | 原因 |
|---------|------|------|
| 两段间原始间距 | < 3 秒 | 间距越近韵律上下文越连贯 |
| F0 均值差异 | < 15% | 避免音高突变 |
| 能量均值差异 | < 20% | 避免音量跳变 |
| 最多拼接段数 | <= 3 段 | 过多拼接点增加失真风险 |
| 单段最短时长 | >= 3 秒 | 太短的碎片没有完整韵律信息 |
| 拼接处静音 | 150-300ms | 模拟自然停顿 |

### 7.4 拼接一致性罚分

路径 B（拼接）的候选需要额外扣分，只有扣完罚分后仍优于路径 A 最优单段时才选拼接方案：

```
拼接罚分 = α * F0_diff_norm + β * energy_diff_norm + γ * (num_joints - 1)

Final_Score_B = Final_Score - 拼接罚分
```

推荐初始参数：`α = 0.15, β = 0.10, γ = 0.05`，需根据实际合成效果调优。

### 7.5 三条路径的选择逻辑

```
路径 A（单段连续）  ──→ score_A = Final_Score
路径 B（相邻拼接）  ──→ score_B = Final_Score - 拼接罚分
路径 C（多 prompt） ──→ 不参与单 prompt 排序，独立输出

if score_A >= score_B:
    单 prompt 输出 = 路径 A 最优段
else:
    单 prompt 输出 = 路径 B 最优拼接段

多 prompt 输出 = 路径 C 的 Top 3-5 段（各自独立，用于 embedding 融合）
```

### 7.6 更推荐的方向：多 prompt 独立融合（路径 C）

与其拼接出一段"完美"的 10 秒音频，不如保持多段独立，通过 speaker embedding 层面融合：

- 选 3-5 段各 3-5 秒的高质量片段
- 分别提取 speaker embedding（CosyVoice3 使用 ERes2Net）
- 加权融合 embedding，权重可按质量分/韵律分分配
- 既保留每段的韵律自然性，又扩展了表现力范围

Mega-TTS 2 论文已验证多句 prompt 的有效性（详见第八章）。这是架构上更优雅的解法，也契合项目的"多 prompt 融合"核心策略。

---

## 八、Mega-TTS 2 多句 Prompt 机制深度分析

### 8.1 核心架构：音色-韵律显式解耦

Mega-TTS 2 将语音信号分解为三个独立潜在空间：**内容（Content）**、**音色（Timbre）**、**韵律（Prosody）**。

理论依据基于互信息假设：
```
I(y_t; ỹ) = H(z_t) + H(g)
```
目标语音和参考语音之间的互信息**仅包含**音色和全局风格，**不包含**细粒度韵律。这保证了从参考语音中只能提取到音色，韵律必须从目标语音自身获取。

VQ-GAN 声学自编码器包含三个编码器：

| 编码器 | 输入 | 输出 | 压缩方式 |
|--------|------|------|----------|
| 内容编码器 E_c | 音素序列 | 内容隐状态 z_c | 音素嵌入 |
| 音色编码器 E_t | 参考 mel 频谱 ỹ | 音色隐状态 z_t | 时间压缩 d=16 |
| 韵律编码器 E_p | 目标 mel 频谱 y_t | 韵律编码 z_p | 信息瓶颈: 时间压缩 r=8 + VQ(codebook=1024, dim=256) |

消融实验验证：移除 MRTE 后 WER 从 2.28% 升至 **5.57%**，说话人相似度从 0.905 降至 **0.841**。

### 8.2 多参考音色编码器（MRTE）— 注意力加权融合

MRTE 的多句信息聚合流程（**非简单平均**）：

```
多段参考语音 {ỹ₁, ỹ₂, ..., ỹₙ}
    │
    ▼
[时间轴拼接] → ỹ' = Concat(ỹ₁, ỹ₂, ..., ỹₙ)
    │
    ▼
[Mel 编码器] → 压缩因子 d=16 → 音色隐状态 z_t (Keys, Values)
    │
    ▼
[交叉注意力] ← 内容隐状态 z_c (Queries)
    │
    ▼
[长度调节器] → 上采样至目标 mel 长度
```

**关键机制**：通过缩放点积注意力（Scaled Dot-Product Attention），每个内容 token 可以从所有参考语音的音色特征中**选择性地提取**语义相关的音色信息：
- 不同音素/词可以从不同参考句子中获取最匹配的音色细节
- 长参考自然提供更丰富的音色样本空间
- 注意力权重自动学习哪些参考片段对当前合成内容最有用

| 融合方法 | 信息保留 | 粒度 | 适应性 |
|---------|---------|------|--------|
| 简单平均 | 全局统计量 | 粗粒度（一个向量） | 无 |
| **注意力融合（MRTE）** | **局部时变特征** | **细粒度（逐帧）** | **内容自适应** |

### 8.3 韵律潜在语言模型（P-LLM）— 序列拼接 + 自回归

P-LLM 规格：12 层 Decoder-only Transformer，1024 隐藏维度，16 注意力头，1.51 亿参数。

多句韵律处理方式：
1. 分别提取各段韵律编码：`{z_p1, z_p2, ..., z_pn}`
2. 时间维度拼接：`z_p' = Concat(z_p1, z_p2, ..., z_pn)`
3. 同步拼接内容隐状态：`z_c' = Concat(z_c1, z_c2, ..., z_cn)`
4. 自回归条件生成：`p(u'|z_c'; θ) = ∏ p(u_l'|u_{<l}', z_c'; θ)`

句间添加 **start/end token** 引导 P-LLM 处理边界。理论最大支持 **300 秒**提示。

### 8.4 韵律插值技术

在自回归生成的**每一步**，混合两个说话人的概率分布：

```
p(û) = ∏ [(1-γ) · p(û_t | 说话人B韵律) + γ · p(û_t | 说话人A韵律)]
```

- `γ = 0.8`：将说话人 A 的 80% 韵律风格迁移到目标说话人
- MRTE 仍从目标说话人提取音色 → 保持声音不变，只改韵律
- 比直接替换韵律编码更柔和、更细粒度

### 8.5 不同提示长度的性能数据

| 提示长度 | WER↓ | 说话人相似度↑ | DTW↓ | 质量MOS↑ | 相似度MOS↑ |
|---------|------|-------------|------|---------|-----------|
| 3 秒 | 2.46% | 0.898 | 34.39 | 3.99 | 3.75 |
| 10 秒 | 2.28% | 0.905 | 32.30 | 4.05 | 3.79 |
| 60 秒 | 2.24% | 0.926 | 30.55 | 4.11 | 3.95 |
| 300 秒 | 2.23% | 0.932 | 29.95 | 4.12 | 4.01 |
| 录音原声 | 1.83% | 1.000 | - | 4.18 | - |

**关键结论**：
- 3s→10s：相似度提升显著（0.898→0.905）
- 10s→60s：相似度大幅跳升（0.905→0.926），MOS +0.16
- 60s→300s：收益递减（0.926→0.932）
- **10 秒以上即可超越微调基线**，性能随提示长度单调递增

### 8.6 与 CosyVoice3 的适配性分析

**关键架构差异**：

| 差异点 | Mega-TTS 2 | CosyVoice3 |
|--------|-----------|------------|
| 韵律-音色解耦 | 显式解耦（VQ+MRTE） | 隐式（speech token 包含韵律+内容） |
| 说话人条件注入点 | 解码器全程 | 仅 CFM 阶段（ERes2Net 192维） |
| 提示处理 | 多段拼接 mel 频谱 | 单段 speech token 序列 |
| 韵律建模 | 显式 P-LLM 自回归 | LLM 隐式建模 |

**核心问题**：CosyVoice3 的 speech tokenizer 以 25 tokens/秒编码，**韵律信息已耦合在语音 token 中**，无法像 Mega-TTS 2 那样独立操控韵律。

**推荐适配方案（不修改模型架构）**：

```
策略: 音色融合 + 韵律选优

[多段候选音频]
    │
    ├──→ [全部] ERes2Net 提取 embedding → 加权平均 → 融合后说话人嵌入（注入 CFM）
    │         权重 = softmax(0.6 * 质量分 + 0.4 * 韵律丰富度)
    │
    └──→ [选最优1段] 作为 speech token 上下文（保持韵律一致性）
             选择标准: 综合评分最高的单段连续片段
```

```python
# 加权平均融合（最简单可行的起步方案）
embeddings = [eres2net.extract(audio_i) for audio_i in prompt_audios]  # 各192维
weights = softmax([0.6 * Q_i + 0.4 * P_i for i in range(len(prompt_audios))])
fused_embedding = sum(w * e for w, e in zip(weights, embeddings))
```

### 8.7 相关系统的多 prompt 策略对比

| 系统 | 融合方式 | 多句数量 | 饱和点 |
|------|---------|---------|--------|
| Mega-TTS 2 | 注意力融合(音色) + 序列拼接(韵律) | 5-10 段 | ~60 秒 |
| 多尺度声学提示 | 双尺度(风格+音色)注意力融合 | 10-20 段 | ~1 分钟（SECS 0.798） |
| ExpPro (CosyVoice) | 静态筛选 + 动态选择最优单段 | 选 1 段最优 | N/A |
| MRMI-TTS | 互信息最小化 + 句子相似度选择 | 多段参考 | 未公开 |
| StyleFusion TTS | 通用风格融合编码器(GSF-enc) | 文本+音频混合 | 未公开 |

### 8.8 本项目推荐的多段数量

- **音色嵌入融合**：3-5 段，每段 3-5 秒，覆盖不同音素环境
- **韵律参考**：选 1 段最优 8-10 秒连续段（韵律耦合在 token 中，多段拼接有风险）
- 参考 Mega-TTS 2 和多尺度提示论文，**约 1 分钟的多句提示已接近性能上限**

---

## 九、MegaTTS 3 及 2025 TTS 前沿趋势

### 9.1 MegaTTS 3 确认存在

- **论文**: "MegaTTS 3: Sparse Alignment Enhanced Latent Diffusion Transformer for Zero-Shot Speech Synthesis"
- **ArXiv**: [2502.18924](https://arxiv.org/abs/2502.18924)（2025.02，同一团队：浙大/字节跳动 Ziyue Jiang, Yi Ren 等）
- **GitHub**: [bytedance/MegaTTS3](https://github.com/bytedance/MegaTTS3)（已开源推理代码）
- **HuggingFace**: [ByteDance/MegaTTS3](https://huggingface.co/ByteDance/MegaTTS3)

### 9.2 MegaTTS 3 vs MegaTTS 2：彻底的架构重设计

**MegaTTS 2 的三大核心组件 — MRTE、P-LLM、VQ-GAN — 在 MegaTTS 3 中全部被替换。**

| 维度 | MegaTTS 2 (ICLR 2024) | MegaTTS 3 (2025) |
|------|----------------------|------------------|
| 骨干架构 | VQ-GAN + P-LLM | WaveVAE + Latent Diffusion Transformer (DiT) |
| 音色建模 | MRTE（多参考音色编码器，交叉注意力） | **In-context learning（参考语音潜在向量拼接）** |
| 韵律建模 | P-LLM（显式自回归韵律模型） | **DiT 隐式建模（扩散过程中自动捕获）** |
| 音色-韵律解耦 | 显式三路解耦 | **不做显式解耦，统一潜在空间** |
| 对齐策略 | 无特殊对齐 | **稀疏对齐（Sparse Alignment）— 核心创新** |
| 多参考支持 | **原生支持**（MRTE 拼接 + P-LLM 序列拼接） | **不原生支持**，仅单段 prompt |
| 参数量 | P-LLM 1.51亿 | DiT 0.45B |

### 9.3 MegaTTS 3 核心创新

#### WaveVAE 语音压缩

- 基于 WavTokenizer 的 VAE 编码器，32 维潜在通道
- 24kHz → 25 帧/秒 的**连续**潜在向量（非离散 token）
- 注意：字节跳动出于安全考虑**未公开 WaveVAE 编码器参数**

#### 稀疏对齐（Sparse Alignment）

解决了 TTS 对齐的"鲁棒性 vs 自然度"两难：

| 对齐策略 | SIM-O↑ | WER↓ | CMOS |
|---------|--------|------|------|
| **稀疏对齐（提出）** | **0.71** | 1.82% | **0.00** |
| 强制对齐 | 0.70 | 1.80% | -0.17 |
| 无对齐 | 0.67 | 2.14% | -0.12 |

每个音素仅保留一个锚点 token，其余用 mask 填充 → 在保证鲁棒性的同时给模型更大的韵律搜索空间。

#### 参考语音处理方式

- 训练时将语音潜在向量随机分为 prompt 区域和 masked target 区域（分割比 U(0.1, 0.9)）
- 推理时参考语音通过 WaveVAE 编码后拼接在目标前面（in-context）
- **多条件 CFG**：说话人引导尺度 α_spk=3.5，文本引导尺度 α_txt=2.5

#### 性能（LibriSpeech test-clean）

| 系统 | SIM-O↑ | WER↓ | SMOS↑ | RTF↓ |
|------|--------|------|-------|------|
| **MegaTTS 3** | **0.71** | **1.82%** | **3.98** | 0.188 |
| NaturalSpeech 3 | 0.67 | 1.81% | 3.95 | 0.296 |
| F5-TTS | 0.66 | 1.96% | 3.96 | 0.307 |
| MegaTTS 3 (8步加速) | 0.70 | 1.86% | - | **0.124** |

### 9.4 行业趋势：2025 主流 TTS 全部转向单段参考

| 系统 | 年份 | 多参考支持 | 参考方式 |
|------|------|----------|---------|
| MegaTTS 2 | 2024 | 原生支持 | MRTE + P-LLM |
| **MegaTTS 3** | **2025** | **不支持** | **In-context 单段** |
| **CosyVoice3** | **2025** | **不支持** | **In-context 单段** |
| **F5-TTS** | **2024** | **不支持** | **Speech infilling 单段** |
| **MaskGCT** | **2024** | **不支持** | **In-context 单段** |
| Seed-TTS | 2024 | 不支持 | In-context + 说话人嵌入 |

**结论**：2025 年主流系统一致转向**统一潜在空间 + 单段参考 + in-context learning**，放弃显式音色-韵律解耦和多参考融合架构。这意味着：

1. **"选出最优单段"成为最关键的策略** — 不是多段融合，而是精准地选出那一段最好的
2. **我们项目的路径 A（单段连续最优）优先级进一步提升**
3. **多 prompt ERes2Net embedding 融合**仍作为增强手段保留（路径 C），但定位为辅助

### 9.5 值得关注的相关工作

#### EmoPro/ExpPro (2024, arXiv:2409.18512) — 情感 prompt 选取

- 两阶段：静态筛选（质量+韵律聚类）+ 动态选择（语义匹配）
- **关键发现**：prompt 有效性是模型相关的 — 同一 prompt 在 CosyVoice 和 GPT-SoVITS 表现不同
- **启示**：评分体系可加入"CosyVoice3 实际合成效果反馈"维度

#### 音色-韵律解耦最新进展

| 方法 | 代表系统 | 解耦程度 | 趋势 |
|------|---------|---------|------|
| VQ-GAN + 显式编码器 | MegaTTS 2 | 高 | 被弃用 |
| FACodec (FVQ) | NaturalSpeech 3 | 高 | 学术研究 |
| DisCodec (三因子+GRL) | DisCo-Speech (2025) | 高 | 可控性场景 |
| **统一潜在空间（不解耦）** | **MegaTTS 3, CosyVoice3** | **无** | **工业主流** |

显式解耦在可控性场景（如情感迁移）仍有学术价值，但工业级零样本克隆已转向统一表示 + 大规模数据。

### 9.6 对本项目 Pipeline 的建议更新

1. **单段选优策略提升为核心策略** — 所有 2025 主流系统都是单段参考，选出最好的那一段至关重要
2. **推荐参考时长 5-10 秒** — MegaTTS 3 推荐 5 秒即可，与我们 8-10 秒目标吻合
3. **考虑加入"合成反馈"评分维度** — 参考 ExpPro，用候选 prompt 实际合成一小段，评估合成质量作为加分项

---

## 十、依赖库清单

```bash
# 核心流水线
pip install whisperx              # ASR + 强制对齐
pip install silero-vad            # VAD 语音活动检测
pip install pydub                 # 静音处理/填充
pip install librosa               # 音频加载/特征分析
pip install soundfile             # 音频 I/O

# 质量评估
pip install speechmos             # DNSMOS
pip install nisqa                 # NISQA
pip install torchaudio            # SQUIM

# 韵律分析
pip install praat-parselmouth     # F0/能量/HNR (Praat 引擎)
pip install opensmile             # eGeMAPS 综合特征

# 情感识别
pip install transformers          # wav2vec2 情感模型

# 中文分词
pip install jieba                 # 中文分词辅助

# 备选（精度优先时）
conda install -c conda-forge montreal-forced-aligner  # MFA
```

---

## 十一、参考文献与来源

- [CosyVoice3 论文 (arXiv:2505.17589)](https://arxiv.org/abs/2505.17589)
- [CosyVoice GitHub](https://github.com/FunAudioLLM/CosyVoice)
- [阿里云 CosyVoice 声音复刻 API](https://help.aliyun.com/zh/model-studio/cosyvoice-clone-design-api)
- [ExpPro: Expressive Prompting (arXiv:2409.18512)](https://arxiv.org/abs/2409.18512)
- [Mega-TTS 2 论文](https://arxiv.org/abs/2307.07218)
- [Emilia 数据集论文 (arXiv:2407.05361)](https://arxiv.org/abs/2407.05361)
- [WhisperX GitHub](https://github.com/m-bain/whisperX)
- [Silero VAD GitHub](https://github.com/snakers4/silero-vad)
- [Montreal Forced Aligner](https://montreal-forced-aligner.readthedocs.io/)
- [DNSMOS 论文](https://arxiv.org/abs/2010.15258)
- [NISQA GitHub](https://github.com/gabrielmittag/NISQA)
- [TorchAudio SQUIM](https://docs.pytorch.org/audio/stable/tutorials/squim_tutorial.html)
- [audeering wav2vec2 emotion](https://huggingface.co/audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim)
- [Voice Cloning 综述 (arXiv:2505.00579)](https://arxiv.org/abs/2505.00579)
- [Mega-TTS 2 Demo](https://mega-tts.github.io/mega2_demo/)
- [Mega-TTS 2 OpenReview](https://openreview.net/forum?id=mvMI3N4AvD)
- [多尺度声学提示论文 (arXiv:2309.11977)](https://arxiv.org/abs/2309.11977)
- [Boosting LLM for Speech Synthesis (arXiv:2401.00246)](https://arxiv.org/abs/2401.00246)
- [ERes2Net 论文](https://arxiv.org/abs/2305.12838)
- [StyleFusion TTS (arXiv:2409.15741)](https://arxiv.org/abs/2409.15741)
- [MegaTTS 3 论文 (arXiv:2502.18924)](https://arxiv.org/abs/2502.18924)
- [MegaTTS 3 GitHub](https://github.com/bytedance/MegaTTS3)
- [MegaTTS 3 HuggingFace](https://huggingface.co/ByteDance/MegaTTS3)
- [MaskGCT (arXiv:2409.00750)](https://arxiv.org/abs/2409.00750)
- [F5-TTS (arXiv:2410.06885)](https://arxiv.org/abs/2410.06885)
- [Seed-TTS (arXiv:2406.02430)](https://arxiv.org/abs/2406.02430)
- [DisCo-Speech / DisCodec (arXiv:2512.13251)](https://arxiv.org/abs/2512.13251)
- [NaturalSpeech 3 / FACodec (arXiv:2403.03100)](https://arxiv.org/abs/2403.03100)
- [IndexTTS2 (arXiv:2506.21619)](https://arxiv.org/abs/2506.21619)
- [Spark-TTS / BiCodec (arXiv:2503.01710)](https://arxiv.org/abs/2503.01710)
