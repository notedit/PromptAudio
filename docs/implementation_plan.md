# CosyVoice3 Prompt Audio 选取方案 — 技术实现

## 一、方案总览

### 1.1 目标

输入一段 5 分钟以内的音频，自动选出最优的 10 秒以内片段作为 CosyVoice3 的 prompt audio。

### 1.2 核心约束

| 约束 | 规格 |
|------|------|
| 不切断字/词 | WhisperX 词级强制对齐保证 |
| 首尾静音 | 各 150ms，优先自然静音 |
| 输出时长 | 8-10 秒（CosyVoice3 推荐 3-10 秒） |
| 有效语音占比 | >= 60% |
| 输出格式 | 16kHz / 单声道 / 16bit WAV |

### 1.3 Pipeline 架构

```
输入: 5 分钟音频 (.wav/.mp3/.m4a)
  │
  ├─ Step 1: 预处理 (preprocessor.py)
  ├─ Step 2: 转录 + 对齐 (transcriber.py)
  ├─ Step 3: 候选片段生成 (candidate_generator.py)
  ├─ Step 4: 质量门控 (quality_gate.py)
  ├─ Step 5: 综合评分排序 (scorer.py)
  │
  └─ 输出: 最优 prompt audio + 元数据 JSON
```

---

## 二、项目结构

```
prompt_audio/
├── __init__.py
├── cli.py                    # CLI 入口
├── pipeline.py               # 主流水线编排
├── preprocessor.py           # Step 1: 音频预处理
├── transcriber.py            # Step 2: ASR + 强制对齐
├── candidate_generator.py    # Step 3: 候选片段生成
├── quality_gate.py           # Step 4: 质量门控
├── scorer.py                 # Step 5: 综合评分
├── silence_handler.py        # 静音检测与填充
├── config.py                 # 全局配置/阈值
└── models/
    └── __init__.py            # 模型懒加载管理
```

---

## 三、各模块技术实现

### Step 1: 预处理 — `preprocessor.py`

职责：将任意格式音频标准化为 CosyVoice3 所需格式。

```python
import librosa
import soundfile as sf
import numpy as np

class AudioPreprocessor:
    def __init__(self, target_sr=16000):
        self.target_sr = target_sr

    def process(self, input_path: str, output_path: str) -> np.ndarray:
        """加载、重采样、单声道转换、音量标准化"""
        # 1. 加载并重采样
        wav, sr = librosa.load(input_path, sr=self.target_sr, mono=True)

        # 2. CosyVoice3 音量标准化
        wav = wav / np.max(np.abs(wav)) * 0.6

        # 3. 保存
        sf.write(output_path, wav, self.target_sr, subtype='PCM_16')
        return wav
```

**依赖**: `librosa`, `soundfile`

### Step 2: 转录 + 强制对齐 — `transcriber.py`

职责：用 WhisperX 获取词级时间戳，按标点重组为句级时间戳。

```python
import whisperx

class Transcriber:
    def __init__(self, model_size="large-v3", device="cuda", language="zh"):
        self.device = device
        self.language = language
        self.model = whisperx.load_model(model_size, device=device)
        self.align_model, self.align_metadata = whisperx.load_align_model(
            language_code=language, device=device
        )

    def transcribe_and_align(self, audio_path: str) -> dict:
        """返回包含词级和句级时间戳的结构化结果"""
        audio = whisperx.load_audio(audio_path)

        # ASR 转录
        result = self.model.transcribe(audio, language=self.language)

        # 强制对齐 → 词级时间戳
        result = whisperx.align(
            result["segments"],
            self.align_model,
            self.align_metadata,
            audio,
            device=self.device,
            return_char_alignments=True,  # 中文开启字符级
        )

        return self._build_sentences(result["segments"])

    def _build_sentences(self, segments: list) -> list:
        """将 WhisperX segments 整理为句级结构
        返回: [{text, start, end, words: [{word, start, end}]}]
        """
        sentences = []
        for seg in segments:
            words = seg.get("words", [])
            if not words:
                continue
            # 过滤掉没有时间戳的词
            valid_words = [w for w in words if "start" in w and "end" in w]
            if not valid_words:
                continue
            sentences.append({
                "text": seg["text"],
                "start": valid_words[0]["start"],
                "end": valid_words[-1]["end"],
                "words": valid_words,
            })
        return sentences
```

**依赖**: `whisperx`（内含 faster-whisper + wav2vec2 对齐模型）

### Step 3: 候选片段生成 — `candidate_generator.py`

职责：基于句级时间戳，生成满足时长约束的候选片段列表。

核心策略是**路径 A（单段连续）**，因为 2025 年所有主流 TTS（MegaTTS 3、CosyVoice3、F5-TTS）都采用单段参考。

```python
import numpy as np
from dataclasses import dataclass

@dataclass
class Candidate:
    start: float          # 起始时间（秒）
    end: float            # 结束时间（秒）
    text: str             # 转录文本
    sentence_indices: list # 包含的句子索引
    duration: float       # 时长

    @property
    def speech_ratio(self) -> float:
        """有效语音占比（由外部 VAD 填充）"""
        return getattr(self, '_speech_ratio', 1.0)

class CandidateGenerator:
    def __init__(
        self,
        min_duration: float = 5.0,
        max_duration: float = 10.0,
        silence_pad: float = 0.15,  # 150ms
        min_speech_ratio: float = 0.6,
    ):
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.silence_pad = silence_pad
        self.min_speech_ratio = min_speech_ratio

    def generate(self, sentences: list, total_duration: float) -> list[Candidate]:
        """遍历连续句子组合，生成满足时长约束的候选片段"""
        candidates = []
        n = len(sentences)

        for i in range(n):
            text_parts = []
            for j in range(i, n):
                text_parts.append(sentences[j]["text"])

                # 片段起止（含 150ms 静音 padding）
                start = max(0, sentences[i]["start"] - self.silence_pad)
                end = min(total_duration, sentences[j]["end"] + self.silence_pad)
                duration = end - start

                if duration < self.min_duration:
                    continue
                if duration > self.max_duration:
                    break  # 后续只会更长，剪枝

                candidates.append(Candidate(
                    start=start,
                    end=end,
                    text="".join(text_parts),
                    sentence_indices=list(range(i, j + 1)),
                    duration=duration,
                ))

        return candidates

    def extract_audio(self, wav: np.ndarray, candidate: Candidate, sr: int = 16000) -> np.ndarray:
        """从完整音频中截取候选片段"""
        start_sample = int(candidate.start * sr)
        end_sample = int(candidate.end * sr)
        return wav[start_sample:end_sample]
```

### Step 4: 质量门控 — `quality_gate.py`

职责：淘汰质量不达标的候选片段。

```python
import numpy as np
import parselmouth

class QualityGate:
    def __init__(
        self,
        dnsmos_threshold: float = 3.5,
        clipping_threshold: float = 0.01,
        hnr_threshold: float = 15.0,
        min_speech_ratio: float = 0.6,
    ):
        self.dnsmos_threshold = dnsmos_threshold
        self.clipping_threshold = clipping_threshold
        self.hnr_threshold = hnr_threshold
        self.min_speech_ratio = min_speech_ratio
        self._dnsmos_model = None

    @property
    def dnsmos_model(self):
        if self._dnsmos_model is None:
            from speechmos import dnsmos
            self._dnsmos_model = dnsmos
        return self._dnsmos_model

    def check(self, wav: np.ndarray, sr: int = 16000) -> dict:
        """返回 {passed: bool, metrics: dict, reasons: list}"""
        metrics = {}
        reasons = []

        # 1. DNSMOS
        dnsmos_scores = self.dnsmos_model.run(wav, sr)
        metrics["dnsmos_ovrl"] = dnsmos_scores["ovrl"]
        metrics["dnsmos_sig"] = dnsmos_scores["sig"]
        metrics["dnsmos_bak"] = dnsmos_scores["bak"]
        if metrics["dnsmos_ovrl"] < self.dnsmos_threshold:
            reasons.append(f"DNSMOS OVRL {metrics['dnsmos_ovrl']:.2f} < {self.dnsmos_threshold}")

        # 2. 削波检测
        clipping_ratio = np.sum(np.abs(wav) >= 0.99) / len(wav)
        metrics["clipping_ratio"] = clipping_ratio
        if clipping_ratio >= self.clipping_threshold:
            reasons.append(f"Clipping ratio {clipping_ratio:.4f} >= {self.clipping_threshold}")

        # 3. HNR
        snd = parselmouth.Sound(wav, sampling_frequency=sr)
        harmonicity = snd.to_harmonicity()
        hnr_values = harmonicity.values[harmonicity.values != -200]
        hnr = np.mean(hnr_values) if len(hnr_values) > 0 else 0
        metrics["hnr"] = hnr
        if hnr < self.hnr_threshold:
            reasons.append(f"HNR {hnr:.1f} dB < {self.hnr_threshold}")

        # 4. 有效语音占比（通过 Silero VAD）
        speech_ratio = self._compute_speech_ratio(wav, sr)
        metrics["speech_ratio"] = speech_ratio
        if speech_ratio < self.min_speech_ratio:
            reasons.append(f"Speech ratio {speech_ratio:.2f} < {self.min_speech_ratio}")

        return {
            "passed": len(reasons) == 0,
            "metrics": metrics,
            "reasons": reasons,
        }

    def _compute_speech_ratio(self, wav: np.ndarray, sr: int) -> float:
        """用 Silero VAD 计算有效语音占比"""
        import torch
        model, utils = torch.hub.load('snakers4/silero-vad', 'silero_vad')
        get_speech_timestamps = utils[0]

        wav_tensor = torch.FloatTensor(wav)
        timestamps = get_speech_timestamps(wav_tensor, model, sampling_rate=sr)

        speech_samples = sum(ts["end"] - ts["start"] for ts in timestamps)
        return speech_samples / len(wav)
```

**依赖**: `speechmos`, `parselmouth`, `silero-vad`

### Step 5: 综合评分 — `scorer.py`

职责：对通过门控的候选片段计算综合得分并排序。

```python
import numpy as np
import parselmouth
import librosa

class PromptScorer:
    """
    Final_Score = w1 * Q_norm + w2 * P_norm

    Q = 0.5 * dnsmos_ovrl + 0.2 * nisqa_mos + 0.15 * squim_pesq + 0.15 * snr
    P = 0.35 * f0_cv + 0.25 * energy_cv + 0.20 * speech_rate_var + 0.20 * arousal
    """

    def __init__(self, w_quality: float = 0.6, w_prosody: float = 0.4):
        self.w_quality = w_quality
        self.w_prosody = w_prosody
        self._emotion_model = None

    def score(self, wav: np.ndarray, sr: int, gate_metrics: dict) -> dict:
        """计算综合评分，gate_metrics 来自 QualityGate.check()"""
        q_metrics = self._compute_quality(wav, sr, gate_metrics)
        p_metrics = self._compute_prosody(wav, sr)

        q_score = (
            0.50 * q_metrics["dnsmos_ovrl_norm"]
            + 0.20 * q_metrics["nisqa_mos_norm"]
            + 0.15 * q_metrics["squim_pesq_norm"]
            + 0.15 * q_metrics["snr_norm"]
        )
        p_score = (
            0.35 * p_metrics["f0_cv_norm"]
            + 0.25 * p_metrics["energy_cv_norm"]
            + 0.20 * p_metrics["speech_rate_var_norm"]
            + 0.20 * p_metrics["arousal_norm"]
        )

        final_score = self.w_quality * q_score + self.w_prosody * p_score

        return {
            "final_score": final_score,
            "quality_score": q_score,
            "prosody_score": p_score,
            "quality_metrics": q_metrics,
            "prosody_metrics": p_metrics,
        }

    # ── 质量分 Q ──

    def _compute_quality(self, wav: np.ndarray, sr: int, gate_metrics: dict) -> dict:
        metrics = {}

        # DNSMOS (已在门控阶段计算，复用)
        metrics["dnsmos_ovrl_norm"] = self._normalize(gate_metrics["dnsmos_ovrl"], 1.0, 5.0)

        # NISQA
        nisqa_mos = self._get_nisqa_score(wav, sr)
        metrics["nisqa_mos_norm"] = self._normalize(nisqa_mos, 1.0, 5.0)

        # SQUIM (torchaudio)
        squim_pesq = self._get_squim_pesq(wav, sr)
        metrics["squim_pesq_norm"] = self._normalize(squim_pesq, 1.0, 4.5)

        # SNR 估算
        snr = self._estimate_snr(wav)
        metrics["snr_norm"] = self._normalize(snr, 0, 40)

        return metrics

    def _get_nisqa_score(self, wav: np.ndarray, sr: int) -> float:
        """NISQA 无参考 MOS 预测"""
        try:
            from torchmetrics.audio import NonIntrusiveSpeechQualityAssessment
            import torch
            nisqa = NonIntrusiveSpeechQualityAssessment()
            tensor = torch.FloatTensor(wav).unsqueeze(0)
            return nisqa(tensor).item()
        except Exception:
            return 3.0  # fallback

    def _get_squim_pesq(self, wav: np.ndarray, sr: int) -> float:
        """TorchAudio SQUIM 无参考 PESQ 估算"""
        import torch
        import torchaudio
        model = torchaudio.pipelines.SQUIM_OBJECTIVE.get_model()
        tensor = torch.FloatTensor(wav).unsqueeze(0)
        if sr != 16000:
            tensor = torchaudio.functional.resample(tensor, sr, 16000)
        stoi, pesq, si_sdr = model(tensor)
        return pesq.item()

    def _estimate_snr(self, wav: np.ndarray) -> float:
        """简单 SNR 估算：用最安静的 10% 帧作为噪声基底"""
        frame_len = 400  # 25ms at 16kHz
        frames = [wav[i:i+frame_len] for i in range(0, len(wav)-frame_len, frame_len)]
        energies = [np.mean(f**2) for f in frames]
        energies.sort()
        noise_energy = np.mean(energies[:max(1, len(energies)//10)])
        signal_energy = np.mean(energies)
        if noise_energy < 1e-10:
            return 40.0
        return 10 * np.log10(signal_energy / noise_energy)

    # ── 韵律丰富度分 P ──

    def _compute_prosody(self, wav: np.ndarray, sr: int) -> dict:
        metrics = {}
        snd = parselmouth.Sound(wav, sampling_frequency=sr)

        # F0 变异系数
        pitch = snd.to_pitch_ac()
        f0 = pitch.selected_array["frequency"]
        f0 = f0[f0 > 0]
        f0_cv = np.std(f0) / np.mean(f0) if len(f0) > 0 and np.mean(f0) > 0 else 0
        metrics["f0_cv_norm"] = self._normalize(f0_cv, 0, 0.5)

        # 能量变异系数
        intensity = snd.to_intensity()
        int_values = intensity.values[0]
        energy_cv = np.std(int_values) / np.mean(int_values) if np.mean(int_values) > 0 else 0
        metrics["energy_cv_norm"] = self._normalize(energy_cv, 0, 0.5)

        # 语速变化（RMS 能量的帧间差分标准差作为代理）
        rms = librosa.feature.rms(y=wav, frame_length=2048, hop_length=512)[0]
        rms_diff = np.diff(rms)
        speech_rate_var = np.std(rms_diff)
        metrics["speech_rate_var_norm"] = self._normalize(speech_rate_var, 0, 0.05)

        # 情感激活度（arousal）
        arousal = self._get_arousal(wav, sr)
        metrics["arousal_norm"] = self._normalize(arousal, 0, 1.0)

        return metrics

    def _get_arousal(self, wav: np.ndarray, sr: int) -> float:
        """wav2vec2 情感维度模型提取 arousal"""
        try:
            import torch
            from transformers import Wav2Vec2Processor, Wav2Vec2ForSequenceClassification

            if self._emotion_model is None:
                model_name = "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"
                self._emotion_processor = Wav2Vec2Processor.from_pretrained(model_name)
                self._emotion_model = Wav2Vec2ForSequenceClassification.from_pretrained(model_name)

            inputs = self._emotion_processor(wav, sampling_rate=sr, return_tensors="pt")
            with torch.no_grad():
                outputs = self._emotion_model(**inputs)
            # outputs.logits: [arousal, dominance, valence]
            arousal = outputs.logits[0, 0].item()
            return max(0, min(1, arousal))
        except Exception:
            return 0.5  # fallback

    @staticmethod
    def _normalize(value: float, min_val: float, max_val: float) -> float:
        """归一化到 [0, 1]"""
        return max(0, min(1, (value - min_val) / (max_val - min_val)))
```

**依赖**: `parselmouth`, `librosa`, `torchaudio`, `transformers`, `speechmos`, `torchmetrics`

### 静音处理 — `silence_handler.py`

```python
import numpy as np
import torch

class SilenceHandler:
    """确保片段首尾有 ~150ms 静音"""

    def __init__(self, target_silence_ms: int = 150, sr: int = 16000):
        self.target_samples = int(target_silence_ms / 1000 * sr)
        self.sr = sr

    def ensure_padding(self, wav: np.ndarray) -> np.ndarray:
        """检查首尾静音，不足时补充"""
        head_silence = self._measure_leading_silence(wav)
        tail_silence = self._measure_trailing_silence(wav)

        head_pad = max(0, self.target_samples - head_silence)
        tail_pad = max(0, self.target_samples - tail_silence)

        if head_pad > 0 or tail_pad > 0:
            silence_head = np.zeros(head_pad, dtype=wav.dtype)
            silence_tail = np.zeros(tail_pad, dtype=wav.dtype)
            wav = np.concatenate([silence_head, wav, silence_tail])

        return wav

    def _measure_leading_silence(self, wav: np.ndarray, threshold: float = 0.01) -> int:
        """测量开头静音采样数"""
        for i, sample in enumerate(wav):
            if abs(sample) > threshold:
                return i
        return len(wav)

    def _measure_trailing_silence(self, wav: np.ndarray, threshold: float = 0.01) -> int:
        """测量结尾静音采样数"""
        for i in range(len(wav) - 1, -1, -1):
            if abs(wav[i]) > threshold:
                return len(wav) - 1 - i
        return len(wav)
```

### 全局配置 — `config.py`

```python
from dataclasses import dataclass

@dataclass
class PipelineConfig:
    # 音频格式
    target_sr: int = 16000

    # 候选生成
    min_duration: float = 5.0
    max_duration: float = 10.0
    silence_pad_ms: int = 150

    # 质量门控阈值
    dnsmos_threshold: float = 3.5
    clipping_threshold: float = 0.01
    hnr_threshold: float = 15.0
    min_speech_ratio: float = 0.6

    # 评分权重
    w_quality: float = 0.6
    w_prosody: float = 0.4

    # WhisperX
    whisper_model: str = "large-v3"
    language: str = "zh"
    device: str = "cuda"

    # 输出
    top_k: int = 5

# 预设场景配置
PRESETS = {
    "default":   PipelineConfig(),
    "broadcast": PipelineConfig(w_quality=0.7, w_prosody=0.3),
    "audiobook": PipelineConfig(w_quality=0.4, w_prosody=0.6),
    "noisy":     PipelineConfig(w_quality=0.8, w_prosody=0.2),
}
```

### 主流水线编排 — `pipeline.py`

```python
import numpy as np
import soundfile as sf
import json
from pathlib import Path

from .config import PipelineConfig
from .preprocessor import AudioPreprocessor
from .transcriber import Transcriber
from .candidate_generator import CandidateGenerator
from .quality_gate import QualityGate
from .scorer import PromptScorer
from .silence_handler import SilenceHandler

class PromptAudioPipeline:
    def __init__(self, config: PipelineConfig = None):
        self.config = config or PipelineConfig()
        self.preprocessor = AudioPreprocessor(self.config.target_sr)
        self.transcriber = Transcriber(
            model_size=self.config.whisper_model,
            device=self.config.device,
            language=self.config.language,
        )
        self.candidate_gen = CandidateGenerator(
            min_duration=self.config.min_duration,
            max_duration=self.config.max_duration,
            silence_pad=self.config.silence_pad_ms / 1000,
            min_speech_ratio=self.config.min_speech_ratio,
        )
        self.quality_gate = QualityGate(
            dnsmos_threshold=self.config.dnsmos_threshold,
            clipping_threshold=self.config.clipping_threshold,
            hnr_threshold=self.config.hnr_threshold,
            min_speech_ratio=self.config.min_speech_ratio,
        )
        self.scorer = PromptScorer(
            w_quality=self.config.w_quality,
            w_prosody=self.config.w_prosody,
        )
        self.silence_handler = SilenceHandler(
            target_silence_ms=self.config.silence_pad_ms,
            sr=self.config.target_sr,
        )

    def run(self, input_path: str, output_dir: str) -> dict:
        """
        执行完整 pipeline。
        返回 {best: {...}, candidates: [...]}
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        sr = self.config.target_sr

        # Step 1: 预处理
        preprocessed_path = str(output_dir / "preprocessed.wav")
        wav = self.preprocessor.process(input_path, preprocessed_path)
        total_duration = len(wav) / sr

        # Step 2: 转录 + 对齐
        sentences = self.transcriber.transcribe_and_align(preprocessed_path)
        if not sentences:
            return {"best": None, "candidates": [], "error": "No speech detected"}

        # Step 3: 候选片段生成
        candidates = self.candidate_gen.generate(sentences, total_duration)
        if not candidates:
            return {"best": None, "candidates": [], "error": "No valid candidates"}

        # Step 4 + 5: 质量门控 + 评分
        scored_candidates = []
        for candidate in candidates:
            segment_wav = self.candidate_gen.extract_audio(wav, candidate, sr)

            # 质量门控
            gate_result = self.quality_gate.check(segment_wav, sr)
            if not gate_result["passed"]:
                continue

            # 综合评分
            score_result = self.scorer.score(segment_wav, sr, gate_result["metrics"])

            scored_candidates.append({
                "start": candidate.start,
                "end": candidate.end,
                "duration": candidate.duration,
                "text": candidate.text,
                "final_score": score_result["final_score"],
                "quality_score": score_result["quality_score"],
                "prosody_score": score_result["prosody_score"],
                "gate_metrics": gate_result["metrics"],
            })

        if not scored_candidates:
            return {"best": None, "candidates": [], "error": "No candidates passed quality gate"}

        # 排序取 Top-K
        scored_candidates.sort(key=lambda x: x["final_score"], reverse=True)
        top_k = scored_candidates[:self.config.top_k]

        # 导出最优片段
        best = top_k[0]
        best_wav = self.candidate_gen.extract_audio(
            wav,
            type("C", (), {"start": best["start"], "end": best["end"]})(),
            sr,
        )
        best_wav = self.silence_handler.ensure_padding(best_wav)

        best_path = str(output_dir / "best_prompt.wav")
        sf.write(best_path, best_wav, sr, subtype="PCM_16")
        best["output_path"] = best_path

        # 保存元数据
        result = {"best": best, "candidates": top_k}
        with open(output_dir / "result.json", "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        return result
```

### CLI 入口 — `cli.py`

```python
import argparse
from .pipeline import PromptAudioPipeline
from .config import PipelineConfig, PRESETS

def main():
    parser = argparse.ArgumentParser(description="CosyVoice3 Prompt Audio Selector")
    parser.add_argument("input", help="Input audio file path")
    parser.add_argument("-o", "--output", default="./output", help="Output directory")
    parser.add_argument("-p", "--preset", choices=PRESETS.keys(), default="default")
    parser.add_argument("-k", "--top-k", type=int, default=5)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--language", default="zh")
    args = parser.parse_args()

    config = PRESETS[args.preset]
    config.top_k = args.top_k
    config.device = args.device
    config.language = args.language

    pipeline = PromptAudioPipeline(config)
    result = pipeline.run(args.input, args.output)

    if result["best"]:
        print(f"\nBest prompt: {result['best']['output_path']}")
        print(f"  Duration: {result['best']['duration']:.2f}s")
        print(f"  Score:    {result['best']['final_score']:.4f}")
        print(f"  Text:     {result['best']['text']}")
        print(f"\nTop {len(result['candidates'])} candidates saved to {args.output}/result.json")
    else:
        print(f"Error: {result.get('error', 'Unknown')}")

if __name__ == "__main__":
    main()
```

---

## 四、依赖清单

```
# requirements.txt
whisperx>=3.1
librosa>=0.10
soundfile>=0.12
praat-parselmouth>=0.4
speechmos>=0.1
torchaudio>=2.0
torchmetrics[audio]>=1.0
transformers>=4.30
torch>=2.0
pydub>=0.25
```

---

## 五、使用示例

```bash
# 安装
pip install -r requirements.txt

# 基础使用
python -m prompt_audio input.wav -o ./output

# 有声书场景（更重视韵律）
python -m prompt_audio input.wav -o ./output -p audiobook

# 嘈杂音频（更重视质量）
python -m prompt_audio input.wav -o ./output -p noisy

# 输出 Top 10
python -m prompt_audio input.wav -o ./output -k 10
```

输出结构：
```
output/
├── preprocessed.wav     # 标准化后的完整音频
├── best_prompt.wav      # 最优 prompt（含 150ms 首尾静音）
└── result.json          # 元数据（Top-K 候选列表、评分详情）
```

---

## 六、评分公式速查

```
质量门控:
  DNSMOS OVRL >= 3.5  AND  削波率 < 1%  AND  HNR >= 15dB  AND  语音占比 >= 60%

综合评分:
  Final = 0.6 * Q + 0.4 * P

  Q = 0.50 * DNSMOS + 0.20 * NISQA + 0.15 * SQUIM_PESQ + 0.15 * SNR
  P = 0.35 * F0变异系数 + 0.25 * 能量变异 + 0.20 * 语速变化 + 0.20 * 激活度

归一化区间:
  DNSMOS: [1, 5]    NISQA: [1, 5]    SQUIM_PESQ: [1, 4.5]   SNR: [0, 40]dB
  F0_CV: [0, 0.5]   Energy_CV: [0, 0.5]  Speech_rate_var: [0, 0.05]  Arousal: [0, 1]
```
