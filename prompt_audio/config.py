from dataclasses import dataclass, field


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


PRESETS = {
    "default": PipelineConfig(),
    "broadcast": PipelineConfig(w_quality=0.7, w_prosody=0.3),
    "audiobook": PipelineConfig(w_quality=0.4, w_prosody=0.6),
    "noisy": PipelineConfig(w_quality=0.8, w_prosody=0.2),
}
