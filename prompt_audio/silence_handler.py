import numpy as np


class SilenceHandler:
    """确保音频片段首尾各有 ~150ms 静音。"""

    def __init__(self, target_silence_ms: int = 150, sr: int = 16000):
        self.target_samples = int(target_silence_ms / 1000 * sr)
        self.sr = sr

    def ensure_padding(self, wav: np.ndarray) -> np.ndarray:
        """检查首尾静音是否充足，不足时补零。"""
        head_silence = self._measure_leading_silence(wav)
        tail_silence = self._measure_trailing_silence(wav)

        head_pad = max(0, self.target_samples - head_silence)
        tail_pad = max(0, self.target_samples - tail_silence)

        if head_pad > 0 or tail_pad > 0:
            parts = []
            if head_pad > 0:
                parts.append(np.zeros(head_pad, dtype=wav.dtype))
            parts.append(wav)
            if tail_pad > 0:
                parts.append(np.zeros(tail_pad, dtype=wav.dtype))
            wav = np.concatenate(parts)

        return wav

    @staticmethod
    def _measure_leading_silence(
        wav: np.ndarray, threshold: float = 0.01
    ) -> int:
        """返回开头连续低于阈值的采样数。"""
        for i, sample in enumerate(wav):
            if abs(sample) > threshold:
                return i
        return len(wav)

    @staticmethod
    def _measure_trailing_silence(
        wav: np.ndarray, threshold: float = 0.01
    ) -> int:
        """返回结尾连续低于阈值的采样数。"""
        for i in range(len(wav) - 1, -1, -1):
            if abs(wav[i]) > threshold:
                return len(wav) - 1 - i
        return len(wav)
