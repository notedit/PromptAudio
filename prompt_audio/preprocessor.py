import numpy as np
import librosa
import soundfile as sf


class AudioPreprocessor:
    """将任意格式音频标准化为 CosyVoice3 所需的 16kHz 单声道格式。"""

    def __init__(self, target_sr: int = 16000):
        self.target_sr = target_sr

    def process(self, input_path: str, output_path: str) -> np.ndarray:
        """加载、重采样、单声道转换、CosyVoice3 音量标准化。

        Returns:
            标准化后的 numpy 数组
        """
        wav, _ = librosa.load(input_path, sr=self.target_sr, mono=True)

        # CosyVoice3 音量标准化: raw_wav / max(raw_wav) * 0.6
        max_val = np.max(np.abs(wav))
        if max_val > 0:
            wav = wav / max_val * 0.6

        sf.write(output_path, wav, self.target_sr, subtype="PCM_16")
        return wav
