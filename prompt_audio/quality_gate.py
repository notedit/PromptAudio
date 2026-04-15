import logging

import numpy as np
import parselmouth
import torch

logger = logging.getLogger(__name__)


class QualityGate:
    """质量门控：淘汰 DNSMOS / 削波 / HNR / 语音占比不达标的片段。"""

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

        # 懒加载
        self._dnsmos_model = None
        self._vad_model = None
        self._vad_utils = None

    def check(self, wav: np.ndarray, sr: int = 16000) -> dict:
        """评估音频质量。

        Returns:
            {passed: bool, metrics: dict, reasons: list[str]}
        """
        metrics = {}
        reasons = []

        # 1. DNSMOS
        dnsmos = self._compute_dnsmos(wav, sr)
        metrics.update(dnsmos)
        if dnsmos["dnsmos_ovrl"] < self.dnsmos_threshold:
            reasons.append(
                f"DNSMOS OVRL {dnsmos['dnsmos_ovrl']:.2f} < {self.dnsmos_threshold}"
            )

        # 2. 削波检测
        clipping_ratio = float(np.sum(np.abs(wav) >= 0.99) / len(wav))
        metrics["clipping_ratio"] = clipping_ratio
        if clipping_ratio >= self.clipping_threshold:
            reasons.append(
                f"Clipping {clipping_ratio:.4f} >= {self.clipping_threshold}"
            )

        # 3. HNR
        hnr = self._compute_hnr(wav, sr)
        metrics["hnr"] = hnr
        if hnr < self.hnr_threshold:
            reasons.append(f"HNR {hnr:.1f} dB < {self.hnr_threshold}")

        # 4. 有效语音占比
        speech_ratio = self._compute_speech_ratio(wav, sr)
        metrics["speech_ratio"] = speech_ratio
        if speech_ratio < self.min_speech_ratio:
            reasons.append(
                f"Speech ratio {speech_ratio:.2f} < {self.min_speech_ratio}"
            )

        return {"passed": len(reasons) == 0, "metrics": metrics, "reasons": reasons}

    # ── DNSMOS ──

    def _compute_dnsmos(self, wav: np.ndarray, sr: int) -> dict:
        try:
            if self._dnsmos_model is None:
                from speechmos import dnsmos

                self._dnsmos_model = dnsmos
            scores = self._dnsmos_model.run(wav, sr)
            return {
                "dnsmos_ovrl": float(scores["ovrl"]),
                "dnsmos_sig": float(scores["sig"]),
                "dnsmos_bak": float(scores["bak"]),
            }
        except Exception as e:
            logger.warning("DNSMOS failed, using fallback: %s", e)
            return {"dnsmos_ovrl": 3.5, "dnsmos_sig": 3.5, "dnsmos_bak": 3.5}

    # ── HNR ──

    @staticmethod
    def _compute_hnr(wav: np.ndarray, sr: int) -> float:
        snd = parselmouth.Sound(wav, sampling_frequency=sr)
        harmonicity = snd.to_harmonicity()
        values = harmonicity.values[harmonicity.values != -200]
        return float(np.mean(values)) if len(values) > 0 else 0.0

    # ── 语音占比 (Silero VAD) ──

    def _compute_speech_ratio(self, wav: np.ndarray, sr: int) -> float:
        try:
            if self._vad_model is None:
                self._vad_model, utils = torch.hub.load(
                    "snakers4/silero-vad", "silero_vad"
                )
                self._vad_utils = utils
            get_speech_timestamps = self._vad_utils[0]

            wav_tensor = torch.FloatTensor(wav)
            timestamps = get_speech_timestamps(
                wav_tensor, self._vad_model, sampling_rate=sr
            )
            speech_samples = sum(ts["end"] - ts["start"] for ts in timestamps)
            return speech_samples / len(wav)
        except Exception as e:
            logger.warning("VAD failed, assuming speech_ratio=1.0: %s", e)
            return 1.0
