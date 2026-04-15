import logging

import librosa
import numpy as np
import parselmouth

logger = logging.getLogger(__name__)


class PromptScorer:
    """综合评分器。

    Final_Score = w1 * Q + w2 * P

    Q = 0.50*DNSMOS + 0.20*NISQA + 0.15*SQUIM_PESQ + 0.15*SNR
    P = 0.35*F0_CV + 0.25*Energy_CV + 0.20*SpeechRateVar + 0.20*Arousal
    """

    def __init__(self, w_quality: float = 0.6, w_prosody: float = 0.4):
        self.w_quality = w_quality
        self.w_prosody = w_prosody
        self._emotion_processor = None
        self._emotion_model = None

    def score(self, wav: np.ndarray, sr: int, gate_metrics: dict) -> dict:
        """计算综合评分。gate_metrics 来自 QualityGate.check()。"""
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
            "final_score": round(final_score, 6),
            "quality_score": round(q_score, 6),
            "prosody_score": round(p_score, 6),
            "quality_metrics": q_metrics,
            "prosody_metrics": p_metrics,
        }

    # ── 质量分 Q ──

    def _compute_quality(
        self, wav: np.ndarray, sr: int, gate_metrics: dict
    ) -> dict:
        metrics = {}

        # DNSMOS（复用门控结果）
        metrics["dnsmos_ovrl_norm"] = self._norm(
            gate_metrics["dnsmos_ovrl"], 1.0, 5.0
        )

        # NISQA
        nisqa = self._get_nisqa(wav, sr)
        metrics["nisqa_mos_norm"] = self._norm(nisqa, 1.0, 5.0)

        # SQUIM PESQ
        pesq = self._get_squim_pesq(wav, sr)
        metrics["squim_pesq_norm"] = self._norm(pesq, 1.0, 4.5)

        # SNR
        snr = self._estimate_snr(wav)
        metrics["snr_norm"] = self._norm(snr, 0.0, 40.0)

        return metrics

    def _get_nisqa(self, wav: np.ndarray, sr: int) -> float:
        try:
            import torch
            from torchmetrics.audio import (
                NonIntrusiveSpeechQualityAssessment,
            )

            model = NonIntrusiveSpeechQualityAssessment()
            tensor = torch.FloatTensor(wav).unsqueeze(0)
            return float(model(tensor).item())
        except Exception as e:
            logger.debug("NISQA unavailable: %s", e)
            return 3.0

    def _get_squim_pesq(self, wav: np.ndarray, sr: int) -> float:
        try:
            import torch
            import torchaudio

            model = torchaudio.pipelines.SQUIM_OBJECTIVE.get_model()
            tensor = torch.FloatTensor(wav).unsqueeze(0)
            if sr != 16000:
                tensor = torchaudio.functional.resample(tensor, sr, 16000)
            _stoi, pesq, _si_sdr = model(tensor)
            return float(pesq.item())
        except Exception as e:
            logger.debug("SQUIM unavailable: %s", e)
            return 2.5

    @staticmethod
    def _estimate_snr(wav: np.ndarray) -> float:
        """用最安静的 10% 帧估算噪底，计算 SNR。"""
        frame_len = 400  # 25ms at 16kHz
        n_frames = max(1, (len(wav) - frame_len) // frame_len)
        energies = []
        for i in range(n_frames):
            frame = wav[i * frame_len : (i + 1) * frame_len]
            energies.append(np.mean(frame**2))
        energies.sort()
        noise_energy = np.mean(energies[: max(1, len(energies) // 10)])
        signal_energy = np.mean(energies)
        if noise_energy < 1e-10:
            return 40.0
        return float(10 * np.log10(signal_energy / noise_energy))

    # ── 韵律丰富度分 P ──

    def _compute_prosody(self, wav: np.ndarray, sr: int) -> dict:
        metrics = {}
        snd = parselmouth.Sound(wav, sampling_frequency=sr)

        # F0 变异系数
        pitch = snd.to_pitch_ac()
        f0 = pitch.selected_array["frequency"]
        f0 = f0[f0 > 0]
        if len(f0) > 0 and np.mean(f0) > 0:
            f0_cv = float(np.std(f0) / np.mean(f0))
        else:
            f0_cv = 0.0
        metrics["f0_cv_norm"] = self._norm(f0_cv, 0.0, 0.5)

        # 能量变异系数
        intensity = snd.to_intensity()
        int_values = intensity.values[0]
        if np.mean(int_values) > 0:
            energy_cv = float(np.std(int_values) / np.mean(int_values))
        else:
            energy_cv = 0.0
        metrics["energy_cv_norm"] = self._norm(energy_cv, 0.0, 0.5)

        # 语速变化（RMS 帧间差分的标准差）
        rms = librosa.feature.rms(y=wav, frame_length=2048, hop_length=512)[0]
        if len(rms) > 1:
            speech_rate_var = float(np.std(np.diff(rms)))
        else:
            speech_rate_var = 0.0
        metrics["speech_rate_var_norm"] = self._norm(speech_rate_var, 0.0, 0.05)

        # 情感激活度
        arousal = self._get_arousal(wav, sr)
        metrics["arousal_norm"] = self._norm(arousal, 0.0, 1.0)

        return metrics

    def _get_arousal(self, wav: np.ndarray, sr: int) -> float:
        """wav2vec2 情感维度模型提取 arousal。"""
        try:
            import torch
            from transformers import (
                Wav2Vec2Processor,
                Wav2Vec2ForSequenceClassification,
            )

            if self._emotion_model is None:
                model_name = (
                    "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"
                )
                self._emotion_processor = Wav2Vec2Processor.from_pretrained(
                    model_name
                )
                self._emotion_model = (
                    Wav2Vec2ForSequenceClassification.from_pretrained(
                        model_name
                    )
                )

            inputs = self._emotion_processor(
                wav, sampling_rate=sr, return_tensors="pt"
            )
            with torch.no_grad():
                outputs = self._emotion_model(**inputs)
            # logits: [arousal, dominance, valence]
            arousal = float(outputs.logits[0, 0].item())
            return max(0.0, min(1.0, arousal))
        except Exception as e:
            logger.debug("Emotion model unavailable: %s", e)
            return 0.5

    @staticmethod
    def _norm(value: float, min_val: float, max_val: float) -> float:
        if max_val <= min_val:
            return 0.0
        return max(0.0, min(1.0, (value - min_val) / (max_val - min_val)))
