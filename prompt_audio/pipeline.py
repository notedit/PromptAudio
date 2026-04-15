import json
import logging
from pathlib import Path

import numpy as np
import soundfile as sf

from .candidate_generator import Candidate, CandidateGenerator
from .config import PipelineConfig
from .preprocessor import AudioPreprocessor
from .quality_gate import QualityGate
from .scorer import PromptScorer
from .silence_handler import SilenceHandler
from .transcriber import Transcriber

logger = logging.getLogger(__name__)


class PromptAudioPipeline:
    """CosyVoice3 prompt audio 自动选取主流水线。"""

    def __init__(self, config: PipelineConfig | None = None):
        self.config = config or PipelineConfig()
        sr = self.config.target_sr

        self.preprocessor = AudioPreprocessor(sr)
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
            target_silence_ms=self.config.silence_pad_ms, sr=sr
        )

    def run(self, input_path: str, output_dir: str) -> dict:
        """执行完整 pipeline。

        Returns:
            {best: {...} | None, candidates: [...], error: str | None}
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        sr = self.config.target_sr

        # Step 1: 预处理
        logger.info("Step 1: Preprocessing %s", input_path)
        preprocessed_path = str(output_dir / "preprocessed.wav")
        wav = self.preprocessor.process(input_path, preprocessed_path)
        total_duration = len(wav) / sr
        logger.info(
            "Audio loaded: %.1fs, %d samples", total_duration, len(wav)
        )

        # Step 2: 转录 + 对齐
        logger.info("Step 2: Transcribing and aligning")
        sentences = self.transcriber.transcribe_and_align(preprocessed_path)
        if not sentences:
            return {
                "best": None,
                "candidates": [],
                "error": "No speech detected",
            }

        # 保存转录结果
        with open(output_dir / "transcription.json", "w", encoding="utf-8") as f:
            json.dump(sentences, f, ensure_ascii=False, indent=2)

        # Step 3: 候选片段生成
        logger.info("Step 3: Generating candidates")
        candidates = self.candidate_gen.generate(sentences, total_duration)
        logger.info("Generated %d candidates", len(candidates))
        if not candidates:
            return {
                "best": None,
                "candidates": [],
                "error": "No valid candidates in duration range",
            }

        # Step 4 + 5: 质量门控 + 评分
        logger.info("Step 4+5: Quality gate and scoring")
        scored = []
        passed_count = 0
        for idx, candidate in enumerate(candidates):
            segment_wav = self.candidate_gen.extract_audio(wav, candidate, sr)

            gate_result = self.quality_gate.check(segment_wav, sr)
            if not gate_result["passed"]:
                logger.debug(
                    "Candidate %d rejected: %s",
                    idx,
                    "; ".join(gate_result["reasons"]),
                )
                continue

            passed_count += 1
            score_result = self.scorer.score(
                segment_wav, sr, gate_result["metrics"]
            )

            scored.append(
                {
                    "start": round(candidate.start, 3),
                    "end": round(candidate.end, 3),
                    "duration": round(candidate.duration, 3),
                    "text": candidate.text,
                    "final_score": score_result["final_score"],
                    "quality_score": score_result["quality_score"],
                    "prosody_score": score_result["prosody_score"],
                    "gate_metrics": {
                        k: round(v, 4) for k, v in gate_result["metrics"].items()
                    },
                }
            )

        logger.info(
            "%d/%d candidates passed quality gate", passed_count, len(candidates)
        )

        if not scored:
            return {
                "best": None,
                "candidates": [],
                "error": "No candidates passed quality gate",
            }

        # 排序取 Top-K
        scored.sort(key=lambda x: x["final_score"], reverse=True)
        top_k = scored[: self.config.top_k]

        # 导出最优片段
        best = top_k[0]
        best_candidate = Candidate(
            start=best["start"], end=best["end"], text=best["text"]
        )
        best_wav = self.candidate_gen.extract_audio(wav, best_candidate, sr)
        best_wav = self.silence_handler.ensure_padding(best_wav)

        best_path = str(output_dir / "best_prompt.wav")
        sf.write(best_path, best_wav, sr, subtype="PCM_16")
        best["output_path"] = best_path
        logger.info(
            "Best prompt: %.2fs, score=%.4f, saved to %s",
            best["duration"],
            best["final_score"],
            best_path,
        )

        # 导出所有 Top-K 片段
        for rank, item in enumerate(top_k):
            c = Candidate(start=item["start"], end=item["end"], text=item["text"])
            c_wav = self.candidate_gen.extract_audio(wav, c, sr)
            c_wav = self.silence_handler.ensure_padding(c_wav)
            c_path = str(output_dir / f"prompt_rank{rank + 1}.wav")
            sf.write(c_path, c_wav, sr, subtype="PCM_16")
            item["output_path"] = c_path

        # 保存结果
        result = {"best": best, "candidates": top_k, "error": None}
        with open(output_dir / "result.json", "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        return result
