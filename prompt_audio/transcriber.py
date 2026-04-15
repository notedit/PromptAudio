import logging

import whisperx

logger = logging.getLogger(__name__)


class Transcriber:
    """WhisperX 转录 + 强制对齐，输出词级/句级时间戳。"""

    def __init__(
        self,
        model_size: str = "large-v3",
        device: str = "cuda",
        language: str = "zh",
    ):
        self.device = device
        self.language = language
        logger.info("Loading WhisperX model: %s", model_size)
        self.model = whisperx.load_model(
            model_size, device=device, language=language
        )
        logger.info("Loading alignment model for: %s", language)
        self.align_model, self.align_metadata = whisperx.load_align_model(
            language_code=language, device=device
        )

    def transcribe_and_align(self, audio_path: str) -> list[dict]:
        """转录并对齐，返回句级结构列表。

        Returns:
            [{text, start, end, words: [{word, start, end}]}]
        """
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
            return_char_alignments=(self.language == "zh"),
        )

        sentences = self._build_sentences(result["segments"])
        logger.info("Detected %d sentences", len(sentences))
        return sentences

    @staticmethod
    def _build_sentences(segments: list) -> list[dict]:
        """将 WhisperX segments 整理为句级结构。"""
        sentences = []
        for seg in segments:
            words = seg.get("words", [])
            if not words:
                continue
            valid_words = [w for w in words if "start" in w and "end" in w]
            if not valid_words:
                continue
            sentences.append(
                {
                    "text": seg.get("text", ""),
                    "start": valid_words[0]["start"],
                    "end": valid_words[-1]["end"],
                    "words": valid_words,
                }
            )
        return sentences
