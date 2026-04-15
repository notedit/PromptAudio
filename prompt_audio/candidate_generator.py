import numpy as np
from dataclasses import dataclass, field


@dataclass
class Candidate:
    """一个候选 prompt 片段。"""

    start: float
    end: float
    text: str
    sentence_indices: list = field(default_factory=list)

    @property
    def duration(self) -> float:
        return self.end - self.start


class CandidateGenerator:
    """基于句级时间戳，生成满足时长约束的候选片段列表。"""

    def __init__(
        self,
        min_duration: float = 5.0,
        max_duration: float = 10.0,
        silence_pad: float = 0.15,
        min_speech_ratio: float = 0.6,
    ):
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.silence_pad = silence_pad
        self.min_speech_ratio = min_speech_ratio

    def generate(
        self, sentences: list[dict], total_duration: float
    ) -> list[Candidate]:
        """遍历连续句子组合，生成满足时长约束的候选片段。"""
        candidates = []
        n = len(sentences)

        for i in range(n):
            text_parts = []
            for j in range(i, n):
                text_parts.append(sentences[j]["text"])

                start = max(0.0, sentences[i]["start"] - self.silence_pad)
                end = min(total_duration, sentences[j]["end"] + self.silence_pad)
                duration = end - start

                if duration < self.min_duration:
                    continue
                if duration > self.max_duration:
                    break

                candidates.append(
                    Candidate(
                        start=start,
                        end=end,
                        text="".join(text_parts),
                        sentence_indices=list(range(i, j + 1)),
                    )
                )

        return candidates

    @staticmethod
    def extract_audio(
        wav: np.ndarray, candidate: Candidate, sr: int = 16000
    ) -> np.ndarray:
        """从完整音频中截取候选片段。"""
        start_sample = int(candidate.start * sr)
        end_sample = int(candidate.end * sr)
        return wav[start_sample:end_sample]
