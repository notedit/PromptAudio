# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PromptAudio is a Python tool that automatically selects optimal prompt audio segments from longer recordings for CosyVoice3's in-context learning TTS system. Given a <=5 minute audio file, it outputs the best <=10 second segment optimized for quality and prosodic expressiveness.

CosyVoice3's speech tokenizer (25 tokens/s) is highly sensitive to prompt prosody patterns — the prompt's speaking style transfers directly to synthesized output.

## Build & Run

```bash
# Install
pip install -r requirements.txt
# or
pip install -e .

# Run (requires GPU for WhisperX + emotion model)
python -m prompt_audio input.wav -o ./output
python -m prompt_audio input.wav -o ./output -p audiobook   # preset: audiobook/broadcast/noisy
python -m prompt_audio input.wav -o ./output --device cpu    # CPU fallback
```

## Architecture

5-step pipeline in `prompt_audio/`:

1. **preprocessor.py** — Resample to 16kHz mono, CosyVoice3 volume normalization (`wav / max * 0.6`)
2. **transcriber.py** — WhisperX ASR + forced alignment → word-level and sentence-level timestamps
3. **candidate_generator.py** — Enumerate consecutive sentence combinations within 5-10s duration window
4. **quality_gate.py** — Reject segments: DNSMOS < 3.5, clipping >= 1%, HNR < 15dB, speech ratio < 60%
5. **scorer.py** — Rank by `0.6 * Quality + 0.4 * Prosody` (DNSMOS/NISQA/SQUIM/SNR + F0 CV/energy CV/arousal)

Entry points: `cli.py` (CLI), `pipeline.py` (programmatic), `__main__.py` (module execution).

Config and presets in `config.py`. Silence padding (150ms) in `silence_handler.py`.

## Key Dependencies

- **whisperx**: ASR + Chinese forced alignment (wav2vec2 XLSR-53)
- **parselmouth**: F0/energy/HNR via Praat engine
- **speechmos**: DNSMOS non-intrusive quality scoring
- **silero-vad**: Voice activity detection for speech ratio
- **transformers**: wav2vec2 emotion model (arousal dimension)
