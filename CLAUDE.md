# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PromptAudio is a Python project for managing and optimizing reference audio (prompt audio) for CosyVoice3's in-context learning TTS system. CosyVoice3 is highly sensitive to the prosody patterns in prompt audio, so this project focuses on two core strategies:

1. **Prompt Library with Tiered Labeling** - Build a tagging system for reference audio based on speech rate, emotion, and prosodic richness. At inference time, match the optimal prompt to the target scenario.
2. **Multi-Prompt Fusion** - When architecture permits, use weighted speaker embeddings from multiple prompts with different prosody styles to expand expressiveness.

## Build & Run

This is a Python project. No build/test/lint commands are configured yet. When they are added, update this section.

## Architecture

Project is in early stage. Architecture will be documented as it develops.
