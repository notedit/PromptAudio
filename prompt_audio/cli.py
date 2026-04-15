import argparse
import logging
import sys

from .config import PRESETS, PipelineConfig
from .pipeline import PromptAudioPipeline


def main():
    parser = argparse.ArgumentParser(
        description="CosyVoice3 Prompt Audio Selector",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
examples:
  python -m prompt_audio input.wav -o ./output
  python -m prompt_audio input.mp3 -o ./output -p audiobook
  python -m prompt_audio input.wav -o ./output -k 10 --device cpu
""",
    )
    parser.add_argument("input", help="Input audio file path")
    parser.add_argument(
        "-o", "--output", default="./output", help="Output directory (default: ./output)"
    )
    parser.add_argument(
        "-p",
        "--preset",
        choices=PRESETS.keys(),
        default="default",
        help="Scoring preset (default: default)",
    )
    parser.add_argument(
        "-k", "--top-k", type=int, default=5, help="Number of top candidates (default: 5)"
    )
    parser.add_argument(
        "--device", default="cuda", help="Compute device (default: cuda)"
    )
    parser.add_argument(
        "--language", default="zh", help="Language code (default: zh)"
    )
    parser.add_argument(
        "--whisper-model",
        default=None,
        help="WhisperX model size (default: large-v3)",
    )
    parser.add_argument(
        "--min-duration",
        type=float,
        default=None,
        help="Min candidate duration in seconds",
    )
    parser.add_argument(
        "--max-duration",
        type=float,
        default=None,
        help="Max candidate duration in seconds",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Verbose logging"
    )
    args = parser.parse_args()

    # 日志
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # 构建配置
    config = PipelineConfig(**{
        k: v
        for k, v in vars(PRESETS[args.preset]).items()
    })
    config.top_k = args.top_k
    config.device = args.device
    config.language = args.language
    if args.whisper_model:
        config.whisper_model = args.whisper_model
    if args.min_duration is not None:
        config.min_duration = args.min_duration
    if args.max_duration is not None:
        config.max_duration = args.max_duration

    # 执行
    pipeline = PromptAudioPipeline(config)
    result = pipeline.run(args.input, args.output)

    # 输出结果
    if result["best"]:
        best = result["best"]
        print(f"\n{'='*60}")
        print(f"Best prompt audio:")
        print(f"  File:     {best['output_path']}")
        print(f"  Duration: {best['duration']:.2f}s")
        print(f"  Score:    {best['final_score']:.4f} (Q={best['quality_score']:.4f}, P={best['prosody_score']:.4f})")
        print(f"  Text:     {best['text']}")
        print(f"{'='*60}")
        print(f"\nTop {len(result['candidates'])} candidates:")
        for i, c in enumerate(result["candidates"]):
            print(
                f"  #{i+1}  score={c['final_score']:.4f}  "
                f"dur={c['duration']:.2f}s  "
                f"[{c['start']:.1f}-{c['end']:.1f}s]  "
                f"{c['text'][:40]}..."
                if len(c["text"]) > 40
                else f"  #{i+1}  score={c['final_score']:.4f}  "
                f"dur={c['duration']:.2f}s  "
                f"[{c['start']:.1f}-{c['end']:.1f}s]  "
                f"{c['text']}"
            )
        print(f"\nResults saved to {args.output}/result.json")
    else:
        print(f"\nError: {result.get('error', 'Unknown')}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
