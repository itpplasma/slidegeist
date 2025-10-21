"""Command-line interface for Slidegeist."""

import argparse
import logging
import sys
from pathlib import Path

from slidegeist import __version__
from slidegeist.constants import (
    DEFAULT_DEVICE,
    DEFAULT_IMAGE_FORMAT,
    DEFAULT_MIN_SCENE_LEN,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_SCENE_THRESHOLD,
    DEFAULT_START_OFFSET,
    DEFAULT_WHISPER_MODEL,
)
from slidegeist.ffmpeg import check_ffmpeg_available
from slidegeist.pipeline import process_slides_only, process_transcript_only, process_video

logger = logging.getLogger(__name__)


def setup_logging(verbose: bool = False) -> None:
    """Configure logging output.

    Args:
        verbose: If True, set log level to DEBUG, otherwise INFO.
    """
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S"
    )


def check_prerequisites() -> None:
    """Check that required external tools are available.

    Raises:
        SystemExit: If FFmpeg is not found.
    """
    if not check_ffmpeg_available():
        logger.error("FFmpeg not found in PATH")
        logger.error("Please install FFmpeg:")
        logger.error("  macOS:        brew install ffmpeg")
        logger.error("  Ubuntu/Debian: sudo apt-get install ffmpeg")
        logger.error("  Windows:      winget install ffmpeg")
        sys.exit(1)


def handle_process(args: argparse.Namespace) -> None:
    """Handle 'slidegeist process' command."""
    try:
        check_prerequisites()

        result = process_video(
            video_path=args.input,
            output_dir=args.out,
            scene_threshold=args.scene_threshold,
            min_scene_len=args.min_scene_len,
            start_offset=args.start_offset,
            model=args.model,
            device=args.device,
            image_format=args.format
        )

        print("\n" + "=" * 60)
        print("✓ Processing complete!")
        print("=" * 60)
        if 'slides' in result:
            print(f"  Slides:     {len(result['slides'])} images")  # type: ignore
        if 'srt' in result:
            print(f"  SRT:        {result['srt']}")
        if 'manifest' in result:
            print(f"  Manifest:   {result['manifest']}")
        print(f"  Output dir: {result['output_dir']}")
        print("=" * 60)

    except FileNotFoundError as e:
        logger.error(str(e))
        sys.exit(1)
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        if args.verbose:
            logger.exception("Full traceback:")
        sys.exit(1)


def handle_slides(args: argparse.Namespace) -> None:
    """Handle 'slidegeist slides' command."""
    try:
        check_prerequisites()

        result = process_slides_only(
            video_path=args.input,
            output_dir=args.out,
            scene_threshold=args.scene_threshold,
            min_scene_len=args.min_scene_len,
            start_offset=args.start_offset,
            image_format=args.format
        )

        print(f"\n✓ Extracted {len(result['slides'])} slides")  # type: ignore
        print(f"  Manifest:   {result['manifest']}")
        print(f"  Output dir: {result['output_dir']}")

    except Exception as e:
        logger.error(f"Slide extraction failed: {e}")
        if args.verbose:
            logger.exception("Full traceback:")
        sys.exit(1)


def handle_transcribe(args: argparse.Namespace) -> None:
    """Handle 'slidegeist transcribe' command."""
    try:
        check_prerequisites()

        result = process_transcript_only(
            video_path=args.input,
            output_dir=args.out,
            model=args.model,
            device=args.device
        )

        print(f"\n✓ Transcription complete")
        if 'srt' in result:
            print(f"  SRT:        {result['srt']}")
        print(f"  Manifest:   {result['manifest']}")
        print(f"  Output dir: {result['output_dir']}")

    except Exception as e:
        logger.error(f"Transcription failed: {e}")
        if args.verbose:
            logger.exception("Full traceback:")
        sys.exit(1)


def main() -> None:
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(
        prog="slidegeist",
        description="Extract slides and transcripts from lecture videos",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process full video (uses large-v3 model with auto-detection)
  slidegeist process lecture.mp4

  # Use GPU explicitly
  slidegeist process lecture.mp4 --device cuda

  # Use smaller/faster model
  slidegeist process lecture.mp4 --model base

  # Extract only slides
  slidegeist slides lecture.mp4

  # Extract only transcript
  slidegeist transcribe lecture.mp4
        """
    )

    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}"
    )

    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # Process command (full pipeline)
    process_parser = subparsers.add_parser(
        "process",
        help="Process video (extract slides and transcript)"
    )
    process_parser.add_argument(
        "input",
        type=Path,
        help="Input video file"
    )
    process_parser.add_argument(
        "--out",
        type=Path,
        default=Path(DEFAULT_OUTPUT_DIR),
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR}/)"
    )
    process_parser.add_argument(
        "--scene-threshold",
        type=float,
        default=DEFAULT_SCENE_THRESHOLD,
        metavar="NUM",
        help=f"Scene detection threshold 0.02-0.05, lower=more sensitive (default: {DEFAULT_SCENE_THRESHOLD})"
    )
    process_parser.add_argument(
        "--min-scene-len",
        type=float,
        default=DEFAULT_MIN_SCENE_LEN,
        metavar="SEC",
        help=f"Minimum scene length in seconds (default: {DEFAULT_MIN_SCENE_LEN})"
    )
    process_parser.add_argument(
        "--start-offset",
        type=float,
        default=DEFAULT_START_OFFSET,
        metavar="SEC",
        help=f"Skip first N seconds to avoid mouse movement (default: {DEFAULT_START_OFFSET})"
    )
    process_parser.add_argument(
        "--model",
        default=DEFAULT_WHISPER_MODEL,
        choices=["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"],
        help=f"Whisper model size (default: {DEFAULT_WHISPER_MODEL})"
    )
    process_parser.add_argument(
        "--device",
        default=DEFAULT_DEVICE,
        choices=["cpu", "cuda", "auto"],
        help=f"Processing device (default: {DEFAULT_DEVICE} - uses MLX on Apple Silicon if available)"
    )
    process_parser.add_argument(
        "--format",
        default=DEFAULT_IMAGE_FORMAT,
        choices=["jpg", "png"],
        help=f"Slide image format (default: {DEFAULT_IMAGE_FORMAT})"
    )

    # Slides command
    slides_parser = subparsers.add_parser(
        "slides",
        help="Extract only slides (no transcription)"
    )
    slides_parser.add_argument(
        "input",
        type=Path,
        help="Input video file"
    )
    slides_parser.add_argument(
        "--out",
        type=Path,
        default=Path(DEFAULT_OUTPUT_DIR),
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR}/)"
    )
    slides_parser.add_argument(
        "--scene-threshold",
        type=float,
        default=DEFAULT_SCENE_THRESHOLD,
        metavar="NUM",
        help=f"Scene detection threshold 0.02-0.05, lower=more sensitive (default: {DEFAULT_SCENE_THRESHOLD})"
    )
    slides_parser.add_argument(
        "--min-scene-len",
        type=float,
        default=DEFAULT_MIN_SCENE_LEN,
        metavar="SEC",
        help=f"Minimum scene length in seconds (default: {DEFAULT_MIN_SCENE_LEN})"
    )
    slides_parser.add_argument(
        "--start-offset",
        type=float,
        default=DEFAULT_START_OFFSET,
        metavar="SEC",
        help=f"Skip first N seconds to avoid mouse movement (default: {DEFAULT_START_OFFSET})"
    )
    slides_parser.add_argument(
        "--format",
        default=DEFAULT_IMAGE_FORMAT,
        choices=["jpg", "png"],
        help=f"Slide image format (default: {DEFAULT_IMAGE_FORMAT})"
    )

    # Transcribe command
    transcribe_parser = subparsers.add_parser(
        "transcribe",
        help="Extract only transcript (no slides)"
    )
    transcribe_parser.add_argument(
        "input",
        type=Path,
        help="Input video file"
    )
    transcribe_parser.add_argument(
        "--out",
        type=Path,
        default=Path(DEFAULT_OUTPUT_DIR),
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR}/)"
    )
    transcribe_parser.add_argument(
        "--model",
        default=DEFAULT_WHISPER_MODEL,
        choices=["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"],
        help=f"Whisper model size (default: {DEFAULT_WHISPER_MODEL})"
    )
    transcribe_parser.add_argument(
        "--device",
        default=DEFAULT_DEVICE,
        choices=["cpu", "cuda", "auto"],
        help=f"Processing device (default: {DEFAULT_DEVICE} - uses MLX on Apple Silicon if available)"
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.verbose)

    # Dispatch to appropriate handler
    if args.command == "process":
        handle_process(args)
    elif args.command == "slides":
        handle_slides(args)
    elif args.command == "transcribe":
        handle_transcribe(args)


if __name__ == "__main__":
    main()
