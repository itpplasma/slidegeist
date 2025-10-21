"""Audio transcription using faster-whisper."""

import logging
import platform
from pathlib import Path
from typing import TypedDict

from slidegeist.constants import (
    COMPRESSION_RATIO_THRESHOLD,
    DEFAULT_DEVICE,
    DEFAULT_WHISPER_MODEL,
    LOG_PROB_THRESHOLD,
    NO_SPEECH_THRESHOLD,
)

logger = logging.getLogger(__name__)


def is_mlx_available() -> bool:
    """Check if MLX is available (Apple Silicon Mac).

    Returns:
        True if running on Apple Silicon with MLX support, False otherwise.
    """
    # Check if we're on macOS ARM64 (Apple Silicon)
    if platform.system() != "Darwin":
        return False
    if platform.machine() != "arm64":
        return False

    # Try importing mlx-whisper
    try:
        import mlx_whisper  # type: ignore[import-untyped]  # noqa: F401
        return True
    except ImportError:
        return False


class Word(TypedDict):
    """A single word with timing information."""
    word: str
    start: float
    end: float


class Segment(TypedDict):
    """A transcript segment with timing and words."""
    start: float
    end: float
    text: str
    words: list[Word]


class TranscriptResult(TypedDict):
    """Complete transcription result."""
    language: str
    segments: list[Segment]


def transcribe_video(
    video_path: Path,
    model_size: str = DEFAULT_WHISPER_MODEL,
    device: str = DEFAULT_DEVICE,
    compute_type: str = "int8"
) -> TranscriptResult:
    """Transcribe video audio using faster-whisper.

    Args:
        video_path: Path to the video file.
        model_size: Whisper model size: tiny, base, small, medium, large-v3, large-v2, large.
        device: Device to use: 'cpu', 'cuda', or 'auto' (auto-detects MLX on Apple Silicon).
        compute_type: Computation type for CTranslate2.
                     Use 'int8' for CPU, 'float16' for GPU.

    Returns:
        Dictionary with language and segments containing timestamped text.

    Raises:
        ImportError: If faster-whisper is not installed.
        Exception: If transcription fails.
    """
    try:
        from faster_whisper import WhisperModel  # type: ignore[import-untyped]
    except ImportError:
        raise ImportError(
            "faster-whisper not installed. Install with: pip install faster-whisper"
        )

    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    # Auto-detect MLX on Apple Silicon
    use_mlx = False
    if device == "auto":
        if is_mlx_available():
            use_mlx = True
            device = "cpu"  # MLX uses its own backend
            logger.info("MLX detected - using MLX-optimized Whisper for Apple Silicon")
        elif platform.system() == "Darwin" and platform.machine() == "arm64":
            device = "cpu"
            logger.info("Apple Silicon detected but MLX not available, using CPU. Install with: pip install mlx-whisper")
        else:
            device = "cpu"
            logger.info("Auto-detected device: CPU")

    # Use MLX-optimized transcription if available
    if use_mlx:
        try:
            # Suppress MLX verbose debug output
            logging.getLogger("mlx").setLevel(logging.WARNING)
            logging.getLogger("mlx_whisper").setLevel(logging.WARNING)

            import mlx_whisper  # type: ignore[import-untyped]

            # Map faster-whisper model names to MLX model names
            mlx_model_map = {
                "large-v3": "mlx-community/whisper-large-v3-mlx",
                "large-v2": "mlx-community/whisper-large-v2-mlx",
                "large": "mlx-community/whisper-large-v2-mlx",
                "medium": "mlx-community/whisper-medium-mlx",
                "small": "mlx-community/whisper-small-mlx",
                "base": "mlx-community/whisper-base-mlx",
                "tiny": "mlx-community/whisper-tiny-mlx",
            }
            mlx_model = mlx_model_map.get(model_size, f"mlx-community/whisper-{model_size}-mlx")

            logger.info(f"Loading MLX Whisper model: {mlx_model}")
            result = mlx_whisper.transcribe(
                str(video_path),
                path_or_hf_repo=mlx_model,
                word_timestamps=True,
            )
            # Convert MLX result to our format
            mlx_segments: list[Segment] = []
            for segment in result.get("segments", []):
                mlx_words: list[Word] = []
                for word in segment.get("words", []):
                    mlx_words.append({
                        "word": word["word"],
                        "start": word["start"],
                        "end": word["end"]
                    })
                mlx_segments.append({
                    "start": segment["start"],
                    "end": segment["end"],
                    "text": segment["text"].strip(),
                    "words": mlx_words
                })
            logger.info(f"MLX transcription complete: {len(mlx_segments)} segments")
            return {
                "language": result.get("language", "unknown"),
                "segments": mlx_segments
            }
        except Exception as e:
            logger.warning(f"MLX transcription failed ({e}), falling back to faster-whisper")
            use_mlx = False

    # Adjust compute type based on device
    if device == "cuda" and compute_type == "int8":
        compute_type = "float16"

    logger.info(f"Loading Whisper model: {model_size} on {device}")
    model = WhisperModel(model_size, device=device, compute_type=compute_type)

    logger.info(f"Transcribing: {video_path.name}")
    segments_iterator, info = model.transcribe(
        str(video_path),
        word_timestamps=True,
        vad_filter=True,  # Voice activity detection for better accuracy
        compression_ratio_threshold=COMPRESSION_RATIO_THRESHOLD,
        log_prob_threshold=LOG_PROB_THRESHOLD,
        no_speech_threshold=NO_SPEECH_THRESHOLD,
    )

    # Convert iterator to list and extract data
    segments_list: list[Segment] = []
    for segment in segments_iterator:
        words_list: list[Word] = []
        if segment.words:
            for word in segment.words:
                words_list.append({
                    "word": word.word,
                    "start": word.start,
                    "end": word.end
                })

        segments_list.append({
            "start": segment.start,
            "end": segment.end,
            "text": segment.text.strip(),
            "words": words_list
        })

    logger.info(f"Transcription complete: {len(segments_list)} segments, language: {info.language}")

    return {
        "language": info.language,
        "segments": segments_list
    }
