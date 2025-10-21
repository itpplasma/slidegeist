"""Audio transcription using faster-whisper."""

import logging
from pathlib import Path
from typing import TypedDict

logger = logging.getLogger(__name__)


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
    model_size: str = "base",
    device: str = "cpu",
    compute_type: str = "int8"
) -> TranscriptResult:
    """Transcribe video audio using faster-whisper.

    Args:
        video_path: Path to the video file.
        model_size: Whisper model size: tiny, base, small, medium, large.
        device: Device to use: 'cpu' or 'cuda'.
        compute_type: Computation type for CTranslate2.
                     Use 'int8' for CPU, 'float16' for GPU.

    Returns:
        Dictionary with language and segments containing timestamped text.

    Raises:
        ImportError: If faster-whisper is not installed.
        Exception: If transcription fails.
    """
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        raise ImportError(
            "faster-whisper not installed. Install with: pip install faster-whisper"
        )

    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

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
