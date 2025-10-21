"""Export transcripts to subtitle formats."""

import logging
from pathlib import Path

from slidegeist.transcribe import Segment

logger = logging.getLogger(__name__)


def format_timestamp_srt(seconds: float) -> str:
    """Format seconds to SRT timestamp format: HH:MM:SS,mmm

    Args:
        seconds: Time in seconds (can include fractional seconds).

    Returns:
        Formatted timestamp string like '00:02:05,300'
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def export_srt(segments: list[Segment], output_path: Path) -> None:
    """Export transcription segments to SRT subtitle format.

    Args:
        segments: List of transcript segments with start, end, and text.
        output_path: Path where the SRT file will be saved.

    The SRT format is:
        sequence_number
        start_timestamp --> end_timestamp
        subtitle_text
        (blank line)
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Exporting {len(segments)} segments to SRT: {output_path}")

    with output_path.open('w', encoding='utf-8') as f:
        for i, segment in enumerate(segments, start=1):
            # Sequence number
            f.write(f"{i}\n")

            # Timestamps
            start = format_timestamp_srt(segment['start'])
            end = format_timestamp_srt(segment['end'])
            f.write(f"{start} --> {end}\n")

            # Text (strip whitespace and ensure it's not empty)
            text = segment['text'].strip()
            if not text:
                text = "[No speech detected]"
            f.write(f"{text}\n")

            # Blank line separator
            f.write("\n")

    logger.info(f"SRT export complete: {output_path}")
