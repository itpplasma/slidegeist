"""Export slides with transcripts to JSON format."""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from slidegeist.transcribe import Segment

logger = logging.getLogger(__name__)


def export_slides_json(
    video_path: Path,
    slide_metadata: list[tuple[int, float, float, Path]],
    transcript_segments: list[Segment],
    output_path: Path,
    model: str
) -> None:
    """Export slides with their transcripts to JSON format.

    Args:
        video_path: Path to the source video file.
        slide_metadata: List of (index, t_start, t_end, image_path) tuples.
        transcript_segments: List of Whisper transcript segments.
        output_path: Path where the JSON file will be saved.
        model: Whisper model used for transcription.

    The JSON structure groups transcript text by slide time ranges.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Creating slides.json with {len(slide_metadata)} slides")

    # Build the slides data structure
    slides_data = []

    for index, t_start, t_end, image_path in slide_metadata:
        # Collect all transcript text that falls within this slide's time range
        transcript_parts = []

        for segment in transcript_segments:
            seg_start = segment['start']
            seg_end = segment['end']

            # Include segment if its start time is within this slide's time range
            # This avoids duplicates while ensuring complete coverage
            if seg_start >= t_start and seg_start < t_end:
                text = segment['text'].strip()
                if text:
                    transcript_parts.append(text)

        # Combine all text for this slide
        full_transcript = ' '.join(transcript_parts)

        # Make image path relative to output directory
        relative_path = image_path.relative_to(output_path.parent)

        slide_entry = {
            "slide_number": index,
            "image_path": str(relative_path),
            "time_start": int(t_start),
            "time_end": int(t_end),
            "transcript": full_transcript
        }

        slides_data.append(slide_entry)

    # Build final JSON structure
    json_data = {
        "metadata": {
            "video_file": video_path.name,
            "duration_seconds": int(slide_metadata[-1][2]) if slide_metadata else 0,
            "processed_at": datetime.now(timezone.utc).isoformat(),
            "model": model
        },
        "slides": slides_data
    }

    # Write JSON file
    with output_path.open('w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)

    logger.info(f"Exported slides.json: {output_path}")
