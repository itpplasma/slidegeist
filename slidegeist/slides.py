"""Slide extraction from videos based on scene detection."""

import logging
from pathlib import Path

from slidegeist.ffmpeg import extract_frame, get_video_duration

logger = logging.getLogger(__name__)


def format_slide_filename(index: int, total_slides: int) -> str:
    """Format zero-padded slide filename.

    Args:
        index: Slide index (0-based).
        total_slides: Total number of slides (for padding calculation).

    Returns:
        Formatted string like 'slide_000' or 'slide_0042'
    """
    # Determine padding based on total slides
    padding = max(3, len(str(total_slides - 1)))
    return f"slide_{index:0{padding}d}"


def extract_slides(
    video_path: Path,
    scene_timestamps: list[float],
    output_dir: Path,
    image_format: str = "jpg"
) -> list[tuple[int, float, float, Path]]:
    """Extract slides from video at scene change timestamps.

    Each slide is extracted at 80% through the segment to capture complete content.
    Returns metadata for each slide including index, time range, and file path.

    Args:
        video_path: Path to the video file.
        scene_timestamps: List of timestamps (seconds) where scenes change.
        output_dir: Directory to save slide images.
        image_format: Image format ('jpg' or 'png').

    Returns:
        List of (index, t_start, t_end, image_path) tuples in chronological order.

    Raises:
        ValueError: If timestamps are not sorted or contain invalid values.
    """
    if scene_timestamps and scene_timestamps != sorted(scene_timestamps):
        raise ValueError("Scene timestamps must be sorted")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Get video duration to know the end time
    duration = get_video_duration(video_path)

    # Build segment boundaries
    # First segment: 0 to first scene change
    # Middle segments: between scene changes
    # Last segment: last scene change to end
    boundaries = [0.0] + scene_timestamps + [duration]

    slide_data: list[tuple[int, float, float, Path]] = []
    total_slides = len(boundaries) - 1

    logger.info(f"Extracting {total_slides} slides")

    for i in range(total_slides):
        start_time = boundaries[i]
        end_time = boundaries[i + 1]
        segment_duration = end_time - start_time

        # Validate segment duration
        if segment_duration < 0.01:  # 10ms minimum
            logger.warning(f"Skipping very short segment at {start_time:.2f}s (duration: {segment_duration*1000:.1f}ms)")
            continue

        # Extract at 80% through the segment
        # This avoids both the initial transition AND the final transition
        # Captures the segment in its most stable, complete state
        if segment_duration < 2.0:
            # Short segments: use midpoint
            extract_time = start_time + segment_duration / 2
        else:
            # Extract at 80% of segment duration
            # This captures complete content before the next page flip
            extract_time = start_time + (segment_duration * 0.8)

        # Clamp extract_time to video duration (avoid ffmpeg seeking beyond end)
        extract_time = min(extract_time, duration - 0.1)

        # Create indexed filename
        filename_base = format_slide_filename(i, total_slides)
        filename = f"{filename_base}.{image_format}"
        output_path = output_dir / filename

        logger.debug(
            f"Slide {i}: {start_time:.2f}s - {end_time:.2f}s "
            f"(extracting at {extract_time:.2f}s)"
        )

        extract_frame(video_path, extract_time, output_path, image_format)
        slide_data.append((i, start_time, end_time, output_path))

    logger.info(f"Extracted {len(slide_data)} slides to {output_dir}")
    return slide_data
