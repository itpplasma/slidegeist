"""FFmpeg wrapper for video processing and scene detection."""

import logging
import shutil
import subprocess
from pathlib import Path

from slidegeist.constants import (
    DEFAULT_MIN_SCENE_LEN,
    DEFAULT_SCENE_THRESHOLD,
    DEFAULT_START_OFFSET,
)

logger = logging.getLogger(__name__)


class FFmpegError(Exception):
    """Raised when FFmpeg operations fail."""
    pass


def check_ffmpeg_available() -> bool:
    """Check if FFmpeg is installed and available in PATH.

    Returns:
        True if FFmpeg is available, False otherwise.
    """
    return shutil.which("ffmpeg") is not None


def get_video_duration(video_path: Path) -> float:
    """Get the duration of a video file in seconds.

    Args:
        video_path: Path to the video file.

    Returns:
        Duration in seconds.

    Raises:
        FFmpegError: If unable to determine video duration.
    """
    if not check_ffmpeg_available():
        raise FFmpegError("FFmpeg not found. Please install FFmpeg.")

    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(video_path)
    ]

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        return float(result.stdout.strip())
    except (subprocess.CalledProcessError, ValueError) as e:
        raise FFmpegError(f"Failed to get video duration: {e}")


def detect_scenes(
    video_path: Path,
    threshold: float = DEFAULT_SCENE_THRESHOLD,
    min_scene_len: float = DEFAULT_MIN_SCENE_LEN,
    start_offset: float = DEFAULT_START_OFFSET
) -> list[float]:
    """Detect slide changes in a video using Global Pixel Difference.

    Uses pixel-level differencing on binarized frames to detect content changes.
    Research shows this is the most effective method for lecture videos.

    Based on research: "An experimental comparative study on slide change detection
    in lecture videos" (Eruvaram et al., 2018)

    Args:
        video_path: Path to the video file.
        threshold: Scene detection threshold (0-1 scale, normalized pixel difference).
                  Lower = more sensitive. Typical range: 0.02-0.20 for presentations.
        min_scene_len: Minimum scene length in seconds (filters rapid clicks).
        start_offset: Skip first N seconds to avoid mouse movement during setup.

    Returns:
        List of timestamps (in seconds) where slide changes occur, sorted.

    Raises:
        FFmpegError: If video file not found or processing fails.
    """
    if not video_path.exists():
        raise FFmpegError(f"Video file not found: {video_path}")

    # Use Global Pixel Difference method with adaptive threshold selection
    # Research: Best performance for lecture videos (high recall and precision)
    from slidegeist.pixel_diff_detector import detect_slides_adaptive, detect_slides_pixel_diff

    # Use adaptive detection by default (automatically finds optimal threshold)
    # Falls back to fixed threshold if specified
    if threshold == DEFAULT_SCENE_THRESHOLD:
        # Use adaptive - no explicit threshold set by user
        return detect_slides_adaptive(
            video_path,
            start_offset=start_offset,
            min_scene_len=min_scene_len
        )
    else:
        # User specified explicit threshold - honor it
        return detect_slides_pixel_diff(
            video_path,
            start_offset=start_offset,
            min_scene_len=min_scene_len,
            threshold=threshold
        )


def extract_frame(
    video_path: Path,
    timestamp: float,
    output_path: Path,
    image_format: str = "jpg"
) -> None:
    """Extract a single frame from a video at the specified timestamp.

    Args:
        video_path: Path to the video file.
        timestamp: Time in seconds to extract the frame.
        output_path: Path where the frame image will be saved.
        image_format: Output image format ('jpg' or 'png').

    Raises:
        FFmpegError: If frame extraction fails.
    """
    if not check_ffmpeg_available():
        raise FFmpegError("FFmpeg not found. Please install FFmpeg.")

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Quality settings
    quality_args = []
    if image_format == "jpg":
        quality_args = ["-q:v", "2"]  # High quality JPEG (2-5 is good range)

    cmd = [
        "ffmpeg",
        "-ss", str(timestamp),  # Seek to timestamp
        "-i", str(video_path),
        "-frames:v", "1",  # Extract one frame
        *quality_args,
        "-strict", "unofficial",  # Allow non-standard YUV colorspace
        "-y",  # Overwrite output file
        str(output_path)
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        logger.debug(f"Extracted frame at {timestamp}s to {output_path}")
    except subprocess.CalledProcessError as e:
        raise FFmpegError(f"Failed to extract frame: {e.stderr}")
