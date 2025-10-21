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
    """Detect slide changes in a video using histogram comparison.

    Uses OpenCV histogram correlation to detect page flips while ignoring
    gradual handwriting changes. Optimized for presentation videos.

    Based on research: "Automatic detection of slide transitions in lecture videos"

    Args:
        video_path: Path to the video file.
        threshold: Scene detection threshold (0-100 scale).
                  Lower = more sensitive. Default 25 -> 0.75 correlation.
                  Typical range: 20-30 for presentations.
        min_scene_len: Minimum scene length in seconds (filters rapid clicks).
        start_offset: Skip first N seconds to avoid mouse movement during setup.

    Returns:
        List of timestamps (in seconds) where slide changes occur, sorted.

    Raises:
        FFmpegError: If video file not found or processing fails.
    """
    if not video_path.exists():
        raise FFmpegError(f"Video file not found: {video_path}")

    # Use TransNetV2 neural network (state-of-the-art shot boundary detection)
    # Paper: https://arxiv.org/abs/2008.04838
    from slidegeist.transnet_detector import detect_slides_transnet

    return detect_slides_transnet(
        video_path,
        start_offset=start_offset,
        min_scene_len=min_scene_len
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
