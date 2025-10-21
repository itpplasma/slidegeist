"""FFmpeg wrapper for video processing and scene detection."""

import logging
import re
import shutil
import subprocess
from pathlib import Path

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


def detect_scenes(video_path: Path, threshold: float = 0.4) -> list[float]:
    """Detect scene changes in a video using FFmpeg's scene filter.

    Args:
        video_path: Path to the video file.
        threshold: Scene detection threshold (0.0-1.0). Higher values mean
                  fewer scene changes detected. Default 0.4 works well for slides.

    Returns:
        List of timestamps (in seconds) where scene changes occur, sorted.

    Raises:
        FFmpegError: If FFmpeg is not available or processing fails.
        ValueError: If threshold is not in valid range.
    """
    if not check_ffmpeg_available():
        raise FFmpegError("FFmpeg not found. Please install FFmpeg.")

    if not video_path.exists():
        raise FFmpegError(f"Video file not found: {video_path}")

    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"Scene threshold must be between 0.0 and 1.0, got {threshold}")

    logger.info(f"Detecting scenes with threshold {threshold}")

    cmd = [
        "ffmpeg",
        "-i", str(video_path),
        "-vf", f"select='gt(scene,{threshold})',showinfo",
        "-f", "null",
        "-"
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        # Scene detection info is in stderr
        if result.returncode != 0:
            raise FFmpegError(f"FFmpeg scene detection failed: {result.stderr}")
        output = result.stderr
    except subprocess.SubprocessError as e:
        raise FFmpegError(f"Failed to run FFmpeg: {e}")

    # Parse timestamps from showinfo output
    # Looking for lines like: pts_time:125.333
    timestamps = []
    pattern = re.compile(r"pts_time:(\d+\.?\d*)")

    for match in pattern.finditer(output):
        timestamp = float(match.group(1))
        timestamps.append(timestamp)

    timestamps.sort()
    logger.info(f"Found {len(timestamps)} scene changes")

    return timestamps


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
        "-y",  # Overwrite output file
        str(output_path)
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        logger.debug(f"Extracted frame at {timestamp}s to {output_path}")
    except subprocess.CalledProcessError as e:
        raise FFmpegError(f"Failed to extract frame: {e.stderr}")
