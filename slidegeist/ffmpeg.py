"""FFmpeg wrapper for video processing and scene detection."""

import logging
import shutil
import subprocess
from pathlib import Path

from slidegeist.constants import DEFAULT_SCENE_THRESHOLD

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


def detect_scenes(video_path: Path, threshold: float = DEFAULT_SCENE_THRESHOLD) -> list[float]:
    """Detect scene changes in a video using PySceneDetect.

    Args:
        video_path: Path to the video file.
        threshold: Scene detection threshold (lower = more sensitive).
                  Default 27.0 works well for most content including handwritten slides.

    Returns:
        List of timestamps (in seconds) where scene changes occur, sorted.

    Raises:
        FFmpegError: If video file not found or processing fails.
    """
    if not video_path.exists():
        raise FFmpegError(f"Video file not found: {video_path}")

    try:
        from scenedetect import ContentDetector, detect  # type: ignore[import-untyped]
    except ImportError:
        raise FFmpegError(
            "PySceneDetect not installed. Install with: pip install scenedetect[opencv]"
        )

    logger.info(f"Detecting scenes with threshold {threshold}")

    try:
        # Use ContentDetector for content-aware scene detection
        # This is better for gradual changes like handwritten slides
        scene_list = detect(str(video_path), ContentDetector(threshold=threshold))

        # Extract start times (in seconds) from scene list
        # scene_list contains tuples of (start_time, end_time)
        timestamps = [scene[0].get_seconds() for scene in scene_list]

        # Skip the first timestamp (0.0) as it's the start of the video
        if timestamps and timestamps[0] == 0.0:
            timestamps = timestamps[1:]

        timestamps.sort()
        logger.info(f"Found {len(timestamps)} scene changes")

        return timestamps
    except Exception as e:
        raise FFmpegError(f"Scene detection failed: {e}")


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
