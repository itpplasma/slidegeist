"""Global Pixel Difference method for slide detection in lecture videos.

Based on research: "An experimental comparative study on slide change detection
in lecture videos" (Eruvaram et al., 2018).

This method was shown to have the best overall performance (high recall and
precision) for both slide-only and slide+presenter lecture videos.
"""

import logging
import os
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def detect_slides_pixel_diff(
    video_path: Path,
    start_offset: float = 3.0,
    min_scene_len: float = 0.5,
    threshold: float = 0.03,
    sample_interval: float = 1.0,
    max_resolution: int = 360,
    target_fps: float = 5.0
) -> list[float]:
    """Detect slide changes using Global Pixel Difference method.

    This method binarizes frames and computes pixel-level differences,
    normalized by image size. Research shows this is the most effective
    method for lecture video slide detection.

    For speed optimization, this function pre-downscales videos and reduces FPS.
    Since we're doing binary pixel difference, quality loss is minimal.

    Args:
        video_path: Path to the video file.
        start_offset: Skip first N seconds to avoid setup mouse movement.
        min_scene_len: Minimum scene length in seconds (filters rapid changes).
        threshold: Detection threshold (0-1). Default 0.03 from research.
                  Lower = more sensitive. Typical range: 0.02-0.05.
        sample_interval: Time interval between frames to compare (seconds).
                        Default 1.0s balances accuracy and speed.
        max_resolution: Maximum resolution (height) for processing. Videos larger
                       than this will be downscaled for faster processing.
                       Default: 360p (good balance of speed/accuracy).
        target_fps: Target FPS for processing. Lower = faster.
                   Default: 5 FPS (good for slide detection).

    Returns:
        List of timestamps (seconds) where slide changes occur.
    """
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    # Check video properties
    cap = cv2.VideoCapture(str(video_path))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    logger.info(
        f"Detecting slides with Global Pixel Difference: start_offset={start_offset}s, "
        f"min_scene_len={min_scene_len}s, threshold={threshold}, "
        f"sample_interval={sample_interval}s, video={width}x{height}@{fps:.2f}fps"
    )

    # Pre-process video for speed if needed
    working_video = video_path
    temp_file = None
    needs_processing = height > max_resolution or fps > target_fps
    working_fps = fps

    if needs_processing:
        scale = max_resolution / height if height > max_resolution else 1.0
        new_width = int(width * scale)
        new_height = int(height * scale)
        # Make dimensions divisible by 2 for h264
        new_width = (new_width // 2) * 2
        new_height = (new_height // 2) * 2

        # Build filter string
        filters = []
        if scale < 1.0:
            filters.append(f'scale={new_width}:{new_height}')
        if fps > target_fps:
            # Use integer ratio for fps to avoid encoding issues
            fps_ratio = int(round(fps / target_fps))
            actual_fps = fps / fps_ratio
            filters.append(f'fps=fps={actual_fps}')
            working_fps = actual_fps

        filter_str = ','.join(filters)

        logger.info(
            f"Preprocessing video: {width}x{height}@{fps:.2f}fps -> "
            f"{new_width}x{new_height}@{working_fps:.2f}fps"
        )

        # Create temporary optimized video
        temp_file = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
        temp_file.close()

        cmd = [
            'ffmpeg', '-i', str(video_path),
            '-vf', filter_str,
            '-c:v', 'libx264', '-preset', 'ultrafast', '-crf', '28',
            '-y', temp_file.name
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.error(f"FFmpeg error: {result.stderr}")
            raise RuntimeError("Video preprocessing failed")

        working_video = Path(temp_file.name)
        logger.info(f"Preprocessed video created at {working_video}")

    # Now process the working video
    cap = cv2.VideoCapture(str(working_video))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {working_video}")

    working_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    working_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / working_fps

    logger.info(
        f"Processing: {working_width}x{working_height}@{working_fps:.2f}fps, "
        f"{total_frames} frames, {duration:.1f}s duration"
    )

    # Calculate frame sampling parameters using working_fps
    start_frame = int(start_offset * working_fps)
    frame_interval = int(sample_interval * working_fps)
    min_frames_between = int(min_scene_len * working_fps)
    image_size = working_width * working_height

    timestamps = []
    prev_frame_binary = None
    last_change_frame = start_frame
    frame_num = start_frame

    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        while frame_num < total_frames:
            ret, frame = cap.read()
            if not ret:
                break

            # Process only at sample intervals
            if (frame_num - start_frame) % frame_interval != 0:
                frame_num += 1
                continue

            # Convert to grayscale and binarize
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(
                gray, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )

            if prev_frame_binary is not None:
                # Compute pixel-level difference
                diff = np.abs(binary.astype(np.int16) - prev_frame_binary.astype(np.int16))
                non_zero_count = np.count_nonzero(diff)

                # Normalize by image size
                normalized_diff = non_zero_count / image_size

                # Check if difference exceeds threshold
                if normalized_diff >= threshold:
                    # Check minimum scene length constraint
                    if frame_num - last_change_frame >= min_frames_between:
                        timestamp = frame_num / working_fps
                        timestamps.append(timestamp)
                        last_change_frame = frame_num

                        logger.debug(
                            f"Slide change at {timestamp:.2f}s "
                            f"(frame {frame_num}, diff={normalized_diff:.4f})"
                        )

            prev_frame_binary = binary
            frame_num += 1

    finally:
        cap.release()

        # Clean up temp file if created
        if temp_file is not None:
            try:
                os.unlink(temp_file.name)
                logger.debug(f"Cleaned up temporary file {temp_file.name}")
            except Exception:
                pass

    logger.info(f"Found {len(timestamps)} slide changes")
    return timestamps
