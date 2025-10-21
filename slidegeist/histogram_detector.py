"""Histogram-based slide detection optimized for presentation videos.

This detector is specifically designed for detecting slide/page transitions
in presentation recordings (like GoodNotes) where:
- Page flips create large histogram changes
- Handwriting creates small gradual changes
- We want to detect only the page flips, not the writing
"""

import logging
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def detect_slides_histogram(
    video_path: Path,
    threshold: float = 0.75,
    min_scene_len: float = 0.5,
    start_offset: float = 3.0,
    sample_rate: int = 1,
    adaptive: bool = True
) -> list[float]:
    """Detect slide changes using histogram comparison.

    This method compares histograms of consecutive frames to detect
    large changes (page flips) while ignoring gradual changes (handwriting).

    Based on research: "Automatic detection of slide transitions in lecture videos"
    and "Shot Boundary Detection from Lecture Video Sequences Using HOG".

    Args:
        video_path: Path to the video file.
        threshold: Similarity threshold (0-1). Values < threshold indicate scene change.
                  Lower = more sensitive. Default 0.75 based on research.
                  Typical range: 0.7-0.85 for presentation videos.
        min_scene_len: Minimum scene length in seconds.
        start_offset: Skip first N seconds.
        sample_rate: Process every Nth frame (1=all frames, 2=every other, etc).
        adaptive: Use adaptive thresholding based on video statistics (recommended).

    Returns:
        List of timestamps (seconds) where slide changes occur.
    """
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    logger.info(
        f"Detecting slides with histogram method: threshold={threshold}, "
        f"min_scene_len={min_scene_len}s, start_offset={start_offset}s, "
        f"sample_rate={sample_rate}, adaptive={adaptive}"
    )

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    start_frame = int(start_offset * fps)
    min_frames_between = int(min_scene_len * fps)

    logger.info(f"Video: {fps:.2f} fps, {total_frames} frames")

    # Seek to start offset
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    prev_hist = None
    similarities = []
    scene_changes = []
    last_scene_frame = start_frame
    frame_count = start_frame

    # First pass: collect similarity scores if adaptive
    if adaptive:
        logger.info("Pass 1: Collecting similarity statistics...")
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if (frame_count - start_frame) % sample_rate != 0:
                frame_count += 1
                continue

            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist([hsv], [0, 1, 2], None,
                               [50, 60, 60],
                               [0, 180, 0, 256, 0, 256])
            hist = cv2.normalize(hist, hist).flatten()

            if prev_hist is not None:
                similarity = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL)
                similarities.append(similarity)

            prev_hist = hist
            frame_count += 1

        # Calculate adaptive threshold from statistics
        similarities_array = np.array(similarities)
        mean_sim = np.mean(similarities_array)
        std_sim = np.std(similarities_array)

        # Adaptive threshold: mean - 2*std (catches values 2 standard deviations below mean)
        # This typically catches significant changes while ignoring noise
        adaptive_threshold = mean_sim - 2.0 * std_sim

        # Clamp to reasonable range
        adaptive_threshold = max(0.5, min(0.85, adaptive_threshold))

        logger.info(
            f"Similarity statistics: mean={mean_sim:.3f}, std={std_sim:.3f}, "
            f"adaptive_threshold={adaptive_threshold:.3f}"
        )
        threshold = adaptive_threshold

        # Reset for second pass
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        prev_hist = None
        frame_count = start_frame
        last_scene_frame = start_frame

    # Second pass (or only pass if not adaptive): detect scenes
    logger.info("Pass 2: Detecting scene changes...")
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if (frame_count - start_frame) % sample_rate != 0:
                frame_count += 1
                continue

            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist([hsv], [0, 1, 2], None,
                               [50, 60, 60],
                               [0, 180, 0, 256, 0, 256])
            hist = cv2.normalize(hist, hist).flatten()

            if prev_hist is not None:
                similarity = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL)

                if similarity < threshold:
                    if frame_count - last_scene_frame >= min_frames_between:
                        timestamp = frame_count / fps
                        scene_changes.append(timestamp)
                        last_scene_frame = frame_count
                        logger.debug(
                            f"Scene change at {timestamp:.2f}s "
                            f"(similarity: {similarity:.3f})"
                        )

            prev_hist = hist
            frame_count += 1

    finally:
        cap.release()

    logger.info(f"Found {len(scene_changes)} slide changes")
    return scene_changes
