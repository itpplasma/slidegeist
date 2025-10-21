"""TransNetV2-based slide detection for presentation videos.

TransNetV2 is a state-of-the-art deep learning model for shot boundary detection.
Paper: https://arxiv.org/abs/2008.04838
"""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def detect_slides_transnet(
    video_path: Path,
    start_offset: float = 3.0,
    min_scene_len: float = 0.5,
    max_resolution: int = 360,
    target_fps: float = 5.0
) -> list[float]:
    """Detect slide changes using TransNetV2 neural network.

    TransNetV2 is a state-of-the-art deep learning model that achieves
    F1 score of 0.898+ on shot boundary detection benchmarks.

    For speed optimization, this function pre-downscales videos and reduces FPS.
    TransNetV2 downsamples to 48x27 internally anyway, so quality loss is minimal.

    Args:
        video_path: Path to the video file.
        start_offset: Skip first N seconds.
        min_scene_len: Minimum scene length in seconds (post-processing filter).
        max_resolution: Maximum resolution (height) for processing. Videos larger
                       than this will be downscaled for faster processing.
                       Default: 360p (aggressive downscaling for speed).
        target_fps: Target FPS for processing. Lower = faster but might miss quick
                   transitions. Default: 5 FPS (good for slide detection).

    Returns:
        List of timestamps (seconds) where slide changes occur.
    """
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    try:
        from transnetv2_pytorch import TransNetV2
    except ImportError:
        raise RuntimeError(
            "TransNetV2 (PyTorch) not installed. Install with: "
            "pip install transnetv2-pytorch torch"
        )

    import cv2
    import tempfile
    import os

    # Check video resolution
    cap = cv2.VideoCapture(str(video_path))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    logger.info(
        f"Detecting slides with TransNetV2: start_offset={start_offset}s, "
        f"min_scene_len={min_scene_len}s, video={width}x{height}@{fps:.2f}fps"
    )

    # Pre-process video for speed if needed
    working_video = video_path
    temp_file = None
    needs_processing = height > max_resolution or fps > target_fps
    working_fps = fps  # Track the FPS of the video we'll actually process

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
            # Calculate frame skip to get approximately target_fps
            fps_ratio = int(round(fps / target_fps))
            actual_fps = fps / fps_ratio
            filters.append(f'fps=fps={actual_fps}')

        filter_str = ','.join(filters)

        # Calculate actual target fps for logging
        if fps > target_fps:
            fps_ratio = int(round(fps / target_fps))
            working_fps = fps / fps_ratio
        else:
            working_fps = fps

        logger.info(
            f"Optimizing video: {width}x{height}@{fps:.2f}fps -> "
            f"{new_width}x{new_height}@{working_fps:.2f}fps"
        )

        # Create temporary optimized video
        temp_file = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
        temp_file.close()

        import subprocess
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
        logger.info(f"Optimized video created at {working_video}")

    # Initialize model (will download weights on first run)
    logger.info("Loading TransNetV2 PyTorch model...")

    # Use GPU if available, but disable MPS (Apple Silicon) due to numerical issues
    import torch
    if torch.cuda.is_available():
        device = 'cuda'
        logger.info("Using NVIDIA GPU (CUDA) for acceleration")
    else:
        device = 'cpu'
        logger.info("Using CPU for inference (MPS disabled due to numerical inconsistencies)")

    model = TransNetV2(device=device)

    # Run inference
    logger.info("Running TransNetV2 inference...")
    try:
        video_frames, single_frame_predictions, all_frame_predictions = \
            model.predict_video(str(working_video))
    finally:
        # Clean up temp file if created
        if temp_file is not None:
            try:
                os.unlink(temp_file.name)
                logger.debug(f"Cleaned up temporary file {temp_file.name}")
            except Exception:
                pass

    # Convert PyTorch tensor to numpy for scene detection
    single_frame_np = single_frame_predictions.cpu().numpy()

    # Get scene boundaries (returns list of (start_frame, end_frame) tuples)
    # Lower threshold = more sensitive (default 0.5)
    scenes = model.predictions_to_scenes(single_frame_np, threshold=0.5)

    # Extract scene change timestamps (start of each scene, skip first)
    timestamps = []
    for i, (start_frame, end_frame) in enumerate(scenes):
        if i == 0:
            continue  # Skip first scene (start of video)

        # Convert frame to timestamp using the working video's FPS
        start_time = start_frame / working_fps

        # Apply start offset
        if start_time < start_offset:
            continue

        timestamps.append(start_time)

    # Filter by minimum scene length
    filtered_timestamps = []
    if timestamps:
        filtered_timestamps.append(timestamps[0])
        for ts in timestamps[1:]:
            if ts - filtered_timestamps[-1] >= min_scene_len:
                filtered_timestamps.append(ts)

    logger.info(f"Found {len(filtered_timestamps)} slide changes")
    return filtered_timestamps
