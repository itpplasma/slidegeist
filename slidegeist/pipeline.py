"""Main processing pipeline orchestration."""

import logging
from pathlib import Path

from slidegeist.export import export_srt
from slidegeist.ffmpeg import detect_scenes
from slidegeist.slides import extract_slides
from slidegeist.transcribe import transcribe_video

logger = logging.getLogger(__name__)


def process_video(
    video_path: Path,
    output_dir: Path,
    scene_threshold: float = 0.4,
    model: str = "base",
    device: str = "cpu",
    image_format: str = "jpg",
    skip_slides: bool = False,
    skip_transcription: bool = False
) -> dict[str, Path | list[Path]]:
    """Process video through the full pipeline.

    Args:
        video_path: Path to the input video file.
        output_dir: Directory where outputs will be saved.
        scene_threshold: Scene detection sensitivity (0.0-1.0).
        model: Whisper model size (tiny, base, small, medium, large).
        device: Device for transcription (cpu or cuda).
        image_format: Output image format (jpg or png).
        skip_slides: If True, skip slide extraction.
        skip_transcription: If True, skip audio transcription.

    Returns:
        Dictionary containing paths to outputs:
        - 'slides': List of slide image paths
        - 'transcript': Path to SRT file
        - 'output_dir': The output directory

    Raises:
        FileNotFoundError: If video file doesn't exist.
        Exception: If any processing step fails.
    """
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    logger.info(f"Processing video: {video_path}")
    logger.info(f"Output directory: {output_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, Path | list[Path]] = {
        'output_dir': output_dir
    }

    # Step 1: Scene detection (needed for slides)
    slide_paths: list[Path] = []
    if not skip_slides:
        logger.info("=" * 60)
        logger.info("STEP 1: Scene Detection")
        logger.info("=" * 60)

        scene_timestamps = detect_scenes(video_path, threshold=scene_threshold)

        if not scene_timestamps:
            logger.warning("No scene changes detected. Extracting single slide.")

        # Step 2: Extract slides
        logger.info("=" * 60)
        logger.info("STEP 2: Slide Extraction")
        logger.info("=" * 60)

        slide_paths = extract_slides(
            video_path,
            scene_timestamps,
            output_dir,
            image_format
        )
        results['slides'] = slide_paths

    # Step 3: Transcription
    transcript_path = output_dir / "transcript.srt"
    if not skip_transcription:
        logger.info("=" * 60)
        logger.info("STEP 3: Audio Transcription")
        logger.info("=" * 60)

        transcript_data = transcribe_video(
            video_path,
            model_size=model,
            device=device
        )

        # Step 4: Export SRT
        logger.info("=" * 60)
        logger.info("STEP 4: Export Transcript")
        logger.info("=" * 60)

        export_srt(transcript_data['segments'], transcript_path)
        results['transcript'] = transcript_path

    # Summary
    logger.info("=" * 60)
    logger.info("PROCESSING COMPLETE")
    logger.info("=" * 60)
    if not skip_slides:
        logger.info(f"✓ Extracted {len(slide_paths)} slides")
    if not skip_transcription:
        logger.info(f"✓ Created transcript: {transcript_path.name}")
    logger.info(f"✓ All outputs in: {output_dir}")

    return results


def process_slides_only(
    video_path: Path,
    output_dir: Path,
    scene_threshold: float = 0.4,
    image_format: str = "jpg"
) -> list[Path]:
    """Extract only slides from video (no transcription).

    Args:
        video_path: Path to the input video file.
        output_dir: Directory where slide images will be saved.
        scene_threshold: Scene detection sensitivity (0.0-1.0).
        image_format: Output image format (jpg or png).

    Returns:
        List of paths to extracted slide images.
    """
    logger.info("Extracting slides only (no transcription)")
    result = process_video(
        video_path,
        output_dir,
        scene_threshold=scene_threshold,
        image_format=image_format,
        skip_transcription=True
    )
    return result.get('slides', [])  # type: ignore


def process_transcript_only(
    video_path: Path,
    output_dir: Path,
    model: str = "base",
    device: str = "cpu"
) -> Path:
    """Extract only transcript from video (no slides).

    Args:
        video_path: Path to the input video file.
        output_dir: Directory where transcript will be saved.
        model: Whisper model size (tiny, base, small, medium, large).
        device: Device for transcription (cpu or cuda).

    Returns:
        Path to the SRT transcript file.
    """
    logger.info("Transcribing audio only (no slides)")
    result = process_video(
        video_path,
        output_dir,
        model=model,
        device=device,
        skip_slides=True
    )
    return result['transcript']  # type: ignore
