"""Main processing pipeline orchestration."""

import logging
from pathlib import Path

from slidegeist.constants import (
    DEFAULT_DEVICE,
    DEFAULT_IMAGE_FORMAT,
    DEFAULT_MIN_SCENE_LEN,
    DEFAULT_SCENE_THRESHOLD,
    DEFAULT_START_OFFSET,
    DEFAULT_WHISPER_MODEL,
)
from slidegeist.export import export_srt
from slidegeist.ffmpeg import detect_scenes, get_video_duration
from slidegeist.manifest import create_manifest, save_manifest
from slidegeist.slides import extract_slides
from slidegeist.transcribe import transcribe_video

logger = logging.getLogger(__name__)


def process_video(
    video_path: Path,
    output_dir: Path,
    scene_threshold: float = DEFAULT_SCENE_THRESHOLD,
    min_scene_len: float = DEFAULT_MIN_SCENE_LEN,
    start_offset: float = DEFAULT_START_OFFSET,
    model: str = DEFAULT_WHISPER_MODEL,
    device: str = DEFAULT_DEVICE,
    image_format: str = DEFAULT_IMAGE_FORMAT,
    skip_slides: bool = False,
    skip_transcription: bool = False
) -> dict[str, Path | list[Path]]:
    """Process video through the full pipeline.

    Args:
        video_path: Path to the input video file.
        output_dir: Directory where outputs will be saved.
        scene_threshold: Scene detection threshold (0-100, lower = more sensitive).
        min_scene_len: Minimum scene length in seconds.
        start_offset: Skip first N seconds to avoid setup noise.
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

    # Create slidegeist subdirectory structure
    slidegeist_dir = output_dir / "slidegeist"
    slides_dir = slidegeist_dir / "slides"
    slidegeist_dir.mkdir(parents=True, exist_ok=True)
    slides_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, Path | list[Path]] = {
        'output_dir': slidegeist_dir
    }

    # Get video duration
    duration = get_video_duration(video_path)

    # Step 1: Scene detection (needed for slides)
    slide_metadata: list[tuple[int, float, float, Path]] = []
    scene_timestamps: list[float] = []
    if not skip_slides:
        logger.info("=" * 60)
        logger.info("STEP 1: Scene Detection")
        logger.info("=" * 60)

        scene_timestamps = detect_scenes(
            video_path,
            threshold=scene_threshold,
            min_scene_len=min_scene_len,
            start_offset=start_offset
        )

        if not scene_timestamps:
            logger.warning("No scene changes detected. Extracting single slide.")

        # Step 2: Extract slides to slides/ subdirectory
        logger.info("=" * 60)
        logger.info("STEP 2: Slide Extraction")
        logger.info("=" * 60)

        slide_metadata = extract_slides(
            video_path,
            scene_timestamps,
            slides_dir,
            image_format
        )
        results['slides'] = [path for _, _, _, path in slide_metadata]

    # Step 3: Transcription
    transcript_data = None
    if not skip_transcription:
        logger.info("=" * 60)
        logger.info("STEP 3: Audio Transcription")
        logger.info("=" * 60)

        transcript_data = transcribe_video(
            video_path,
            model_size=model,
            device=device
        )

        # Step 4: Export SRT (optional, legacy format)
        logger.info("=" * 60)
        logger.info("STEP 4: Export SRT (legacy)")
        logger.info("=" * 60)

        srt_path = output_dir / "transcript.srt"
        export_srt(transcript_data['segments'], srt_path)
        results['srt'] = srt_path

    # Step 5: Generate JSON manifest (primary output)
    logger.info("=" * 60)
    logger.info("STEP 5: Generate JSON Manifest")
    logger.info("=" * 60)

    # Convert slide metadata to relative paths for manifest
    manifest_slides = []
    for idx, t_start, t_end, abs_path in slide_metadata:
        rel_path = f"slides/{abs_path.name}"
        manifest_slides.append((idx, t_start, t_end, rel_path))

    # Build detector config string
    detector_config = f"pixel-diff(threshold={scene_threshold},min_scene_len={min_scene_len})"

    manifest = create_manifest(
        video_path=video_path,
        duration=duration,
        language=transcript_data.get('language', 'unknown') if transcript_data else 'unknown',
        model_name=model,
        compute_type='int8',  # Default for CPU
        vad_threshold=0.50,
        beam_size=5,
        max_segment_length=30,
        detector_config=detector_config,
        segments=transcript_data.get('segments', []) if transcript_data else [],
        slides=manifest_slides,
        compute_hashes=False,  # Disabled for speed
    )

    manifest_path = slidegeist_dir / "index.json"
    save_manifest(manifest, manifest_path)
    results['manifest'] = manifest_path
    logger.info(f"Saved manifest: {manifest_path}")

    # Summary
    logger.info("=" * 60)
    logger.info("PROCESSING COMPLETE")
    logger.info("=" * 60)
    if not skip_slides:
        logger.info(f"✓ Extracted {len(slide_metadata)} slides")
    if not skip_transcription:
        logger.info(f"✓ Transcribed {len(transcript_data['segments'])} segments")
    logger.info(f"✓ Generated manifest: {manifest_path.name}")
    logger.info(f"✓ All outputs in: {slidegeist_dir}")

    return results


def process_slides_only(
    video_path: Path,
    output_dir: Path,
    scene_threshold: float = DEFAULT_SCENE_THRESHOLD,
    min_scene_len: float = DEFAULT_MIN_SCENE_LEN,
    start_offset: float = DEFAULT_START_OFFSET,
    image_format: str = DEFAULT_IMAGE_FORMAT
) -> dict:
    """Extract only slides from video (no transcription).

    Args:
        video_path: Path to the input video file.
        output_dir: Directory where slide images will be saved.
        scene_threshold: Scene detection threshold (0-1 scale, lower = more sensitive).
        min_scene_len: Minimum scene length in seconds.
        start_offset: Skip first N seconds to avoid setup noise.
        image_format: Output image format (jpg or png).

    Returns:
        Dictionary with 'slides' list and 'manifest' path.
    """
    logger.info("Extracting slides only (no transcription)")
    result = process_video(
        video_path,
        output_dir,
        scene_threshold=scene_threshold,
        min_scene_len=min_scene_len,
        start_offset=start_offset,
        image_format=image_format,
        skip_transcription=True
    )
    return result


def process_transcript_only(
    video_path: Path,
    output_dir: Path,
    model: str = DEFAULT_WHISPER_MODEL,
    device: str = DEFAULT_DEVICE
) -> dict:
    """Extract only transcript from video (no slides).

    Args:
        video_path: Path to the input video file.
        output_dir: Directory where transcript will be saved.
        model: Whisper model size (tiny, base, small, medium, large).
        device: Device for transcription (cpu or cuda).

    Returns:
        Dictionary with 'srt' path and 'manifest' path.
    """
    logger.info("Transcribing audio only (no slides)")
    result = process_video(
        video_path,
        output_dir,
        model=model,
        device=device,
        skip_slides=True
    )
    return result
