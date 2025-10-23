"""Export slide metadata to manifest plus per-slide payloads."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2

from slidegeist.ocr import OcrPipeline, build_default_ocr_pipeline
from slidegeist.transcribe import Segment

logger = logging.getLogger(__name__)


def export_slides_json(
    video_path: Path,
    slide_metadata: list[tuple[int, float, float, Path]],
    transcript_segments: list[Segment],
    output_path: Path,
    model: str,
    ocr_pipeline: OcrPipeline | None = None,
) -> None:
    """Export slides manifest and per-slide JSON files with OCR/transcript payloads."""
    output_dir = output_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    per_slide_dir = output_dir / "slides_meta"
    per_slide_dir.mkdir(parents=True, exist_ok=True)

    if ocr_pipeline is None:
        ocr_pipeline = build_default_ocr_pipeline()

    logger.info("Creating slides manifest with %d slides", len(slide_metadata))

    manifest_slides: List[Dict[str, Any]] = []
    total_slides = len(slide_metadata)

    for index, (slide_index, t_start, t_end, image_path) in enumerate(slide_metadata):
        slide_id = image_path.stem or f"slide_{slide_index:03d}"
        image_relative = image_path.relative_to(output_dir)

        transcript_payload, transcript_text = _collect_transcript_payload(
            transcript_segments,
            t_start,
            t_end,
        )

        ocr_payload = ocr_pipeline.process(
            image_path=image_path,
            transcript_full_text=transcript_text,
            transcript_segments=transcript_payload["segments"],
        )

        width_height = _read_image_size(image_path)
        slide_json = {
            "schema_version": "1.0",
            "id": slide_id,
            "index": slide_index,
            "time": {
                "start": t_start,
                "end": t_end,
            },
            "image": {
                "path": str(image_relative),
                "width": width_height[0],
                "height": width_height[1],
            },
            "transcript": transcript_payload,
            "ocr": ocr_payload,
        }

        per_slide_path = per_slide_dir / f"{slide_id}.json"
        with per_slide_path.open("w", encoding="utf-8") as handle:
            json.dump(slide_json, handle, indent=2, ensure_ascii=False)

        manifest_slides.append(
            {
                "id": slide_id,
                "index": slide_index,
                "json_path": str(per_slide_path.relative_to(output_dir)),
                "image_path": str(image_relative),
                "time_start": t_start,
                "time_end": t_end,
            }
        )

        logger.debug("Wrote slide payload %s (%d/%d)", per_slide_path, index + 1, total_slides)

    manifest = {
        "version": "1.0",
        "metadata": {
            "video_file": video_path.name,
            "duration_seconds": slide_metadata[-1][2] if slide_metadata else 0.0,
            "processed_at": datetime.now(timezone.utc).isoformat(),
            "model": model,
        },
        "slides": manifest_slides,
    }

    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)

    logger.info("Exported slide manifest to %s", output_path)


def _collect_transcript_payload(
    transcript_segments: List[Segment],
    start_time: float,
    end_time: float,
) -> tuple[Dict[str, Any], str]:
    """Filter transcript segments to those overlapping the slide interval."""
    segments: List[Dict[str, Any]] = []

    for segment in transcript_segments:
        seg_start = segment["start"]
        seg_end = segment["end"]
        overlap = seg_start < end_time and seg_end > start_time

        if not overlap:
            continue

        text = segment["text"].strip()
        if not text:
            continue

        segments.append(
            {
                "start": seg_start,
                "end": seg_end,
                "text": text,
                "words": segment.get("words", []),
            }
        )

    full_text = " ".join(item["text"] for item in segments)

    payload = {
        "full_text": full_text,
        "segments": segments,
    }
    return payload, full_text


def _read_image_size(image_path: Path) -> tuple[Optional[int], Optional[int]]:
    """Return width/height for image; tolerate missing files."""
    if not image_path.exists():
        return (None, None)

    image = cv2.imread(str(image_path))
    if image is None:
        return (None, None)

    height, width = image.shape[:2]
    return (int(width), int(height))
