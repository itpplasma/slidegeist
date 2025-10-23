"""Export slide metadata to Markdown files."""

from __future__ import annotations

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
    source_url: str | None = None,
) -> None:
    """Export slides as Markdown files with an index."""
    output_dir = output_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    if ocr_pipeline is None:
        ocr_pipeline = build_default_ocr_pipeline()

    logger.info("Creating slides markdown with %d slides", len(slide_metadata))

    index_lines: List[str] = []
    total_slides = len(slide_metadata)

    for index, (slide_index, t_start, t_end, image_path) in enumerate(slide_metadata):
        slide_id = image_path.stem or f"slide_{slide_index:03d}"
        image_filename = image_path.name

        transcript_text = _collect_transcript_text(transcript_segments, t_start, t_end)

        ocr_available = ocr_pipeline._primary is not None and ocr_pipeline._primary.is_available
        if ocr_available:
            try:
                transcript_payload = _collect_transcript_payload(
                    transcript_segments, t_start, t_end
                )
                ocr_payload = ocr_pipeline.process(
                    image_path=image_path,
                    transcript_full_text=transcript_text,
                    transcript_segments=transcript_payload["segments"],
                )
                ocr_text = ocr_payload.get("final_text", "").strip()
                visual_elements = ocr_payload.get("visual_elements", [])
            except Exception as exc:
                logger.warning("OCR failed for %s: %s", image_path, exc)
                ocr_text = ""
                visual_elements = []
        else:
            logger.warning("Tesseract not available, skipping OCR for %s", image_path)
            ocr_text = ""
            visual_elements = []

        markdown_content = _build_slide_markdown(
            slide_id=slide_id,
            slide_index=slide_index,
            t_start=t_start,
            t_end=t_end,
            image_filename=image_filename,
            transcript_text=transcript_text,
            ocr_text=ocr_text,
            visual_elements=visual_elements,
        )

        per_slide_path = output_dir / f"{slide_id}.md"
        per_slide_path.write_text(markdown_content, encoding="utf-8")

        time_str = f"{_format_timestamp(t_start)}-{_format_timestamp(t_end)}"
        index_lines.append(
            f"{slide_index}. [Slide {slide_index}]({slide_id}.md) • "
            f"![thumb](slides/{image_filename}) • {time_str}"
        )

        logger.debug("Wrote slide %s (%d/%d)", per_slide_path, index + 1, total_slides)

    index_content = _build_index_markdown(
        video_path=video_path,
        source_url=source_url,
        duration=slide_metadata[-1][2] if slide_metadata else 0.0,
        model=model,
        slide_lines=index_lines,
    )

    output_path.write_text(index_content, encoding="utf-8")
    logger.info("Exported slides index to %s", output_path)


def _collect_transcript_text(
    transcript_segments: List[Segment],
    start_time: float,
    end_time: float,
) -> str:
    """Collect transcript text overlapping the slide interval."""
    texts: List[str] = []
    for segment in transcript_segments:
        seg_start = segment["start"]
        seg_end = segment["end"]
        overlap = seg_start < end_time and seg_end > start_time
        if overlap:
            text = segment["text"].strip()
            if text:
                texts.append(text)
    return " ".join(texts)


def _collect_transcript_payload(
    transcript_segments: List[Segment],
    start_time: float,
    end_time: float,
) -> Dict[str, Any]:
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

    return {
        "full_text": full_text,
        "segments": segments,
    }


def _format_timestamp(seconds: float) -> str:
    """Format seconds as MM:SS."""
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{mins:02d}:{secs:02d}"


def _build_slide_markdown(
    slide_id: str,
    slide_index: int,
    t_start: float,
    t_end: float,
    image_filename: str,
    transcript_text: str,
    ocr_text: str,
    visual_elements: List[str],
) -> str:
    """Build Markdown content for a single slide."""
    lines = [
        "---",
        f"id: {slide_id}",
        f"index: {slide_index}",
        f"time_start: {t_start}",
        f"time_end: {t_end}",
        f"image: slides/{image_filename}",
        "---",
        "",
        f"# Slide {slide_index}",
        "",
        f"![Slide Image](slides/{image_filename})",
        "",
    ]

    if transcript_text:
        lines.extend([
            "## Transcript",
            "",
            transcript_text,
            "",
        ])

    if ocr_text or visual_elements:
        lines.extend([
            "## Slide Content",
            "",
        ])
        if ocr_text:
            lines.extend([ocr_text, ""])
        if visual_elements:
            elements_str = ", ".join(visual_elements)
            lines.append(f"**Visual Elements:** {elements_str}")
            lines.append("")

    return "\n".join(lines)


def _build_index_markdown(
    video_path: Path,
    source_url: str | None,
    duration: float,
    model: str,
    slide_lines: List[str],
) -> str:
    """Build the index Markdown file."""
    lines = [
        "# Lecture Slides",
        "",
        f"**Video:** {video_path.name}  ",
    ]

    if source_url:
        lines.append(f"**Source:** {source_url}  ")

    duration_str = _format_timestamp(duration)
    processed_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    lines.extend([
        f"**Duration:** {duration_str}  ",
        f"**Transcription Model:** {model}  ",
        f"**Processed:** {processed_at}",
        "",
        "## Slides",
        "",
    ])

    lines.extend(slide_lines)
    lines.append("")

    return "\n".join(lines)
