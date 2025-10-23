"""Tests for Markdown export functionality."""

from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np

from slidegeist.export import export_slides_json
from slidegeist.transcribe import Segment


class _StubPrimaryExtractor:
    """Stub primary OCR extractor."""
    @property
    def is_available(self) -> bool:
        return True


class _StubOcrPipeline:
    """Deterministic OCR output for testing."""

    def __init__(self) -> None:
        self._primary = _StubPrimaryExtractor()

    def process(
        self,
        image_path: Path,
        transcript_full_text: str,
        transcript_segments: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        refined_text = f"refined {transcript_full_text.strip()}" if transcript_full_text else "refined"
        return {
            "engine": {
                "primary": "stub",
                "primary_version": "1.0",
                "refiner": None,
                "refiner_version": None,
            },
            "raw_text": "raw text",
            "final_text": refined_text,
            "blocks": [],
            "visual_elements": ["arrow"],
            "model_response": "{\"text\": \"stub\", \"visual_elements\": [\"arrow\"]}",
        }


def _make_image(path: Path, color: int) -> None:
    matrix = np.full((10, 20, 3), color, dtype=np.uint8)
    cv2.imwrite(str(path), matrix)


def test_export_slides_manifest_and_payloads(tmp_path: Path) -> None:
    video_path = Path("/fake/video.mp4")

    slides_dir = tmp_path / "slides"
    slides_dir.mkdir()
    img1 = slides_dir / "slide_001.jpg"
    img2 = slides_dir / "slide_002.jpg"
    _make_image(img1, 128)
    _make_image(img2, 64)

    slide_metadata = [
        (1, 0.0, 10.0, img1),
        (2, 10.0, 20.0, img2),
    ]

    transcript_segments: list[Segment] = [
        {"start": 0.0, "end": 5.0, "text": "Welcome to the lecture.", "words": []},
        {"start": 5.0, "end": 10.0, "text": "Today we discuss physics.", "words": []},
        {"start": 10.0, "end": 15.0, "text": "Let's start with Newton.", "words": []},
        {"start": 15.0, "end": 20.0, "text": "And then Einstein.", "words": []},
    ]

    output_file = tmp_path / "index.md"
    ocr_stub = _StubOcrPipeline()

    export_slides_json(
        video_path,
        slide_metadata,
        transcript_segments,
        output_file,
        "tiny",
        ocr_pipeline=ocr_stub,
    )

    assert output_file.exists()

    index_content = output_file.read_text()
    assert "# Lecture Slides" in index_content
    assert "video.mp4" in index_content
    assert "tiny" in index_content
    assert "Slide 1" in index_content
    assert "Slide 2" in index_content

    # Check per-slide markdown (in output root, not slides/)
    slide1_md = tmp_path / "slide_001.md"
    assert slide1_md.exists()
    slide1_content = slide1_md.read_text()
    assert "---" in slide1_content
    assert "id: slide_001" in slide1_content
    assert "index: 1" in slide1_content
    assert "time_start: 0.0" in slide1_content
    assert "time_end: 10.0" in slide1_content
    assert "# Slide 1" in slide1_content
    assert "Welcome to the lecture." in slide1_content
    assert "refined" in slide1_content
    assert "arrow" in slide1_content


def test_export_slides_handles_empty_transcript(tmp_path: Path) -> None:
    video_path = Path("/fake/video.mp4")
    slides_dir = tmp_path / "slides"
    slides_dir.mkdir()
    img1 = slides_dir / "slide_001.jpg"
    _make_image(img1, 255)

    slide_metadata = [
        (1, 0.0, 10.0, img1),
    ]

    transcript_segments: list[Segment] = []

    output_file = tmp_path / "index.md"

    export_slides_json(
        video_path,
        slide_metadata,
        transcript_segments,
        output_file,
        "base",
        ocr_pipeline=_StubOcrPipeline(),
    )

    slide1_md = tmp_path / "slide_001.md"
    assert slide1_md.exists()
    content = slide1_md.read_text()

    # With empty transcript, no transcript section
    assert "## Transcript" not in content or content.count("## Transcript") == 0 or "## Transcript\n\n##" in content
    assert "refined" in content
    assert "arrow" in content


def test_export_slides_empty_metadata(tmp_path: Path) -> None:
    video_path = Path("/fake/video.mp4")
    slide_metadata: list[tuple[int, float, float, Path]] = []
    transcript_segments: list[Segment] = []

    output_file = tmp_path / "index.md"

    export_slides_json(
        video_path,
        slide_metadata,
        transcript_segments,
        output_file,
        "tiny",
        ocr_pipeline=_StubOcrPipeline(),
    )

    assert output_file.exists()
    content = output_file.read_text()
    assert "# Lecture Slides" in content
    assert "video.mp4" in content
    # No slide markdown files in root
    md_files = list(tmp_path.glob("slide_*.md"))
    assert len(md_files) == 0
