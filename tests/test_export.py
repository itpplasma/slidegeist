"""Tests for JSON export functionality."""

import json
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np

from slidegeist.export import export_slides_json
from slidegeist.transcribe import Segment


class _StubOcrPipeline:
    """Deterministic OCR output for testing."""

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

    output_file = tmp_path / "slides.json"
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

    with output_file.open() as handle:
        manifest = json.load(handle)

    assert manifest["metadata"]["video_file"] == "video.mp4"
    assert manifest["metadata"]["duration_seconds"] == 20.0
    assert manifest["metadata"]["model"] == "tiny"
    assert len(manifest["slides"]) == 2

    slide_entry = manifest["slides"][0]
    assert slide_entry["id"] == img1.stem
    assert slide_entry["image_path"] == "slides/slide_001.jpg"
    assert slide_entry["time_start"] == 0.0
    assert slide_entry["time_end"] == 10.0

    per_slide_path = tmp_path / slide_entry["json_path"]
    assert per_slide_path.exists()

    with per_slide_path.open() as handle:
        slide_payload = json.load(handle)

    assert slide_payload["schema_version"] == "1.0"
    assert slide_payload["image"]["width"] == 20
    assert slide_payload["image"]["height"] == 10
    assert "Welcome to the lecture." in slide_payload["transcript"]["full_text"]
    assert slide_payload["ocr"]["final_text"].startswith("refined")
    assert slide_payload["ocr"]["visual_elements"] == ["arrow"]


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

    output_file = tmp_path / "slides.json"

    export_slides_json(
        video_path,
        slide_metadata,
        transcript_segments,
        output_file,
        "base",
        ocr_pipeline=_StubOcrPipeline(),
    )

    with output_file.open() as handle:
        manifest = json.load(handle)

    assert len(manifest["slides"]) == 1
    slide_entry = manifest["slides"][0]
    per_slide = tmp_path / slide_entry["json_path"]
    with per_slide.open() as handle:
        slide_payload = json.load(handle)

    assert slide_payload["transcript"]["full_text"] == ""
    assert slide_payload["ocr"]["final_text"] == "refined"
    assert slide_payload["ocr"]["visual_elements"] == ["arrow"]


def test_export_slides_empty_metadata(tmp_path: Path) -> None:
    video_path = Path("/fake/video.mp4")
    slide_metadata: list[tuple[int, float, float, Path]] = []
    transcript_segments: list[Segment] = []

    output_file = tmp_path / "slides.json"

    export_slides_json(
        video_path,
        slide_metadata,
        transcript_segments,
        output_file,
        "tiny",
        ocr_pipeline=_StubOcrPipeline(),
    )

    with output_file.open() as handle:
        manifest = json.load(handle)

    assert manifest["metadata"]["duration_seconds"] == 0
    assert len(manifest["slides"]) == 0
