"""Tests for JSON export functionality."""

import json
from pathlib import Path

from slidegeist.export import export_slides_json
from slidegeist.transcribe import Segment


def test_export_slides_json_basic(tmp_path: Path) -> None:
    """Test basic slides.json export."""
    video_path = Path("/fake/video.mp4")

    # Create fake image files
    slides_dir = tmp_path / "slides"
    slides_dir.mkdir()
    img1 = slides_dir / "slide_001.jpg"
    img2 = slides_dir / "slide_002.jpg"
    img1.touch()
    img2.touch()

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
    export_slides_json(video_path, slide_metadata, transcript_segments, output_file, "tiny")

    assert output_file.exists()

    with output_file.open() as f:
        data = json.load(f)

    # Check metadata
    assert data["metadata"]["video_file"] == "video.mp4"
    assert data["metadata"]["duration_seconds"] == 20
    assert data["metadata"]["model"] == "tiny"
    assert "processed_at" in data["metadata"]

    # Check slides
    assert len(data["slides"]) == 2

    # First slide
    assert data["slides"][0]["slide_number"] == 1
    assert data["slides"][0]["time_start"] == 0
    assert data["slides"][0]["time_end"] == 10
    assert data["slides"][0]["image_path"] == "slides/slide_001.jpg"
    assert "Welcome to the lecture." in data["slides"][0]["transcript"]
    assert "Today we discuss physics." in data["slides"][0]["transcript"]

    # Second slide
    assert data["slides"][1]["slide_number"] == 2
    assert data["slides"][1]["time_start"] == 10
    assert data["slides"][1]["time_end"] == 20
    assert data["slides"][1]["image_path"] == "slides/slide_002.jpg"
    assert "Let's start with Newton." in data["slides"][1]["transcript"]
    assert "And then Einstein." in data["slides"][1]["transcript"]


def test_export_slides_json_empty_transcript(tmp_path: Path) -> None:
    """Test export with slide that has no transcript."""
    video_path = Path("/fake/video.mp4")

    slides_dir = tmp_path / "slides"
    slides_dir.mkdir()
    img1 = slides_dir / "slide_001.jpg"
    img1.touch()

    slide_metadata = [
        (1, 0.0, 10.0, img1),
    ]

    transcript_segments: list[Segment] = []

    output_file = tmp_path / "slides.json"
    export_slides_json(video_path, slide_metadata, transcript_segments, output_file, "base")

    assert output_file.exists()

    with output_file.open() as f:
        data = json.load(f)

    assert len(data["slides"]) == 1
    assert data["slides"][0]["transcript"] == ""


def test_export_slides_json_whitespace_text(tmp_path: Path) -> None:
    """Test export filters out whitespace-only transcript text."""
    video_path = Path("/fake/video.mp4")

    slides_dir = tmp_path / "slides"
    slides_dir.mkdir()
    img1 = slides_dir / "slide_001.jpg"
    img1.touch()

    slide_metadata = [
        (1, 0.0, 10.0, img1),
    ]

    transcript_segments: list[Segment] = [
        {"start": 0.0, "end": 5.0, "text": "   ", "words": []},
        {"start": 5.0, "end": 10.0, "text": "Valid text.", "words": []},
    ]

    output_file = tmp_path / "slides.json"
    export_slides_json(video_path, slide_metadata, transcript_segments, output_file, "tiny")

    with output_file.open() as f:
        data = json.load(f)

    # Should only include "Valid text." not the whitespace
    assert data["slides"][0]["transcript"] == "Valid text."


def test_export_slides_json_empty_metadata(tmp_path: Path) -> None:
    """Test export with no slides."""
    video_path = Path("/fake/video.mp4")
    slide_metadata: list[tuple[int, float, float, Path]] = []
    transcript_segments: list[Segment] = []

    output_file = tmp_path / "slides.json"
    export_slides_json(video_path, slide_metadata, transcript_segments, output_file, "tiny")

    assert output_file.exists()

    with output_file.open() as f:
        data = json.load(f)

    assert data["metadata"]["duration_seconds"] == 0
    assert len(data["slides"]) == 0
