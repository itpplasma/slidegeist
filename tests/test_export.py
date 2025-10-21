"""Tests for SRT export functionality."""

import pytest
from pathlib import Path

from slidegeist.export import format_timestamp_srt, export_srt
from slidegeist.transcribe import Segment


def test_format_timestamp_srt():
    """Test SRT timestamp formatting."""
    assert format_timestamp_srt(0.0) == "00:00:00,000"
    assert format_timestamp_srt(5.2) == "00:00:05,200"
    assert format_timestamp_srt(65.5) == "00:01:05,500"
    assert format_timestamp_srt(3665.123) == "01:01:05,123"
    # Note: floating point precision - 125.333 may be 125.332999...
    result = format_timestamp_srt(125.333)
    assert result in ("00:02:05,332", "00:02:05,333")


def test_export_srt(tmp_path: Path):
    """Test SRT file export."""
    segments: list[Segment] = [
        {
            "start": 0.0,
            "end": 5.2,
            "text": "Welcome to the lecture.",
            "words": []
        },
        {
            "start": 5.2,
            "end": 12.8,
            "text": "Today we'll discuss quantum mechanics.",
            "words": []
        }
    ]

    output_file = tmp_path / "test.srt"
    export_srt(segments, output_file)

    assert output_file.exists()

    content = output_file.read_text(encoding='utf-8')

    # Check basic structure
    assert "1\n" in content
    assert "2\n" in content
    assert "00:00:00,000 --> 00:00:05,200" in content
    assert "00:00:05,200 --> 00:00:12,800" in content
    assert "Welcome to the lecture." in content
    assert "Today we'll discuss quantum mechanics." in content


def test_export_srt_empty_text(tmp_path: Path):
    """Test SRT export with empty text segment."""
    segments: list[Segment] = [
        {
            "start": 0.0,
            "end": 2.0,
            "text": "   ",  # Empty/whitespace
            "words": []
        }
    ]

    output_file = tmp_path / "test_empty.srt"
    export_srt(segments, output_file)

    content = output_file.read_text(encoding='utf-8')
    assert "[No speech detected]" in content
