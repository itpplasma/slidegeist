"""Tests for FFmpeg wrapper functionality."""


from slidegeist.ffmpeg import check_ffmpeg_available
from slidegeist.slides import format_slide_filename, format_timestamp_hhmmss


def test_check_ffmpeg_available():
    """Test FFmpeg availability check."""
    # This will pass if FFmpeg is installed, skip otherwise
    result = check_ffmpeg_available()
    assert isinstance(result, bool)


def test_format_slide_filename():
    """Test slide filename formatting with HH:MM:SS timestamps."""
    # Test basic formatting
    assert format_slide_filename(0, 100, 0.0, 125.3) == "slide_000_00:00:00-00:02:05"
    assert format_slide_filename(42, 100, 125.3, 287.6) == "slide_042_00:02:05-00:04:47"

    # Test padding based on total slides
    assert format_slide_filename(0, 10, 0.0, 10.0) == "slide_000_00:00:00-00:00:10"
    assert format_slide_filename(0, 1000, 0.0, 10.0) == "slide_000_00:00:00-00:00:10"  # max(3, len("999")) = 3
    assert format_slide_filename(999, 1000, 0.0, 10.0) == "slide_999_00:00:00-00:00:10"


def test_format_timestamp_hhmmss():
    """Test HH:MM:SS timestamp formatting."""
    assert format_timestamp_hhmmss(0.0) == "00:00:00"
    assert format_timestamp_hhmmss(125.3) == "00:02:05"
    assert format_timestamp_hhmmss(3661.5) == "01:01:01"
