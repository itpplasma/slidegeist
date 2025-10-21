"""Tests for FFmpeg wrapper functionality."""


from slidegeist.ffmpeg import check_ffmpeg_available
from slidegeist.slides import format_timestamp_filename


def test_check_ffmpeg_available():
    """Test FFmpeg availability check."""
    # This will pass if FFmpeg is installed, skip otherwise
    result = check_ffmpeg_available()
    assert isinstance(result, bool)


def test_format_timestamp_filename():
    """Test timestamp filename formatting."""
    assert format_timestamp_filename(0, 125300) == "000000000-000125300"
    assert format_timestamp_filename(125300, 287600) == "000125300-000287600"
    assert format_timestamp_filename(3600000, 7200000) == "003600000-007200000"
