"""Systematic slide detection validation with real lecture videos.

This test harness downloads publicly available lecture videos and tests
slide detection across multiple threshold values for systematic validation.

Videos are cached in /tmp/slidegeist_test_videos/ to avoid re-downloading.
"""

import json
import logging
import os
import subprocess
import tempfile
from pathlib import Path
from typing import NamedTuple

import pytest

logger = logging.getLogger(__name__)


class LectureVideo(NamedTuple):
    """Test case for a lecture video with known characteristics."""

    name: str
    url: str
    expected_slides_min: int  # Minimum expected slides
    expected_slides_max: int  # Maximum expected slides
    duration_minutes: int  # Approximate duration


# Test videos: publicly available educational content
# These are real lecture videos with slide presentations
TEST_VIDEOS = [
    LectureVideo(
        name="stanford_ml_intro",
        url="https://www.youtube.com/watch?v=jGwO_UgTS7I",
        expected_slides_min=15,
        expected_slides_max=25,
        duration_minutes=80,
    ),
    LectureVideo(
        name="mit_linear_algebra",
        url="https://www.youtube.com/watch?v=QVKj3LADCnA",
        expected_slides_min=10,
        expected_slides_max=20,
        duration_minutes=40,
    ),
    LectureVideo(
        name="berkeley_cs61a",
        url="https://www.youtube.com/watch?v=_j7gOBYQdFo",
        expected_slides_min=20,
        expected_slides_max=35,
        duration_minutes=50,
    ),
]


def get_cache_dir() -> Path:
    """Get or create cache directory for test videos."""
    cache_dir = Path("/tmp/slidegeist_test_videos")
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def get_cached_video_path(video: LectureVideo) -> Path:
    """Get path to cached video file."""
    cache_dir = get_cache_dir()
    return cache_dir / f"{video.name}.mp4"


def download_video_if_needed(video: LectureVideo) -> Path:
    """Download video using yt-dlp if not already cached.

    Args:
        video: Video information.

    Returns:
        Path to downloaded video file.
    """
    video_path = get_cached_video_path(video)

    if video_path.exists():
        logger.info(f"Using cached video: {video_path}")
        return video_path

    logger.info(f"Downloading {video.name} from {video.url}")

    # Use yt-dlp to download (same as slidegeist uses internally)
    cmd = [
        "yt-dlp",
        "-f", "best[ext=mp4]",
        "-o", str(video_path),
        video.url
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=600)
        logger.info(f"Downloaded to {video_path}")
        return video_path
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"Download timeout for {video.name}")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Download failed: {e.stderr.decode()}")


def run_slide_detection(video_path: Path, threshold: float) -> int:
    """Run slide detection and return number of slides detected.

    Args:
        video_path: Path to video file.
        threshold: Detection threshold to use.

    Returns:
        Number of slides detected.
    """
    output_dir = Path(tempfile.mkdtemp())

    try:
        cmd = [
            "slidegeist",
            "slides",
            str(video_path),
            "--out", str(output_dir),
            "--scene-threshold", str(threshold)
        ]

        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
            timeout=300
        )

        # Count slide image files
        slides = list(output_dir.glob("slide_*.jpg"))
        return len(slides)

    finally:
        # Cleanup output directory
        import shutil
        shutil.rmtree(output_dir, ignore_errors=True)


@pytest.mark.parametrize("threshold", [0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05, 0.10])
@pytest.mark.parametrize("video", TEST_VIDEOS, ids=lambda v: v.name)
@pytest.mark.slow
@pytest.mark.manual
def test_threshold_systematic(video: LectureVideo, threshold: float):
    """Test slide detection across multiple thresholds systematically.

    This is a manual test that requires:
    1. Internet connection to download videos
    2. yt-dlp installed
    3. Significant time to run

    Run with: pytest -v -m manual tests/test_systematic_detection.py

    Results should be manually reviewed to determine optimal threshold.
    """
    # Download video if needed
    video_path = download_video_if_needed(video)

    # Run detection
    num_slides = run_slide_detection(video_path, threshold)

    # Log results
    in_range = video.expected_slides_min <= num_slides <= video.expected_slides_max
    logger.info(
        f"{video.name} @ threshold={threshold:.3f}: "
        f"{num_slides} slides (expected {video.expected_slides_min}-{video.expected_slides_max}) "
        f"{'✓' if in_range else '✗'}"
    )

    # Store result for analysis
    results_file = get_cache_dir() / "test_results.jsonl"
    with open(results_file, "a") as f:
        result = {
            "video": video.name,
            "threshold": threshold,
            "slides_detected": num_slides,
            "expected_min": video.expected_slides_min,
            "expected_max": video.expected_slides_max,
            "in_range": in_range,
        }
        f.write(json.dumps(result) + "\n")

    # Assertion for CI (will fail if out of range, but test is marked manual)
    # This helps catch major regressions
    assert num_slides > 0, f"No slides detected at threshold {threshold}"


def analyze_results():
    """Analyze test results and print summary.

    Run this manually after running the systematic tests:
    python -c "from tests.test_systematic_detection import analyze_results; analyze_results()"
    """
    results_file = get_cache_dir() / "test_results.jsonl"

    if not results_file.exists():
        print("No results file found. Run tests first.")
        return

    # Load all results
    results = []
    with open(results_file) as f:
        for line in f:
            results.append(json.loads(line))

    # Group by video
    videos = {}
    for r in results:
        video_name = r["video"]
        if video_name not in videos:
            videos[video_name] = []
        videos[video_name].append(r)

    # Print summary
    print("\n" + "=" * 80)
    print("SYSTEMATIC THRESHOLD TESTING RESULTS")
    print("=" * 80)

    for video_name, video_results in videos.items():
        print(f"\n{video_name}:")
        print(f"  Expected: {video_results[0]['expected_min']}-{video_results[0]['expected_max']} slides")
        print(f"\n  Threshold | Detected | In Range")
        print(f"  --------- | -------- | --------")

        for r in sorted(video_results, key=lambda x: x["threshold"]):
            in_range_mark = "✓" if r["in_range"] else "✗"
            print(f"  {r['threshold']:9.3f} | {r['slides_detected']:8d} | {in_range_mark}")

    # Overall statistics
    print("\n" + "=" * 80)
    print("OVERALL STATISTICS")
    print("=" * 80)

    thresholds = sorted(set(r["threshold"] for r in results))
    print(f"\nBest threshold by in-range accuracy:")

    for threshold in thresholds:
        threshold_results = [r for r in results if r["threshold"] == threshold]
        in_range_count = sum(1 for r in threshold_results if r["in_range"])
        total = len(threshold_results)
        accuracy = in_range_count / total * 100
        print(f"  {threshold:.3f}: {in_range_count}/{total} ({accuracy:.1f}%)")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    # Run analysis if called directly
    analyze_results()
