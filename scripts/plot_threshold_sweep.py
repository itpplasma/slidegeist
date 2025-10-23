#!/usr/bin/env python3
"""Diagnostic script to plot threshold sweep for a video.

Usage:
    python scripts/plot_threshold_sweep.py /path/to/video.mp4
    python scripts/plot_threshold_sweep.py /path/to/video.mp4 --expected-slides 15
    python scripts/plot_threshold_sweep.py /path/to/video.mp4 --compare-pyscenedetect
"""

import argparse
import logging
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

# Minimal plotting - works without matplotlib
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not installed - will only show text output")
    print("Install with: pip install matplotlib")

# Optional PySceneDetect
try:
    from scenedetect import detect, ContentDetector, SceneManager, open_video
    from scenedetect.detectors import HistogramDetector, HashDetector
    from scenedetect.stats_manager import StatsManager
    HAS_PYSCENEDETECT = True
except ImportError:
    HAS_PYSCENEDETECT = False


def compute_frame_diffs(video_path: Path) -> tuple[list[tuple[int, float]], float]:
    """Compute all frame differences for a video.

    Returns:
        (frame_diffs, working_fps) where frame_diffs is list of (frame_num, diff_value)
    """
    from slidegeist.pixel_diff_detector import (
        cv2, np, subprocess, tempfile, os
    )

    # Simplified version of preprocessing from detect_slides_adaptive
    cap = cv2.VideoCapture(str(video_path))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    if fps <= 0:
        fps = 30.0

    # Preprocess if needed
    max_resolution = 360
    target_fps = 5.0
    working_video = video_path
    temp_file = None
    needs_processing = height > max_resolution or fps > target_fps
    working_fps = fps

    if needs_processing:
        scale = max_resolution / height if height > max_resolution else 1.0
        new_width = int(width * scale)
        new_height = int(height * scale)
        new_width = (new_width // 2) * 2
        new_height = (new_height // 2) * 2

        filters = []
        if scale < 1.0:
            filters.append(f'scale={new_width}:{new_height}')
        if fps > target_fps:
            fps_ratio = int(round(fps / target_fps))
            actual_fps = fps / fps_ratio
            filters.append(f'fps=fps={actual_fps}')
            working_fps = actual_fps

        filter_str = ','.join(filters)

        temp_file = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
        temp_file.close()

        print(f"Preprocessing video: {width}x{height}@{fps:.1f}fps -> {new_width}x{new_height}@{working_fps:.1f}fps")
        cmd = [
            'ffmpeg', '-i', str(video_path),
            '-vf', filter_str,
            '-c:v', 'libx264', '-preset', 'ultrafast', '-crf', '28',
            '-y', temp_file.name
        ]
        subprocess.run(cmd, capture_output=True, text=True, check=True)
        working_video = Path(temp_file.name)

    # Compute frame differences
    cap = cv2.VideoCapture(str(working_video))
    working_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    working_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    image_size = working_width * working_height

    start_offset = 3.0
    sample_interval = 1.0
    start_frame = int(start_offset * working_fps)
    frame_interval = max(1, int(round(sample_interval * working_fps)))

    frame_diffs = []
    prev_frame_binary = None
    frame_num = start_frame

    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        print(f"Computing frame differences...")

        while frame_num < total_frames:
            ret, frame = cap.read()
            if not ret:
                break

            if (frame_num - start_frame) % frame_interval != 0:
                frame_num += 1
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            if prev_frame_binary is not None:
                diff = np.abs(binary.astype(np.int16) - prev_frame_binary.astype(np.int16))
                non_zero_count = np.count_nonzero(diff)
                normalized_diff = non_zero_count / image_size
                frame_diffs.append((frame_num, normalized_diff))

            prev_frame_binary = binary
            frame_num += 1

    finally:
        cap.release()
        if temp_file is not None:
            try:
                os.unlink(temp_file.name)
            except Exception:
                pass

    print(f"Computed {len(frame_diffs)} frame differences")
    return frame_diffs, working_fps


def sweep_thresholds(
    frame_diffs: list[tuple[int, float]],
    working_fps: float,
    threshold_range: tuple[float, float] = (0.01, 0.10),
    threshold_step: float = 0.001,
    min_scene_len: float = 2.0,
    start_offset: float = 3.0
) -> tuple[np.ndarray, np.ndarray]:
    """Sweep thresholds and count slides at each.

    Returns:
        (thresholds, slide_counts)
    """
    thresholds = np.arange(threshold_range[0], threshold_range[1] + threshold_step, threshold_step)
    slide_counts = []

    start_frame = int(start_offset * working_fps)
    min_frames_between = max(1, int(round(min_scene_len * working_fps)))

    print(f"Sweeping {len(thresholds)} thresholds from {threshold_range[0]:.3f} to {threshold_range[1]:.3f}...")

    for thresh in thresholds:
        count = 0
        last_change_frame = start_frame

        for frame_num, diff_val in frame_diffs:
            if diff_val >= thresh:
                if frame_num - last_change_frame >= min_frames_between:
                    count += 1
                    last_change_frame = frame_num

        slide_counts.append(count)

    return thresholds, np.array(slide_counts)


def compute_pyscenedetect_scores(video_path: Path, detector_name: str) -> tuple[list[float], float, str]:
    """Compute PySceneDetect detector scores using stats file.

    Args:
        video_path: Path to video
        detector_name: 'content', 'histogram', or 'hash'

    Returns:
        (score_vals, fps, metric_key) where metric_key is the CSV column name
    """
    if not HAS_PYSCENEDETECT:
        raise RuntimeError("PySceneDetect not installed")

    # Create detector and determine metric key prefix
    # Note: actual CSV column names have parameters in brackets
    if detector_name == 'content':
        detector = ContentDetector()
        metric_prefix = 'content_val'
        print("\nRunning PySceneDetect ContentDetector (HSV)...")
    elif detector_name == 'histogram':
        detector = HistogramDetector()
        metric_prefix = 'hist_diff'
        print("\nRunning PySceneDetect HistogramDetector (YUV)...")
    elif detector_name == 'hash':
        detector = HashDetector()
        metric_prefix = 'hash_dist'
        print("\nRunning PySceneDetect HashDetector (perceptual hash)...")
    else:
        raise ValueError(f"Unknown detector: {detector_name}")

    # Create temp stats file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        stats_file = Path(f.name)

    video = None
    try:
        # Process video with detector and save stats
        video = open_video(str(video_path))
        fps = video.frame_rate

        stats_manager = StatsManager()
        scene_manager = SceneManager(stats_manager)
        scene_manager.add_detector(detector)

        # Process all frames
        scene_manager.detect_scenes(video)

        # Save stats to file
        stats_manager.save_to_csv(str(stats_file))

        # Read stats file - find column matching metric prefix
        import csv
        score_vals = []
        with open(stats_file, 'r') as f:
            reader = csv.DictReader(f)
            # Find actual column name (may have parameters in brackets)
            if reader.fieldnames:
                metric_key = None
                for field in reader.fieldnames:
                    if field.startswith(metric_prefix):
                        metric_key = field
                        break

                if not metric_key:
                    raise RuntimeError(f"No column found starting with {metric_prefix}")

                for row in reader:
                    if metric_key in row and row[metric_key]:
                        score_vals.append(float(row[metric_key]))

        print(f"PySceneDetect {detector_name} computed {len(score_vals)} frame scores (column: {metric_key})")
        return score_vals, fps, metric_key

    finally:
        # Clean up video
        if video is not None:
            try:
                del video
            except Exception:
                pass

        # Clean up temp file
        try:
            stats_file.unlink()
        except Exception:
            pass


def sweep_pyscenedetect_thresholds(
    content_vals: list[float],
    fps: float,
    threshold_range: tuple[float, float] = (1.0, 25.0),
    threshold_step: float = 0.5,
    min_scene_len: float = 2.0,
    start_offset: float = 3.0
) -> tuple[np.ndarray, np.ndarray]:
    """Sweep PySceneDetect thresholds on pre-computed scores.

    Returns:
        (thresholds, scene_counts)
    """
    thresholds = np.arange(threshold_range[0], threshold_range[1] + threshold_step, threshold_step)
    scene_counts = []

    start_frame = int(start_offset * fps)
    min_frames_between = max(1, int(round(min_scene_len * fps)))

    print(f"Sweeping {len(thresholds)} PySceneDetect thresholds from {threshold_range[0]:.1f} to {threshold_range[1]:.1f}...")

    for thresh in thresholds:
        count = 0
        last_cut_frame = start_frame

        for frame_num, score in enumerate(content_vals, start=start_frame):
            if score >= thresh:
                if frame_num - last_cut_frame >= min_frames_between:
                    count += 1
                    last_cut_frame = frame_num

        scene_counts.append(count)

    return thresholds, np.array(scene_counts)


def plot_sweep(
    thresholds: np.ndarray,
    slide_counts: np.ndarray,
    expected_slides: int | None = None,
    video_name: str = "Video",
    pyscene_results: list[tuple[str, np.ndarray, np.ndarray]] | None = None
):
    """Plot threshold sweep results.

    Args:
        thresholds: slidegeist thresholds
        slide_counts: slidegeist slide counts
        expected_slides: expected slide count
        video_name: video filename
        pyscene_results: list of (detector_name, thresholds, counts) tuples
    """
    if not HAS_MATPLOTLIB:
        print("\nCannot plot without matplotlib")
        return

    fig, ax = plt.subplots(figsize=(14, 7))

    # Plot our method (slidegeist) - normalize to 0-1 like others
    thresh_min_sg = thresholds.min()
    thresh_max_sg = thresholds.max()
    normalized_thresholds_sg = (thresholds - thresh_min_sg) / (thresh_max_sg - thresh_min_sg)

    ax.plot(normalized_thresholds_sg, slide_counts, 'b-', linewidth=2.5,
            label=f'slidegeist (binary pixel diff, {thresh_min_sg:.2f}-{thresh_max_sg:.2f})', zorder=10)
    ax.scatter(normalized_thresholds_sg, slide_counts, c='blue', s=30, alpha=0.6, zorder=10)

    # Plot PySceneDetect methods if available - normalize their thresholds to 0-1
    colors = ['red', 'orange', 'purple']
    if pyscene_results:
        for idx, (detector_name, pyscene_thresholds, pyscene_counts) in enumerate(pyscene_results):
            color = colors[idx % len(colors)]

            # Normalize thresholds to 0-1 range
            thresh_min = pyscene_thresholds.min()
            thresh_max = pyscene_thresholds.max()
            normalized_thresholds = (pyscene_thresholds - thresh_min) / (thresh_max - thresh_min)

            label_map = {
                'content': f'PySceneDetect ContentDetector (HSV, {thresh_min:.1f}-{thresh_max:.1f})',
                'histogram': f'PySceneDetect HistogramDetector (YUV, {thresh_min:.1f}-{thresh_max:.1f})',
                'hash': f'PySceneDetect HashDetector (pHash, {thresh_min:.2f}-{thresh_max:.2f})'
            }
            label = label_map.get(detector_name, detector_name)

            ax.plot(normalized_thresholds, pyscene_counts, color=color, linestyle='-',
                    linewidth=2, label=label, alpha=0.8)
            ax.scatter(normalized_thresholds, pyscene_counts, c=color, s=20, alpha=0.5)

    # Mark expected slide count if provided
    if expected_slides:
        ax.axhline(y=expected_slides, color='g', linestyle='--', linewidth=2,
                   label=f'Expected: {expected_slides} slides')
        # Shade acceptable range (±20%)
        lower = expected_slides * 0.8
        upper = expected_slides * 1.2
        ax.axhspan(lower, upper, alpha=0.1, color='green', label='±20% range')

    ax.set_xlabel('Normalized Threshold (0=min sensitivity, 1=max sensitivity)', fontsize=12)
    ax.set_ylabel('Number of Slides (log scale)', fontsize=12)
    ax.set_yscale('log')
    ax.set_title(f'Threshold Sweep Comparison: {video_name}', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, which='both')
    ax.legend(loc='upper right', fontsize=9)

    plt.tight_layout()

    # Save plot
    output_path = Path('threshold_sweep.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: {output_path}")

    # Try to display
    try:
        plt.show()
    except:
        print("(Cannot display plot - saved to file)")


def print_text_summary(
    thresholds: np.ndarray,
    slide_counts: np.ndarray,
    expected_slides: int | None = None
):
    """Print text summary of sweep results."""
    print("\n" + "="*80)
    print("THRESHOLD SWEEP RESULTS")
    print("="*80)
    print(f"\n{'Threshold':>10} | {'Slides':>6} | {'Change':>7} | Status")
    print("-"*80)

    # Show every 5th point to avoid too much output
    step = max(1, len(thresholds) // 20)

    for i in range(0, len(thresholds), step):
        thresh = thresholds[i]
        count = slide_counts[i]
        change = "" if i == 0 else f"{count - slide_counts[i-step]:+3d}"

        if expected_slides:
            if expected_slides * 0.8 <= count <= expected_slides * 1.2:
                status = "✓ good"
            elif count < expected_slides:
                status = f"✗ under ({expected_slides - count} missing)"
            else:
                status = f"✗ over (+{count - expected_slides} extra)"
        else:
            status = ""

        print(f"{thresh:10.3f} | {count:6d} | {change:>7} | {status}")

    print("\n" + "="*80)

    # Find best thresholds
    if expected_slides:
        # Find threshold closest to expected count
        diffs = np.abs(slide_counts - expected_slides)
        best_idx = np.argmin(diffs)

        print(f"\nClosest to {expected_slides} slides:")
        print(f"  Threshold: {thresholds[best_idx]:.3f}")
        print(f"  Slides: {slide_counts[best_idx]}")
        print(f"  Error: {slide_counts[best_idx] - expected_slides:+d} slides")


def main():
    parser = argparse.ArgumentParser(
        description="Plot threshold sweep diagnostic for slide detection"
    )
    parser.add_argument("video", type=Path, help="Path to video file")
    parser.add_argument(
        "--expected-slides", "-e", type=int,
        help="Expected number of slides (for comparison)"
    )
    parser.add_argument(
        "--threshold-min", type=float, default=0.01,
        help="Minimum threshold for slidegeist (default: 0.01)"
    )
    parser.add_argument(
        "--threshold-max", type=float, default=0.10,
        help="Maximum threshold for slidegeist (default: 0.10)"
    )
    parser.add_argument(
        "--threshold-step", type=float, default=0.001,
        help="Threshold step size for slidegeist (default: 0.001)"
    )
    parser.add_argument(
        "--compare-pyscenedetect", action="store_true",
        help="Also run PySceneDetect ContentDetector for comparison"
    )
    parser.add_argument(
        "--pyscene-threshold-min", type=float, default=1.0,
        help="Minimum threshold for PySceneDetect (default: 1.0)"
    )
    parser.add_argument(
        "--pyscene-threshold-max", type=float, default=25.0,
        help="Maximum threshold for PySceneDetect (default: 25.0)"
    )
    parser.add_argument(
        "--pyscene-threshold-step", type=float, default=0.5,
        help="Threshold step size for PySceneDetect (default: 0.5)"
    )

    args = parser.parse_args()

    if not args.video.exists():
        print(f"Error: Video not found: {args.video}")
        sys.exit(1)

    # Compute frame differences for slidegeist
    frame_diffs, working_fps = compute_frame_diffs(args.video)

    # Sweep thresholds for slidegeist
    global threshold_range
    threshold_range = (args.threshold_min, args.threshold_max)
    thresholds, slide_counts = sweep_thresholds(
        frame_diffs,
        working_fps,
        threshold_range=threshold_range,
        threshold_step=args.threshold_step
    )

    # Print text summary for slidegeist
    print_text_summary(thresholds, slide_counts, args.expected_slides)

    # Optionally run PySceneDetect comparison
    pyscene_results = []
    if args.compare_pyscenedetect:
        if not HAS_PYSCENEDETECT:
            print("\nWarning: PySceneDetect not installed. Install with: pip install scenedetect[opencv]")
        else:
            # Define detector configs: (name, threshold_range, step)
            detector_configs = [
                ('content', (args.pyscene_threshold_min, args.pyscene_threshold_max), args.pyscene_threshold_step),
                ('hash', (0.10, 0.50), 0.01),
            ]

            for detector_name, threshold_range, threshold_step in detector_configs:
                try:
                    score_vals, pyscene_fps, metric_key = compute_pyscenedetect_scores(args.video, detector_name)
                    pyscene_thresholds, pyscene_counts = sweep_pyscenedetect_thresholds(
                        score_vals,
                        pyscene_fps,
                        threshold_range=threshold_range,
                        threshold_step=threshold_step
                    )
                    pyscene_results.append((detector_name, pyscene_thresholds, pyscene_counts))
                    print(f"PySceneDetect {detector_name} sweep complete: {len(pyscene_thresholds)} thresholds tested")
                except Exception as e:
                    print(f"\nWarning: PySceneDetect {detector_name} comparison failed: {e}")

    # Plot if matplotlib available
    if HAS_MATPLOTLIB:
        plot_sweep(
            thresholds,
            slide_counts,
            args.expected_slides,
            args.video.name,
            pyscene_results if pyscene_results else None
        )
    else:
        print("\nInstall matplotlib to see plot: pip install matplotlib")


if __name__ == "__main__":
    main()
