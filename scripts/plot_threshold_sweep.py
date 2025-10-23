#!/usr/bin/env python3
"""Diagnostic script to plot threshold sweep for a video.

Usage:
    python scripts/plot_threshold_sweep.py /path/to/video.mp4
    python scripts/plot_threshold_sweep.py /path/to/video.mp4 --expected-slides 15
"""

import argparse
import logging
import sys
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


def plot_sweep(
    thresholds: np.ndarray,
    slide_counts: np.ndarray,
    expected_slides: int | None = None,
    video_name: str = "Video"
):
    """Plot threshold sweep results."""
    if not HAS_MATPLOTLIB:
        print("\nCannot plot without matplotlib")
        return

    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot slide count vs threshold
    ax.plot(thresholds, slide_counts, 'b-', linewidth=2, label='Detected slides')
    ax.scatter(thresholds, slide_counts, c='blue', s=20, alpha=0.5)

    # Mark expected slide count if provided
    if expected_slides:
        ax.axhline(y=expected_slides, color='g', linestyle='--', linewidth=2,
                   label=f'Expected: {expected_slides} slides')
        # Shade acceptable range (±20%)
        lower = expected_slides * 0.8
        upper = expected_slides * 1.2
        ax.axhspan(lower, upper, alpha=0.1, color='green', label='±20% range')

    ax.set_xlabel('Threshold', fontsize=12)
    ax.set_ylabel('Number of Slides', fontsize=12)
    ax.set_title(f'Threshold Sweep: {video_name}', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)

    # Add vertical lines at common thresholds
    for t, label in [(0.01, '0.01'), (0.02, '0.02'), (0.03, '0.03'), (0.05, '0.05'), (0.10, '0.10')]:
        if threshold_range[0] <= t <= threshold_range[1]:
            ax.axvline(x=t, color='gray', linestyle=':', alpha=0.5)
            idx = np.argmin(np.abs(thresholds - t))
            ax.annotate(f'{label}\n({slide_counts[idx]} slides)',
                       xy=(t, slide_counts[idx]),
                       xytext=(5, 5),
                       textcoords='offset points',
                       fontsize=8,
                       alpha=0.7)

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
        help="Minimum threshold (default: 0.01)"
    )
    parser.add_argument(
        "--threshold-max", type=float, default=0.10,
        help="Maximum threshold (default: 0.10)"
    )
    parser.add_argument(
        "--threshold-step", type=float, default=0.001,
        help="Threshold step size (default: 0.001)"
    )

    args = parser.parse_args()

    if not args.video.exists():
        print(f"Error: Video not found: {args.video}")
        sys.exit(1)

    # Compute frame differences
    frame_diffs, working_fps = compute_frame_diffs(args.video)

    # Sweep thresholds
    global threshold_range
    threshold_range = (args.threshold_min, args.threshold_max)
    thresholds, slide_counts = sweep_thresholds(
        frame_diffs,
        working_fps,
        threshold_range=threshold_range,
        threshold_step=args.threshold_step
    )

    # Print text summary
    print_text_summary(thresholds, slide_counts, args.expected_slides)

    # Plot if matplotlib available
    if HAS_MATPLOTLIB:
        plot_sweep(thresholds, slide_counts, args.expected_slides, args.video.name)
    else:
        print("\nInstall matplotlib to see plot: pip install matplotlib")


if __name__ == "__main__":
    main()
