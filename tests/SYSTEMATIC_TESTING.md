# Systematic Slide Detection Testing

This document describes how to perform systematic validation of slide detection thresholds using real lecture videos.

## Overview

The `test_systematic_detection.py` test harness downloads publicly available lecture videos and tests slide detection across multiple threshold values to find the optimal default setting.

## Features

- **Cached Downloads**: Videos are downloaded once to `/tmp/slidegeist_test_videos/` and reused
- **Multiple Thresholds**: Tests thresholds from 0.01 to 0.10 to find optimal balance
- **Real Lecture Videos**: Uses actual educational content from YouTube (MIT, Stanford, Berkeley)
- **Results Analysis**: Generates JSON logs and summary statistics
- **Manual Verification**: Designed for human review of results

## Prerequisites

1. **yt-dlp** must be installed (already required by slidegeist)
2. **Internet connection** for first-time video downloads
3. **Disk space**: ~500MB-1GB for cached videos

## Running Tests

### Run All Systematic Tests

```bash
# Run all threshold/video combinations (takes 30-60 minutes)
pytest -v -m manual tests/test_systematic_detection.py
```

### Run Specific Threshold

```bash
# Test only threshold 0.03
pytest -v -m manual tests/test_systematic_detection.py -k "0.03"
```

### Run Specific Video

```bash
# Test only Stanford ML video
pytest -v -m manual tests/test_systematic_detection.py -k "stanford"
```

### Skip Systematic Tests in CI

```bash
# Run normal tests, skip manual tests
pytest -m "not manual"
```

## Analyzing Results

After running tests, analyze the results:

```bash
# Generate summary report
python -c "from tests.test_systematic_detection import analyze_results; analyze_results()"
```

This will print:
- Per-video threshold performance
- Number of slides detected at each threshold
- Whether detection fell within expected range
- Overall best threshold by accuracy

## Expected Results Format

Results are logged to `/tmp/slidegeist_test_videos/test_results.jsonl`:

```json
{"video": "stanford_ml_intro", "threshold": 0.03, "slides_detected": 18, "expected_min": 15, "expected_max": 25, "in_range": true}
{"video": "stanford_ml_intro", "threshold": 0.05, "slides_detected": 12, "expected_min": 15, "expected_max": 25, "in_range": false}
```

## Test Videos

Current test set includes:

| Video | Duration | Expected Slides | Source |
|-------|----------|-----------------|--------|
| Stanford ML Intro | 80 min | 15-25 | Andrew Ng's ML course |
| MIT Linear Algebra | 40 min | 10-20 | Gilbert Strang's lectures |
| Berkeley CS61A | 50 min | 20-35 | Structure & Interpretation |

## Adding New Test Videos

Edit `test_systematic_detection.py` and add to `TEST_VIDEOS`:

```python
LectureVideo(
    name="my_lecture",
    url="https://www.youtube.com/watch?v=VIDEO_ID",
    expected_slides_min=10,  # Minimum expected
    expected_slides_max=20,  # Maximum expected
    duration_minutes=45,
)
```

### Determining Expected Slide Count

1. Watch the video or skim through it
2. Count slide transitions manually (or use the current tool with low threshold)
3. Set `expected_min` to ~80% of actual count (allow for missed detections)
4. Set `expected_max` to ~120% of actual count (allow for false positives)

## Interpreting Results

### Good Threshold Characteristics

- Detects 80-100% of actual slides (within expected range)
- Few false positives (not detecting every minor change)
- Consistent across different video types
- Balanced precision/recall

### Red Flags

- **Too low**: Many false positives (detecting minor changes)
- **Too high**: Missing obvious slide transitions
- **Inconsistent**: Works well for some videos, poorly for others

## Example Analysis Output

```
================================================================================
SYSTEMATIC THRESHOLD TESTING RESULTS
================================================================================

stanford_ml_intro:
  Expected: 15-25 slides

  Threshold | Detected | In Range
  --------- | -------- | --------
      0.010 |       32 | ✗
      0.015 |       27 | ✗
      0.020 |       22 | ✓
      0.025 |       19 | ✓
      0.030 |       18 | ✓
      0.040 |       14 | ✗
      0.050 |       12 | ✗
      0.100 |        6 | ✗

================================================================================
OVERALL STATISTICS
================================================================================

Best threshold by in-range accuracy:
  0.020: 3/3 (100.0%)
  0.025: 3/3 (100.0%)
  0.030: 2/3 (66.7%)
  0.040: 1/3 (33.3%)
```

## Cleaning Up

Remove cached videos and results:

```bash
rm -rf /tmp/slidegeist_test_videos/
```

## Notes

- Tests are marked `@pytest.mark.manual` to prevent running in CI
- Tests are marked `@pytest.mark.slow` due to video processing time
- First run will download videos (can take 10-20 minutes)
- Subsequent runs use cached videos (much faster)
- Internet connection only needed for initial downloads

## Troubleshooting

### yt-dlp Fails

```bash
# Update yt-dlp
pip install -U yt-dlp
```

### Video Download Timeout

- Check internet connection
- Try downloading video manually to verify accessibility
- Increase timeout in `download_video_if_needed()`

### No Slides Detected

- Check video actually contains slides (not just talking head)
- Try lower threshold values
- Verify video quality is sufficient

## Future Improvements

Potential enhancements:

1. Add more diverse test videos (different subjects, formats, quality)
2. Implement automatic slide counting via OCR or visual similarity
3. Add performance benchmarks (speed vs accuracy tradeoffs)
4. Create ground truth annotations for precise validation
5. Test with different video resolutions and frame rates
