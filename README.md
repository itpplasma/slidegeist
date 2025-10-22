# Slidegeist

Extract slides and timestamped transcripts from lecture videos with minimal dependencies.

## Features

- **Scene detection** using global pixel difference (research-based method optimized for lecture videos)
- **Automatic slide extraction** with timestamp ranges in filenames
- **Audio transcription** with Whisper large-v3 model (highest quality)
- **MLX acceleration** on Apple Silicon Macs for 2-3x faster transcription
- **JSON export** with slides grouped by their transcripts

## Requirements

- **Python ≥ 3.10**
- **FFmpeg** (must be installed separately and available in PATH)

### Installing FFmpeg

**macOS:**
```bash
brew install ffmpeg
```

**Ubuntu/Debian:**
```bash
sudo apt-get install ffmpeg
```

**Windows:**
Download from [ffmpeg.org](https://ffmpeg.org/download.html) or use:
```bash
winget install ffmpeg
```

## Installation

```bash
# Clone the repository
git clone https://github.com/itpplasma/slidegeist.git
cd slidegeist

# Install with pip
pip install -e .

# On Apple Silicon Macs, install with MLX for 2-3x faster transcription
pip install -e ".[mlx]"

# Or with development dependencies
pip install -e ".[dev]"
```

## Quick Start

Process a lecture video to extract slides and transcript:

```bash
slidegeist lecture.mp4 --out output/
```

This creates:
```
output/
├── slide_000_00:00:00-00:02:05.jpg  # Slide from 0:00 to 2:05
├── slide_001_00:02:05-00:04:47.jpg  # Slide from 2:05 to 4:47
├── slide_002_00:04:47-00:07:30.jpg  # Slide from 4:47 to 7:30
└── slides.json                      # Slides with transcripts and metadata
```

## Usage

### Full Processing

```bash
# Basic usage (auto-detects MLX on Apple Silicon, uses large-v3 model)
slidegeist video.mp4

# Specify output directory
slidegeist video.mp4 --out my-output/

# Use GPU explicitly (NVIDIA)
slidegeist video.mp4 --device cuda

# Use smaller/faster model
slidegeist video.mp4 --model base

# Adjust scene detection sensitivity (0.0-1.0, default 0.10)
# Lower values detect more subtle changes, higher values only major transitions
slidegeist video.mp4 --scene-threshold 0.05

# Explicit process command (same as default)
slidegeist process video.mp4
```

### Individual Operations

```bash
# Extract only slides (no transcription)
slidegeist slides video.mp4

# Extract only transcript (no slides)
slidegeist transcribe video.mp4
```

## CLI Options

```
slidegeist <video> [options]
slidegeist {process,slides,transcribe} <video> [options]

Options:
  --out DIR              Output directory (default: video filename)
  --scene-threshold NUM  Scene detection sensitivity 0.0-1.0 (default: 0.10)
  --model NAME          Whisper model: tiny, base, small, medium, large, large-v2, large-v3
                        (default: large-v3)
  --device NAME         Device: cpu, cuda, or auto (default: auto)
                        auto = MLX on Apple Silicon if available, else CPU
  --format FMT          Image format: jpg or png (default: jpg)
  -v, --verbose         Enable verbose logging
```

## Output Format

### Slide Filenames

Slides are named with their time range: `slide_[index]_[HH:MM:SS]-[HH:MM:SS].jpg`

- Index is zero-padded (at least 3 digits)
- Timestamps in HH:MM:SS format
- Example: `slide_001_00:02:05-00:04:47.jpg` is slide 1 covering 2:05 to 4:47

### slides.json Format

JSON file with slides grouped by their transcripts:
```json
{
  "metadata": {
    "video_file": "lecture.mp4",
    "duration_seconds": 3600,
    "processed_at": "2025-01-15T10:30:00Z",
    "model": "large-v3"
  },
  "slides": [
    {
      "slide_number": 0,
      "image_path": "slide_000_00:00:00-00:02:05.jpg",
      "time_start": 0,
      "time_end": 125,
      "transcript": "Welcome to today's lecture on quantum mechanics."
    }
  ]
}
```

## How It Works

1. **Scene Detection**: Uses global pixel difference detection (research-based method) to identify slide changes
   - Converts frames to binary (black/white) for robustness to lighting changes
   - Computes normalized pixel differences between consecutive frames
   - Based on "An experimental comparative study on slide change detection in lecture videos" (Eruvaram et al., 2018)
2. **Slide Extraction**: Extracts the final frame before each scene change using FFmpeg
3. **Transcription**: Uses Whisper large-v3 for state-of-the-art speech-to-text with timestamps
   - Auto-detects and uses MLX on Apple Silicon for 2-3x speedup
   - Falls back to faster-whisper on other platforms
4. **Export**: Generates JSON file with slides grouped by their transcript text

## Performance

**Transcription Speed (Apple Silicon with MLX):**
- 1 hour lecture: ~10-15 minutes (large-v3 model)
- Without MLX: ~25-35 minutes

**Model Recommendations:**
- `large-v3`: Best accuracy (default) - recommended for production
- `medium`: Good balance - 2x faster, slightly lower accuracy
- `base`: Quick testing - 5x faster, noticeably lower accuracy
- `tiny`: Very fast - 10x faster, lowest accuracy

## Limitations

- Scene detection may need threshold tuning for some videos (default 0.10 works well for most lectures)
- No speaker diarization
- No automatic slide deduplication

## Development

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run linter
ruff check slidegeist/

# Run type checker
mypy slidegeist/
```

## Legal Notice

Slidegeist is provided for educational and research purposes only.
Users must ensure they have the legal right to access, download, or process any video files they use with this tool.
The author does not endorse or facilitate copyright infringement or violation of platform terms of service.

## License

MIT License - Copyright (c) 2025 Christopher Albert

See [LICENSE](LICENSE) for details.

## Contributing

Contributions welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.
