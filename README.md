# Slidegeist

Extract slides and timestamped transcripts from lecture videos with minimal dependencies.

## Features

- 🎬 **Scene detection** using FFmpeg's built-in scene filter
- 🖼️ **Automatic slide extraction** with timestamp ranges in filenames
- 🎤 **Audio transcription** with Whisper large-v3 model (highest quality)
- 🚀 **MLX acceleration** on Apple Silicon Macs for 2-3x faster transcription
- 📝 **SRT subtitle export** compatible with all video players

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
slidegeist process lecture.mp4 --out output/
```

This creates:
```
output/
├── slide_000000000-000125300.jpg  # Slide from 0:00.000 to 2:05.300
├── slide_000125300-000287600.jpg  # Slide from 2:05.300 to 4:47.600
├── slide_000287600-000450000.jpg  # Slide from 4:47.600 to 7:30.000
└── transcript.srt                  # Full transcript with timestamps
```

## Usage

### Full Processing

```bash
# Basic usage (auto-detects MLX on Apple Silicon, uses large-v3 model)
slidegeist process video.mp4

# Specify output directory
slidegeist process video.mp4 --out my-output/

# Use GPU explicitly (NVIDIA)
slidegeist process video.mp4 --device cuda

# Use smaller/faster model
slidegeist process video.mp4 --model base

# Adjust scene detection sensitivity (0.0-1.0, default 0.10)
# Lower values detect more subtle changes, higher values only major transitions
slidegeist process video.mp4 --scene-threshold 0.15
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
slidegeist process <video> [options]

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

Slides are named with their time range: `slide_[start_ms]-[end_ms].jpg`

- Timestamps in milliseconds (9 digits, zero-padded)
- Example: `slide_000125300-000287600.jpg` covers 2:05.300 to 4:47.600

### Transcript File

Standard SRT subtitle file format:
```srt
1
00:00:00,000 --> 00:00:05,200
Welcome to today's lecture on quantum mechanics.

2
00:00:05,200 --> 00:00:12,800
We'll be covering the fundamentals of wave functions.
```

## How It Works

1. **Scene Detection**: Uses FFmpeg's scene filter to detect slide changes
2. **Slide Extraction**: Extracts the final frame before each scene change
3. **Transcription**: Uses Whisper large-v3 for state-of-the-art speech-to-text with timestamps
   - Auto-detects and uses MLX on Apple Silicon for 2-3x speedup
   - Falls back to faster-whisper on other platforms
4. **Export**: Generates SRT subtitle file compatible with video players

## Performance

**Transcription Speed (Apple Silicon with MLX):**
- 1 hour lecture: ~10-15 minutes (large-v3 model)
- Without MLX: ~25-35 minutes

**Model Recommendations:**
- `large-v3`: Best accuracy (default) - recommended for production
- `medium`: Good balance - 2x faster, slightly lower accuracy
- `base`: Quick testing - 5x faster, noticeably lower accuracy
- `tiny`: Very fast - 10x faster, lowest accuracy

## Limitations (MVP)

- Works with local video files only (no web scraping)
- Basic scene detection (may need threshold tuning for some videos)
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

## License

MIT License - Copyright (c) 2025 Plasma Physics at TU Graz

See [LICENSE](LICENSE) for details.

## Contributing

Contributions welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.
