# Slidegeist

Extract slides and timestamped transcripts from lecture videos with minimal dependencies.

## Features

- 🎬 **Scene detection** using FFmpeg's built-in scene filter
- 🖼️ **Automatic slide extraction** with timestamp ranges in filenames
- 🎤 **Audio transcription** with faster-whisper (no PyTorch required)
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
# Basic usage
slidegeist process video.mp4

# Specify output directory
slidegeist process video.mp4 --out my-output/

# Use GPU for faster transcription
slidegeist process video.mp4 --device cuda --model medium

# Adjust scene detection sensitivity (0.0-1.0, default 0.4)
slidegeist process video.mp4 --scene-threshold 0.3
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
  --out DIR              Output directory (default: output/)
  --scene-threshold NUM  Scene detection sensitivity 0.0-1.0 (default: 0.4)
  --model NAME          Whisper model: tiny, base, small, medium, large (default: base)
  --device NAME         Device: cpu or cuda (default: cpu)
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
3. **Transcription**: Uses faster-whisper for accurate speech-to-text with timestamps
4. **Export**: Generates SRT subtitle file compatible with video players

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
