"""Constants used across the slidegeist package."""

# Scene detection - Histogram-based (optimized for presentation videos)
DEFAULT_SCENE_THRESHOLD = 25.0  # Histogram correlation threshold (0-100 scale, lower = more sensitive)
                                # Typical range: 20-30 for presentations (maps to 0.70-0.80 correlation)
DEFAULT_MIN_SCENE_LEN = 0.5  # Minimum scene length in seconds (filters rapid mouse clicks)
DEFAULT_START_OFFSET = 3.0  # Skip first N seconds to avoid mouse movement during setup

# Whisper transcription
DEFAULT_WHISPER_MODEL = "large-v3"  # Best accuracy
DEFAULT_DEVICE = "auto"  # Auto-detect MLX on Apple Silicon, else CPU

# Transcription quality thresholds
COMPRESSION_RATIO_THRESHOLD = 2.4  # Prevent hanging on compression issues
LOG_PROB_THRESHOLD = -1.0  # Less strict filtering for better results
NO_SPEECH_THRESHOLD = 0.6  # Default whisper value

# Output formats
DEFAULT_IMAGE_FORMAT = "jpg"
DEFAULT_OUTPUT_DIR = "output"
