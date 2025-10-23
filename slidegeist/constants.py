"""Constants used across the slidegeist package."""

# Scene detection - Global Pixel Difference (research-proven best method)
DEFAULT_SCENE_THRESHOLD = 0.03  # Normalized pixel difference threshold (0-1 scale)
                                # Lower = more sensitive. Typical range: 0.02-0.10
DEFAULT_MIN_SCENE_LEN = 2.0  # Minimum scene length in seconds
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
