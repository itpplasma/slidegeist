"""Constants used across the slidegeist package."""

# Scene detection
DEFAULT_SCENE_THRESHOLD = 0.3  # Works well for most slides including handwritten content

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
