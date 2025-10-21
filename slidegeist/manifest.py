"""JSON manifest generation for ChatGPT/GPT-5 ingestion."""

import hashlib
import json
from pathlib import Path
from typing import TypedDict


class ManifestWord(TypedDict):
    """Word-level transcript entry."""
    t_start: float
    t_end: float
    text: str
    conf: float
    lang: str


class ManifestSegment(TypedDict):
    """Segment-level transcript entry."""
    t_start: float
    t_end: float
    text: str
    avg_conf: float
    word_idx_start: int
    word_idx_end: int
    lang: str


class ManifestSlide(TypedDict):
    """Slide entry with image path and transcript span."""
    index: int
    t_start: float
    t_end: float
    image: str
    transcript_span: dict[str, int]  # word_idx_start, word_idx_end


class SlidegeistManifest(TypedDict):
    """Complete manifest structure for ChatGPT ingestion."""
    version: str
    timebase: str
    source: dict[str, str | float]
    models: dict[str, str | float | int]
    hashes: dict[str, str]
    transcript: dict[str, list]
    slides: list[ManifestSlide]


def compute_file_hash(file_path: Path) -> str:
    """Compute SHA-256 hash of a file.

    Args:
        file_path: Path to file to hash.

    Returns:
        Hex-encoded SHA-256 hash.
    """
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(8192):
            sha256.update(chunk)
    return sha256.hexdigest()


def create_manifest(
    video_path: Path,
    duration: float,
    language: str,
    model_name: str,
    compute_type: str,
    vad_threshold: float,
    beam_size: int,
    max_segment_length: int,
    detector_config: str,
    segments: list,
    slides: list[tuple[int, float, float, str]],
    compute_hashes: bool = False,
) -> SlidegeistManifest:
    """Create the index.json manifest structure.

    Args:
        video_path: Path to source video file.
        duration: Video duration in seconds.
        language: Detected language code (ISO-639-1).
        model_name: Whisper model name.
        compute_type: CTranslate2 compute type.
        vad_threshold: VAD threshold used.
        beam_size: Beam size for decoding.
        max_segment_length: Max segment length in seconds.
        detector_config: Scene detector configuration string.
        segments: List of transcript segments with words.
        slides: List of (index, t_start, t_end, image_path) tuples.
        compute_hashes: Whether to compute file hashes (slow for large files).

    Returns:
        Complete manifest dictionary.
    """
    # Build global words list from all segments
    words: list[ManifestWord] = []
    for segment in segments:
        for word in segment.get("words", []):
            words.append({
                "t_start": round(word["start"], 3),
                "t_end": round(word["end"], 3),
                "text": word["word"].strip(),
                "conf": 1.0,  # faster-whisper doesn't provide word-level confidence
                "lang": language,
            })

    # Build segments with word index ranges
    manifest_segments: list[ManifestSegment] = []
    word_idx = 0
    for segment in segments:
        seg_word_count = len(segment.get("words", []))
        manifest_segments.append({
            "t_start": round(segment["start"], 3),
            "t_end": round(segment["end"], 3),
            "text": segment["text"].strip(),
            "avg_conf": 1.0,
            "word_idx_start": word_idx,
            "word_idx_end": word_idx + seg_word_count,
            "lang": language,
        })
        word_idx += seg_word_count

    # Build slides with word mapping using midpoint logic
    manifest_slides: list[ManifestSlide] = []
    for slide_idx, t_start, t_end, image_path in slides:
        # Find words whose midpoint falls in [t_start, t_end)
        word_indices = []
        for idx, word in enumerate(words):
            midpoint = (word["t_start"] + word["t_end"]) / 2.0
            if t_start <= midpoint < t_end:
                word_indices.append(idx)

        # Determine word range
        if word_indices:
            word_idx_start = word_indices[0]
            word_idx_end = word_indices[-1] + 1
        else:
            # No words in this slide (silent segment)
            # Find closest word index
            word_idx_start = 0
            word_idx_end = 0
            for idx, word in enumerate(words):
                if word["t_start"] >= t_start:
                    word_idx_start = idx
                    word_idx_end = idx
                    break

        manifest_slides.append({
            "index": slide_idx,
            "t_start": round(t_start, 3),
            "t_end": round(t_end, 3),
            "image": image_path,
            "transcript_span": {
                "word_idx_start": word_idx_start,
                "word_idx_end": word_idx_end,
            },
        })

    # Compute hashes if requested
    video_hash = ""
    if compute_hashes and video_path.exists():
        video_hash = compute_file_hash(video_path)

    manifest: SlidegeistManifest = {
        "version": "1.0",
        "timebase": "seconds",
        "source": {
            "video_path": str(video_path.name),
            "duration_sec": round(duration, 3),
            "language": language,
        },
        "models": {
            "asr": f"faster-whisper-{model_name}",
            "asr_compute_type": compute_type,
            "vad_threshold": vad_threshold,
            "beam_size": beam_size,
            "max_segment_length_sec": max_segment_length,
            "scene_detector": detector_config,
        },
        "hashes": {
            "video_sha256": video_hash,
            "audio_sha256": "",  # Not computed separately
        },
        "transcript": {
            "words": words,
            "segments": manifest_segments,
        },
        "slides": manifest_slides,
    }

    return manifest


def save_manifest(manifest: SlidegeistManifest, output_path: Path) -> None:
    """Save manifest to JSON file.

    Args:
        manifest: Complete manifest dictionary.
        output_path: Path to write index.json.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
