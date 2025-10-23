"""Integration-style tests exercising the high-level pipeline and CLI."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

from slidegeist import cli
from slidegeist.pipeline import process_video


def test_process_video_produces_slides_json(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure process_video writes slides.json and returns paths."""
    def fake_detect_scenes(*_: Any, **__: Any) -> list[float]:
        return [2.0]

    def fake_extract_slides(
        video_path: Path,
        scene_timestamps: list[float],
        output_dir: Path,
        image_format: str
    ) -> list[tuple[int, float, float, Path]]:
        output_dir.mkdir(parents=True, exist_ok=True)
        paths: list[tuple[int, float, float, Path]] = []
        for index, start in enumerate([0.0, 2.0]):
            end = start + 2.0
            slide_path = output_dir / f"slide_{index:03d}.{image_format}"
            slide_path.write_bytes(b"fake image")
            paths.append((index, start, end, slide_path))
        return paths

    def fake_transcribe_video(*_: Any, **__: Any) -> dict[str, Any]:
        return {
            "language": "en",
            "segments": [
                {"start": 0.0, "end": 1.0, "text": "Hello", "words": []},
                {"start": 2.0, "end": 3.0, "text": "World", "words": []},
            ],
        }

    monkeypatch.setattr("slidegeist.pipeline.detect_scenes", fake_detect_scenes)
    monkeypatch.setattr("slidegeist.pipeline.extract_slides", fake_extract_slides)
    monkeypatch.setattr("slidegeist.pipeline.transcribe_video", fake_transcribe_video)

    video_path = tmp_path / "dummy.mp4"
    video_path.write_bytes(b"fake video content")

    result = process_video(
        video_path=video_path,
        output_dir=tmp_path / "out",
        scene_threshold=0.05,
        min_scene_len=1.0,
        start_offset=0.0,
        model="tiny",
        device="cpu",
        image_format="png",
    )

    slides = result.get("slides")
    assert isinstance(slides, list)
    assert len(slides) == 2

    json_path = result.get("slides_json")
    assert isinstance(json_path, Path)
    assert json_path.exists()

    data = json_path.read_text()
    assert '"slides"' in data
    assert '"slide_number": 0' in data
    assert '"Hello"' in data


def test_cli_process_default_invocation(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Calling cli.main without subcommand should execute process pipeline."""
    output_dir = tmp_path / "cli-out"

    def fake_process_video(*_: Any, **__: Any) -> dict[str, Any]:
        json_path = output_dir / "slides.json"
        output_dir.mkdir(parents=True, exist_ok=True)
        json_path.write_text('{"slides": []}')
        return {
            "output_dir": output_dir,
            "slides": [],
            "slides_json": json_path,
        }

    monkeypatch.setattr("slidegeist.cli.process_video", fake_process_video)
    monkeypatch.setattr("slidegeist.cli.check_prerequisites", lambda: None)
    monkeypatch.setattr("slidegeist.cli.resolve_video_path", lambda value, cookies_from_browser=None: Path(value))
    monkeypatch.setattr(sys, "argv", ["slidegeist", str(tmp_path / "input.mp4")])

    cli.main()

    captured = capsys.readouterr()
    assert "Processing complete" in captured.out
    assert "slides.json" in captured.out
