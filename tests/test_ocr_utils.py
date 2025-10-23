"""Tests for OCR refinement helpers and manual end-to-end checks."""

from pathlib import Path

import pytest

from slidegeist.ocr import RefinementOutput, _parse_model_response


@pytest.mark.parametrize(
    "payload,expected_text,expected_elements",
    [
        (
            '{"text": "Exact text", "visual_elements": ["chart", "arrow"]}',
            "Exact text",
            ["chart", "arrow"],
        ),
        (
            "Answer: {\"text\": \"Slide content\", \"visual_elements\": \"table\"}",
            "Slide content",
            ["table"],
        ),
        (
            "No JSON here",
            "No JSON here",
            [],
        ),
    ],
)
def test_parse_model_response(payload: str, expected_text: str, expected_elements: list[str]) -> None:
    result = _parse_model_response(payload, "fallback")
    assert isinstance(result, RefinementOutput)
    assert result.text == expected_text
    assert result.visual_elements == expected_elements


@pytest.mark.manual
def test_manual_qwen_pipeline(tmp_path: Path) -> None:  # type: ignore[no-redef]
    from PIL import Image, ImageDraw

    from slidegeist.ocr import build_default_ocr_pipeline

    image_path = tmp_path / "slide_manual.jpg"
    image = Image.new("RGB", (640, 360), color=(255, 255, 255))
    draw = ImageDraw.Draw(image)
    draw.rectangle((50, 200, 200, 280), outline="red", width=8)
    draw.text((60, 60), "Deep Learning Overview", fill=(0, 0, 0))
    draw.text((60, 120), "- Convolutional Networks", fill=(0, 0, 0))
    image.save(image_path)

    pipeline = build_default_ocr_pipeline()
    assert pipeline._primary is not None and pipeline._primary.is_available, "Tesseract missing"
    assert pipeline._refiner is not None and pipeline._refiner.is_available(), "Qwen refiner missing"

    result = pipeline.process(
        image_path=image_path,
        transcript_full_text="Today we cover convolutional networks",
        transcript_segments=[{"start": 0.0, "end": 5.0, "text": "Today we cover convolutional networks"}],
    )

    assert "Deep" in result["final_text"]
    assert any("rectangle" in item.lower() or "box" in item.lower() for item in result["visual_elements"])
