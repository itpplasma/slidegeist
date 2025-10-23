"""OCR pipeline utilities combining classic OCR and Qwen-based refinement."""

from __future__ import annotations

import json
import logging
import os
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from PIL import Image

logger = logging.getLogger(__name__)


@dataclass
class RefinementOutput:
    """Structured response from an OCR refinement model."""

    text: str
    visual_elements: List[str]
    raw_response: str


class BaseRefiner:
    """Interface for optional OCR refinement models."""

    name: str = "unknown"
    version: Optional[str] = None

    def is_available(self) -> bool:  # pragma: no cover - interface hook
        return False

    def refine(
        self,
        image_path: Path,
        raw_text: str,
        transcript: str,
        segments: List[Dict[str, Any]],
    ) -> Optional[RefinementOutput]:  # pragma: no cover - interface hook
        raise NotImplementedError


class OcrPipeline:
    """Compose Tesseract OCR with optional refinement."""

    def __init__(
        self,
        primary_extractor: Optional["TesseractExtractor"] = None,
        refiner: Optional[BaseRefiner] = None,
    ) -> None:
        self._primary = primary_extractor
        self._refiner = refiner

    def process(
        self,
        image_path: Path,
        transcript_full_text: str,
        transcript_segments: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        raw_payload: Dict[str, Any] = {
            "engine": {
                "primary": None,
                "primary_version": None,
                "refiner": None,
                "refiner_version": None,
            },
            "raw_text": "",
            "final_text": "",
            "blocks": [],
            "visual_elements": [],
            "model_response": None,
        }

        if self._primary is not None and self._primary.is_available:
            try:
                result = self._primary.extract(image_path)
                raw_payload.update({
                    "engine": result.get("engine", raw_payload["engine"]),
                    "raw_text": result.get("raw_text", ""),
                    "blocks": result.get("blocks", []),
                })
                raw_payload["final_text"] = raw_payload["raw_text"]
            except Exception as exc:  # pragma: no cover - defensive log path
                logger.warning("Primary OCR failed for %s: %s", image_path, exc)

        raw_text = raw_payload["raw_text"]

        if self._refiner is not None and self._refiner.is_available():
            try:
                refined = self._refiner.refine(
                    image_path=image_path,
                    raw_text=raw_text,
                    transcript=transcript_full_text,
                    segments=transcript_segments,
                )
            except Exception as exc:  # pragma: no cover - defensive log path
                logger.warning("OCR refinement failed for %s: %s", image_path, exc)
            else:
                if refined is not None:
                    if refined.text:
                        raw_payload["final_text"] = refined.text.strip()
                    if refined.visual_elements:
                        raw_payload["visual_elements"] = [str(item) for item in refined.visual_elements]
                    raw_payload["model_response"] = refined.raw_response
                    raw_payload["engine"]["refiner"] = self._refiner.name
                    raw_payload["engine"]["refiner_version"] = self._refiner.version

        if not raw_payload["final_text"]:
            raw_payload["final_text"] = raw_payload["raw_text"]

        return raw_payload


class TesseractExtractor:
    """Wrap pytesseract calls, keeping dependency optional."""

    def __init__(self) -> None:
        try:
            import pytesseract
        except ImportError:
            self._pytesseract = None
            self._version = None
            logger.info("pytesseract not installed; OCR will be disabled")
        else:
            self._pytesseract = pytesseract
            try:
                self._version = str(pytesseract.get_tesseract_version()).strip()
            except Exception:  # pragma: no cover - rarely triggered
                self._version = None

    @property
    def is_available(self) -> bool:
        return self._pytesseract is not None

    @property
    def version(self) -> Optional[str]:
        return self._version

    def extract(self, image_path: Path) -> Dict[str, Any]:
        if not self.is_available:
            raise RuntimeError("pytesseract is not available")

        pytesseract = self._pytesseract  # type: ignore[assignment]
        assert pytesseract is not None

        from pytesseract import Output  # type: ignore[attr-defined]

        data = pytesseract.image_to_data(
            str(image_path),
            output_type=Output.DICT,
            config="--psm 6",
        )

        blocks: List[Dict[str, Any]] = []
        raw_lines: List[str] = []

        for idx, text in enumerate(data.get("text", [])):
            text = text.strip()
            conf_str = data.get("conf", ["-1"])[idx]
            try:
                confidence = float(conf_str)
            except ValueError:
                confidence = -1.0

            if not text:
                continue

            raw_lines.append(text)
            block = {
                "text": text,
                "confidence": confidence,
                "bbox": [
                    int(data.get("left", [0])[idx]),
                    int(data.get("top", [0])[idx]),
                    int(data.get("width", [0])[idx]),
                    int(data.get("height", [0])[idx]),
                ],
                "level": int(data.get("level", [5])[idx]),
            }
            blocks.append(block)

        raw_text = " ".join(raw_lines)

        return {
            "engine": {
                "primary": "tesseract",
                "primary_version": self._version,
                "refiner": None,
                "refiner_version": None,
            },
            "raw_text": raw_text,
            "blocks": blocks,
        }


class TransformersQwenRefiner(BaseRefiner):
    """Run Qwen3-VL-4B via transformers on CPU or CUDA."""

    MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"
    CPU_INT8_MODEL_ID = "pytorch/Qwen3-4B-INT8-INT4"

    def __init__(
        self,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
    ) -> None:
        self.name = "Qwen3-VL-4B (transformers)"
        self.version = None
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

        self._torch = None
        self._AutoProcessor = None
        self._AutoModel = None
        self._BitsAndBytesConfig = None
        self._imports_ready = False
        self._load_error: Optional[Exception] = None
        self._model = None
        self._processor = None
        self._device = "cpu"
        self._loaded = False

        try:
            import torch  # pyright: ignore[reportMissingImports]
            from transformers import AutoModelForVision2Seq, AutoProcessor
        except ImportError as exc:
            self._load_error = exc
            return

        self._torch = torch
        self._AutoModel = AutoModelForVision2Seq
        self._AutoProcessor = AutoProcessor
        self._imports_ready = True

        try:  # Optional dependency
            from transformers import BitsAndBytesConfig
        except Exception:  # pragma: no cover - optional
            self._BitsAndBytesConfig = None
        else:
            self._BitsAndBytesConfig = BitsAndBytesConfig

        try:
            import transformers as _tf  # pyright: ignore[reportMissingImports]
        except ImportError:
            self.version = None
        else:
            self.version = getattr(_tf, "__version__", None)

    def is_available(self) -> bool:
        return self._imports_ready

    def refine(
        self,
        image_path: Path,
        raw_text: str,
        transcript: str,
        segments: List[Dict[str, Any]],
    ) -> Optional[RefinementOutput]:
        if not self._imports_ready:
            return None

        if os.getenv("SLIDEGEIST_DISABLE_QWEN", "").lower() in {"1", "true", "yes"}:
            logger.debug("Qwen refinement disabled via SLIDEGEIST_DISABLE_QWEN")
            return None

        try:
            self._ensure_loaded()
        except Exception as exc:  # pragma: no cover - defensive
            self._load_error = exc
            logger.warning("Failed to load Qwen3-VL-4B: %s", exc)
            return None

        if self._model is None or self._processor is None:
            return None

        torch = self._torch
        assert torch is not None

        image = Image.open(image_path).convert("RGB")
        prompt_messages = _build_prompt_messages(raw_text, transcript, segments, image)

        inputs = self._processor.apply_chat_template(
            prompt_messages,
            add_generation_prompt=True,
            return_tensors="pt",
            images=[image],
        )

        inputs = {key: value.to(self._device) for key, value in inputs.items()}

        generation_kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "do_sample": False,
        }

        with torch.no_grad():
            output = self._model.generate(**inputs, **generation_kwargs)

        response = self._processor.batch_decode(output, skip_special_tokens=True)[0]
        parsed = _parse_model_response(response, raw_text)
        return parsed

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        torch = self._torch
        assert torch is not None

        cuda_available = torch.cuda.is_available()
        device = torch.device("cuda" if cuda_available else "cpu")
        self._device = str(device)

        quantized = False
        model = None

        if cuda_available and self._BitsAndBytesConfig is not None:
            try:
                from transformers import BitsAndBytesConfig  # type: ignore
                quant_config = BitsAndBytesConfig(load_in_8bit=True)
                model = self._AutoModel.from_pretrained(  # type: ignore[operator]
                    self.MODEL_ID,
                    quantization_config=quant_config,
                    device_map="auto",
                )
                quantized = True
            except Exception as exc:  # pragma: no cover - optional path
                logger.warning("CUDA 8-bit load failed, falling back to float16: %s", exc)

        if model is None and cuda_available:
            model = self._AutoModel.from_pretrained(  # type: ignore[operator]
                self.MODEL_ID,
                torch_dtype=torch.float16,
                device_map="auto",
            )

        if model is None:
            cpu_model_id = self.CPU_INT8_MODEL_ID
            try:
                model = self._AutoModel.from_pretrained(  # type: ignore[operator]
                    cpu_model_id,
                    device_map={"": "cpu"},
                )
                quantized = True
            except Exception as exc:
                logger.warning("CPU int8 model load failed (%s); using float32", exc)
                model = self._AutoModel.from_pretrained(  # type: ignore[operator]
                    self.MODEL_ID,
                    torch_dtype=torch.float32,
                    device_map={"": "cpu"},
                )

        self._model = model
        self._model.to(device)
        self._model.eval()
        self._processor = self._AutoProcessor.from_pretrained(self.MODEL_ID)  # type: ignore[operator]
        self._loaded = True

        if quantized:
            logger.info("Loaded Qwen3-VL-4B in quantized mode on %s", device)
        else:
            logger.info("Loaded Qwen3-VL-4B in full precision on %s", device)


class MlxQwenRefiner(BaseRefiner):
    """Use MLX VLM to clean up OCR output when available on Apple Silicon."""

    MODEL_ID = "mlx-community/Qwen3-VL-4B-Instruct-4bit"

    def __init__(
        self,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
    ) -> None:
        self.name = "Qwen3-VL-4B (mlx)"
        self.version = None
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

        self._load = None
        self._generate = None
        self._apply_chat_template = None
        self._model = None
        self._processor = None
        self._config = None
        self._available = False

        if platform.system() != "Darwin":
            return

        try:
            from mlx_vlm import generate, load  # type: ignore
            from mlx_vlm.prompt_utils import apply_chat_template  # type: ignore
        except ImportError:
            logger.info("mlx-vlm not installed; MLX refinement disabled")
            return

        self._load = load
        self._generate = generate
        self._apply_chat_template = apply_chat_template
        self._available = True

        try:
            import mlx_vlm  # type: ignore
        except ImportError:  # pragma: no cover - defensive
            self.version = None
        else:
            self.version = getattr(mlx_vlm, "__version__", None)

    def is_available(self) -> bool:
        return self._available

    def refine(
        self,
        image_path: Path,
        raw_text: str,
        transcript: str,
        segments: List[Dict[str, Any]],
    ) -> Optional[RefinementOutput]:
        if not self._available:
            return None

        if os.getenv("SLIDEGEIST_DISABLE_QWEN", "").lower() in {"1", "true", "yes"}:
            logger.debug("Qwen refinement disabled via SLIDEGEIST_DISABLE_QWEN")
            return None

        self._ensure_loaded()

        if self._model is None or self._processor is None:
            return None

        prompt_messages = _build_prompt_messages(raw_text, transcript, segments, str(image_path))

        formatted = self._apply_chat_template(  # type: ignore[misc]
            self._processor,
            self._config,
            prompt_messages,
            add_generation_prompt=True,
        )

        output = self._generate(  # type: ignore[misc]
            self._model,
            self._processor,
            formatted,
            images=[str(image_path)],
            max_tokens=self.max_new_tokens,
            temperature=self.temperature,
            verbose=False,
        )

        if isinstance(output, str):
            response = output
        elif isinstance(output, dict):
            response = str(output.get("choices", [{}])[0].get("text", ""))
        else:  # pragma: no cover - unexpected path
            response = str(output)

        parsed = _parse_model_response(response, raw_text)
        return parsed

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return

        if self._load is None:
            raise RuntimeError("mlx-vlm load function unavailable")

        self._model, self._processor = self._load(self.MODEL_ID)
        self._config = getattr(self._model, "config", None)


def _build_prompt_messages(
    raw_text: str,
    transcript: str,
    segments: List[Dict[str, Any]],
    image_token: Any,
) -> List[Dict[str, Any]]:
    """Construct chat template for Qwen refiners."""

    segment_payload = [
        {
            "start": float(segment.get("start", 0.0)),
            "end": float(segment.get("end", 0.0)),
            "text": segment.get("text", ""),
        }
        for segment in segments
    ]

    context = {
        "raw_ocr_text": raw_text,
        "transcript_context": transcript,
        "segments": segment_payload,
    }

    system_instruction = (
        "You are enhancing OCR for lecture slides."
        " Return precise slide text and describe non-text elements such as tables,"
        " figures, drawings, arrows, boxes, charts, diagrams, and annotations."
        " Answer in JSON with keys 'text' and 'visual_elements'."
    )

    user_text = (
        "Analyze the slide image and the provided context."
        " Respond ONLY with JSON of the form {" "\"text\": ..., \"visual_elements\": [...]" " }"
        " where visual_elements is a list of short descriptions for each visual item beyond raw text."
        f"\nContext: {json.dumps(context, ensure_ascii=False)}"
    )

    content: List[Dict[str, Any]] = []
    if image_token is not None:
        content.append({"type": "image", "image": image_token})
    content.append({"type": "text", "text": user_text})

    return [
        {"role": "system", "content": [{"type": "text", "text": system_instruction}]},
        {"role": "user", "content": content},
    ]


def _parse_model_response(response: str, fallback_text: str) -> RefinementOutput:
    """Parse model response JSON, tolerating minor formatting errors."""

    response = response.strip()
    candidate = response

    if not candidate:
        return RefinementOutput(text=fallback_text, visual_elements=[], raw_response=response)

    json_obj: Optional[Dict[str, Any]] = None
    if candidate.startswith("{"):
        try:
            json_obj = json.loads(candidate)
        except json.JSONDecodeError:
            pass

    if json_obj is None:
        try:
            start = candidate.index("{")
            end = candidate.rfind("}")
            json_obj = json.loads(candidate[start : end + 1])
        except Exception:
            return RefinementOutput(text=candidate, visual_elements=[], raw_response=response)

    text_value = str(json_obj.get("text", "")).strip()
    elements_raw = json_obj.get("visual_elements", [])
    if isinstance(elements_raw, str):
        elements_list = [elements_raw]
    elif isinstance(elements_raw, list):
        elements_list = [str(item).strip() for item in elements_raw if str(item).strip()]
    else:
        elements_list = []

    if not text_value:
        text_value = candidate

    return RefinementOutput(text=text_value, visual_elements=elements_list, raw_response=response)


def build_default_ocr_pipeline() -> OcrPipeline:
    """Create the default OCR pipeline using available components."""

    primary = TesseractExtractor()
    refiner: Optional[BaseRefiner] = None

    if os.getenv("SLIDEGEIST_DISABLE_QWEN", "").lower() not in {"1", "true", "yes"}:
        if platform.system() == "Darwin":
            candidate = MlxQwenRefiner()
            if candidate.is_available():
                refiner = candidate
        if refiner is None:
            transformer_refiner = TransformersQwenRefiner()
            if transformer_refiner.is_available():
                refiner = transformer_refiner

    if not primary.is_available:
        primary = None  # type: ignore[assignment]

    return OcrPipeline(primary_extractor=primary, refiner=refiner)
