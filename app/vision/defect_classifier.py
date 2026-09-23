"""Async service wrapping ProductDefectCNN for use as an agent tool.

Loads weights lazily (thread-pooled to avoid blocking the event loop),
preprocesses incoming images (base64 -> tensor, resize/normalize), and
returns a calibrated verdict including a `requires_manual_review` flag so
the ActionExecutorNode / HITL layer can decide whether a low-confidence or
high-severity classification should be escalated to a human rather than
auto-approving a return/refund.
"""
from __future__ import annotations

import asyncio
import base64
import io
import logging
import time
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from app.vision.cnn_model import DEFECT_CLASSES, ProductDefectCNN

logger = logging.getLogger("vision.defect_classifier")

_SEVERITY_BY_CLASS = {
    "no_defect": 0.0,
    "surface_scratch": 0.3,
    "discoloration_or_stain": 0.3,
    "packaging_damage": 0.4,
    "missing_component": 0.7,
    "crack_or_fracture": 0.9,
}

_PREPROCESS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class DefectClassifierService:
    def __init__(self, weights_path: str | Path | None = None, confidence_threshold: float = 0.75) -> None:
        self._weights_path = Path(weights_path) if weights_path else None
        self._confidence_threshold = confidence_threshold
        self._model: ProductDefectCNN | None = None
        self._lock = asyncio.Lock()

    async def _ensure_loaded(self) -> ProductDefectCNN:
        if self._model is not None:
            return self._model
        async with self._lock:
            if self._model is None:
                self._model = await asyncio.get_event_loop().run_in_executor(None, self._load_sync)
        return self._model

    def _load_sync(self) -> ProductDefectCNN:
        model = ProductDefectCNN()
        if self._weights_path and self._weights_path.exists():
            state_dict = torch.load(self._weights_path, map_location="cpu")
            model.load_state_dict(state_dict)
            logger.info("loaded CNN weights from %s", self._weights_path)
        else:
            logger.warning(
                "no trained weights found at %s; running with randomly-initialized "
                "weights (development/demo mode only — retrain before production use)",
                self._weights_path,
            )
        model.eval()
        return model

    @staticmethod
    def _decode_image(image_base64: str) -> Image.Image:
        raw = base64.b64decode(image_base64)
        return Image.open(io.BytesIO(raw)).convert("RGB")

    async def classify(self, image_base64: str) -> dict:
        start = time.perf_counter()
        model = await self._ensure_loaded()

        def _run_inference() -> dict:
            image = self._decode_image(image_base64)
            tensor = _PREPROCESS(image).unsqueeze(0)
            probs = model.predict_proba(tensor).squeeze(0)
            scores = {cls: float(probs[i]) for i, cls in enumerate(DEFECT_CLASSES)}
            predicted_class = max(scores, key=scores.get)
            confidence = scores[predicted_class]
            severity = _SEVERITY_BY_CLASS.get(predicted_class, 0.5)
            return {
                "predicted_class": predicted_class,
                "confidence": confidence,
                "all_scores": scores,
                "severity_score": severity,
                "requires_manual_review": (
                    confidence < self._confidence_threshold or severity >= 0.7
                ),
            }

        result = await asyncio.get_event_loop().run_in_executor(None, _run_inference)
        result["inference_latency_ms"] = (time.perf_counter() - start) * 1000
        return result
