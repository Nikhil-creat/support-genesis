"""Tool entrypoint for CNN-based product defect / damage classification.

Bound into the agent graph as `classify_product_defect_image` so it can be
invoked either directly by `VisualInspectionNode` (when a photo accompanies
the user's message) or by `ActionExecutorNode`'s LLM-driven tool planner for
damage-claim / return-eligibility intents.
"""
from __future__ import annotations

from app.schemas import ClassifyProductImageInput, ClassifyProductImageResult
from app.vision.defect_classifier import DefectClassifierService


async def classify_product_defect_image(
    service: DefectClassifierService, request: ClassifyProductImageInput
) -> ClassifyProductImageResult:
    raw = await service.classify(request.image_base64)
    return ClassifyProductImageResult(order_id=request.order_id, **raw)
