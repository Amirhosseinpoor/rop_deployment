"""
T4 — ``vision_read``: multimodal reading of an image into structured JSON.

Used by the Eye Vision Analyst (A1) to actually *look at* the conjunctiva crops
the segmentation model produced — pallor, vascularity, image quality — instead of
trusting only the classifier's label. This is the single most "un-clonable" lever
in the report: a text-only model can never reproduce it.

Reuses the data-URL builder from ``medical_test_extraction`` and the JSON-mode
client from ``llm_json``.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List

from .llm_json import json_llm, VISION_MODEL

log = logging.getLogger("test_analysis.deep_research")


def _image_data_url(path: str) -> str:
    from test_analysis.services.medical_test_extraction import _image_data_url as _b
    return _b(path)


def vision_read(image_paths: List[str], system_prompt: str, user_prompt: str,
                *, temperature: float = 0.0) -> Dict[str, Any]:
    """Feed one or more local image files + a prompt to the VLM, get JSON back.

    ``system_prompt`` / ``user_prompt`` must both make the JSON contract explicit.
    Returns the parsed object (or raises — callers are best-effort and catch).
    """
    content: List[Dict[str, Any]] = [{"type": "text", "text": user_prompt}]
    for p in image_paths:
        try:
            content.append({"type": "image_url", "image_url": {"url": _image_data_url(p)}})
        except Exception as e:  # noqa: BLE001
            log.warning("DR | vision could not encode image %s: %s", p, e)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": content},
    ]
    result = json_llm(messages, model=VISION_MODEL, temperature=temperature)
    return result if isinstance(result, dict) else {"raw": result}
