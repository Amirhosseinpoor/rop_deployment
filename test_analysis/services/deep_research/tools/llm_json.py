"""
T5 — ``json_llm``: a guaranteed-JSON structured LLM call.

``LLMClient`` (doctors_marketplace) has no JSON mode, so — exactly like
``medical_test_extraction`` — we talk to the OpenAI-compatible GAPGPT endpoint
directly and ask for ``response_format={"type":"json_object"}``.

Robust by design: falls back to plain completion + JSON-substring extraction if
the provider/model rejects ``response_format``, and drops ``temperature`` if the
model insists on the default. Never returns a half-parsed blob — either a real
``dict``/``list`` or it raises.
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
from typing import Any, Dict, List, Optional

log = logging.getLogger("test_analysis.deep_research")

_API_KEY = os.getenv("GAPGPT_API_KEY") or os.getenv("OPENAI_API_KEY") or os.getenv("METIS_API_KEY")
_BASE_URL = os.getenv("GAPGPT_BASE_URL") or os.getenv("BASE_URL") or "https://api.gapgpt.app/v1"

# Model tier (all GAPGPT, all overridable per deployment):
#   SYNTH     — authoring / synthesis / critic (strongest; set to gpt-4o if available)
#   REASONING — triage / questions / planning / verify
#   VISION    — eye-crop reading
SYNTH_MODEL = os.getenv("DR_SYNTH_MODEL") or os.getenv("DR_REASONING_MODEL") or "gpt-4o-mini"
REASONING_MODEL = os.getenv("DR_REASONING_MODEL") or os.getenv("MEDICAL_TEST_MODEL") or "gpt-4o-mini"
VISION_MODEL = os.getenv("DR_VISION_MODEL") or "gpt-4o-mini"

_client_singleton = None


def _client():
    global _client_singleton
    if _client_singleton is None:
        from openai import OpenAI
        _client_singleton = OpenAI(api_key=_API_KEY, base_url=_BASE_URL, timeout=90)
    return _client_singleton


def _extract_json(raw: str) -> Any:
    """Pull the first JSON object/array out of a model reply (robust to fences)."""
    if not raw:
        raise ValueError("empty LLM reply")
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?|```$", "", raw, flags=re.M).strip()
    m = re.search(r"[\{\[].*[\}\]]", raw, re.S)
    if not m:
        raise ValueError("no JSON found in reply")
    return json.loads(m.group())


def json_llm(messages: List[Dict[str, Any]], *, model: Optional[str] = None,
             temperature: float = 0.3, max_retries: int = 3) -> Any:
    """Call the LLM and return parsed JSON (dict or list).

    ``messages`` must instruct the model to answer in JSON (the word "json" has
    to appear for the provider's JSON mode). All DR prompts do.
    """
    client = _client()
    model = model or REASONING_MODEL
    last_err: Optional[Exception] = None
    use_response_format = True
    use_temperature = True

    for attempt in range(max_retries):
        try:
            kwargs: Dict[str, Any] = {"model": model, "messages": messages}
            if use_temperature:
                kwargs["temperature"] = temperature
            if use_response_format:
                kwargs["response_format"] = {"type": "json_object"}
            resp = client.chat.completions.create(**kwargs)
            raw = resp.choices[0].message.content or ""
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                return _extract_json(raw)
        except Exception as e:  # noqa: BLE001
            last_err = e
            msg = str(e).lower()
            if "response_format" in msg or "json" in msg and "unsupported" in msg:
                use_response_format = False
            if "temperature" in msg:
                use_temperature = False
            log.warning("DR | json_llm attempt %d/%d failed (%s) — retrying",
                        attempt + 1, max_retries, e)
            time.sleep(1.2 * (attempt + 1))

    raise RuntimeError(f"json_llm failed after {max_retries} attempts: {last_err}")
