"""Thin OpenAI-compatible chat client.

Ported from the original ``services/llm.py``. Credentials and model come from
config (env vars). The client is created lazily so the service can boot without
an API key (chat calls will then fail with a clear error, rather than crashing
at import time).
"""
from __future__ import annotations

from openai import OpenAI

from .config import get_settings


class LLMClient:
    """Minimal wrapper around the OpenAI Chat Completions API."""

    def __init__(self) -> None:
        settings = get_settings()
        if not settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY not set")
        # ``base_url=None`` makes the SDK use the default OpenAI endpoint; an
        # empty string would be invalid, so normalise to None.
        self._client = OpenAI(
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url or None,
        )
        self._model = settings.openai_model

    def chat(self, messages: list[dict[str, str]]) -> str:
        """Send a list of ``{role, content}`` messages and return the reply text."""
        resp = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=0.4,
        )
        return (resp.choices[0].message.content or "").strip()
