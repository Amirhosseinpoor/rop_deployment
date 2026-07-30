# doctors_marketplace/services/llm.py
"""
Thin wrapper around the GAPGPT (OpenAI-compatible) chat API.

The marketplace assistants run on `gpt-5-nano`, which is a vision-language
model, so `chat()` accepts the standard OpenAI message format where a message
`content` may either be a plain string or a list of content parts
(`{"type": "text", ...}` / `{"type": "image_url", ...}`). This lets callers
send images straight through to the model.
"""
import os
import time
from typing import List, Dict, Union

from openai import OpenAI


def _env(*names, default=None):
    """Return the first non-empty environment variable among *names."""
    for n in names:
        v = os.getenv(n)
        if v:
            return v.strip().strip('"').strip("'")
    return default


# Prefer GAPGPT; fall back to the generic OPENAI_* / BASE_URL pair.
API_KEY = _env("GAPGPT_API_KEY", "OPENAI_API_KEY")
BASE_URL = _env("GAPGPT_BASE_URL", "BASE_URL", default="https://api.gapgpt.app/v1")
MODEL = _env("GAPGPT_MODEL", "OPENAI_MODEL", default="gpt-5-nano")

Message = Dict[str, Union[str, list]]


class LLMClient:
    def __init__(self, model: str | None = None):
        if not API_KEY:
            raise RuntimeError("No LLM API key set (GAPGPT_API_KEY / OPENAI_API_KEY)")
        self.model = model or MODEL
        self.client = OpenAI(api_key=API_KEY, base_url=BASE_URL, timeout=90)

    def chat(self, messages: List[Message], temperature: float = 0.4,
             max_retries: int = 4) -> str:
        """
        Send a chat completion request. `messages` follows the OpenAI schema and
        may contain multimodal content parts. Retries transient rate limits
        (HTTP 429) with a short exponential backoff.
        """
        last_err = None
        for attempt in range(max_retries):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                )
                return (resp.choices[0].message.content or "").strip()
            except Exception as e:  # noqa: BLE001 - surface a clean message upstream
                last_err = e
                status = getattr(e, "status_code", None)
                msg = str(e).lower()
                transient = status in (429, 500, 502, 503) or "rate" in msg or "429" in msg
                if transient and attempt < max_retries - 1:
                    time.sleep(2 * (attempt + 1))
                    continue
                raise
        raise last_err  # pragma: no cover

    def complete(self, messages: List[Message], tools=None, tool_choice="auto",
                 temperature: float = 0.3, max_retries: int = 3):
        """Non-streaming completion that may return tool calls. Returns the raw
        assistant message object (has `.content` and `.tool_calls`)."""
        kwargs = {"model": self.model, "messages": messages, "temperature": temperature}
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = tool_choice
        last_err = None
        for attempt in range(max_retries):
            try:
                resp = self.client.chat.completions.create(**kwargs)
                return resp.choices[0].message
            except Exception as e:  # noqa: BLE001
                last_err = e
                status = getattr(e, "status_code", None)
                msg = str(e).lower()
                transient = status in (429, 500, 502, 503) or "rate" in msg or "429" in msg
                if transient and attempt < max_retries - 1:
                    time.sleep(2 * (attempt + 1))
                    continue
                raise
        raise last_err  # pragma: no cover

    def chat_stream(self, messages: List[Message], temperature: float = 0.4):
        """
        Yield the assistant reply incrementally (text deltas) using the
        streaming completions API. Raises on the first hard error.
        """
        stream = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            stream=True,
        )
        for ev in stream:
            if not ev.choices:
                continue
            delta = ev.choices[0].delta
            piece = getattr(delta, "content", None) if delta else None
            if piece:
                yield piece
