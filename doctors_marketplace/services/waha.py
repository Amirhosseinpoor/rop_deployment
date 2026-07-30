# doctors_marketplace/services/waha.py
"""Minimal WAHA (WhatsApp HTTP API) client for reminders / follow-ups.

Configure via environment:
    WAHA_URL      e.g. http://localhost:3000   (required to actually send)
    WAHA_SESSION  WAHA session name            (default "default")
    WAHA_API_KEY  optional  -> sent as X-Api-Key header

Contract: POST {WAHA_URL}/api/sendText {session, chatId, text}, chatId = "<digits>@c.us".
Degrades gracefully (returns (False, reason)) when not configured, so the rest of
the app keeps working without a live WhatsApp gateway.
"""
from __future__ import annotations

import logging
import re

import requests

from .llm import _env

log = logging.getLogger("doctors_marketplace.services.waha")


def normalize_phone(phone: str) -> str:
    """Digits only (country code, no +, no separators)."""
    return re.sub(r"\D", "", phone or "")


def base_url() -> str:
    return (_env("WAHA_URL", "WAHA_BASE_URL", default="http://localhost:3000") or "").rstrip("/")


def session_name() -> str:
    return _env("WAHA_SESSION", default="default")


def is_configured() -> bool:
    return bool(_env("WAHA_API_KEY") or _env("WAHA_URL", "WAHA_BASE_URL"))


def send_whatsapp(phone: str, text: str, session: str | None = None) -> tuple[bool, str]:
    """Send a WhatsApp text via WAHA. Returns (ok, detail)."""
    url = base_url()
    if not url:
        return False, "WAHA base URL not configured"
    digits = normalize_phone(phone)
    if len(digits) < 8:
        return False, f"invalid phone '{phone}'"
    sess = session or session_name()
    headers = {"Accept": "application/json", "Content-Type": "application/json"}
    api_key = _env("WAHA_API_KEY")
    if api_key:
        headers["X-Api-Key"] = api_key
    payload = {"session": sess, "chatId": f"{digits}@c.us", "text": text}
    try:
        resp = requests.post(f"{url}/api/sendText", json=payload, headers=headers, timeout=20)
        ok = resp.status_code in (200, 201)
        log.info("WAHA | sendText -> %s (%s) session=%s chat=%s", resp.status_code,
                 "ok" if ok else "fail", sess, payload["chatId"])
        return ok, f"HTTP {resp.status_code}: {resp.text[:180]}"
    except Exception as e:  # noqa: BLE001
        log.warning("WAHA | send failed: %s", e)
        return False, str(e)[:200]
