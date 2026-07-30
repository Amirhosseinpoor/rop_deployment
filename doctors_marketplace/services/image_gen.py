# doctors_marketplace/services/image_gen.py
"""
Generate a realistic doctor profile photo via the GAPGPT (OpenAI-compatible)
image API, used by the Studio Copilot when it drafts a new assistant.

The model is read from the environment (``GAPGPT_IMAGE_MODEL``) — never
hard-coded — and the GAPGPT key/base URL are reused from the chat client config.
"""
from __future__ import annotations

import base64
import os
import uuid

import requests
from openai import OpenAI

from ..models import Doctor


def _env(*names, default=None):
    for n in names:
        v = os.getenv(n)
        if v:
            return v.strip().strip('"').strip("'")
    return default


API_KEY = _env("GAPGPT_API_KEY", "OPENAI_API_KEY")
BASE_URL = _env("GAPGPT_BASE_URL", "BASE_URL", default="https://api.gapgpt.app/v1")
IMAGE_MODEL = _env("GAPGPT_IMAGE_MODEL", default="gpt-image-1-mini")

# Map choice values → readable words for a natural-language image prompt.
_PERSONA_LOOK = {
    "kind": "warm, kind and approachable",
    "in_hurry": "confident and efficient",
    "calm": "calm, gentle and reassuring",
    "analytical": "thoughtful and attentive",
}


def build_face_prompt(name: str = "", specialization: str = "", persona: str = "") -> str:
    """Compose a photorealistic-headshot prompt from the drafted assistant fields."""
    spec_label = ""
    if specialization:
        spec_label = dict(Doctor.Specialization.choices).get(specialization, specialization)
        spec_label = spec_label.split("(")[0].strip()  # drop the parenthetical
    look = _PERSONA_LOOK.get(persona, "warm and professional")
    who = f"named {name} " if name else ""
    role = f"{spec_label} doctor" if spec_label else "doctor"
    return (
        f"A realistic, high-quality professional headshot portrait photograph of a "
        f"{role} {who}with a {look} expression, wearing a white medical coat, "
        f"a clean modern clinic softly blurred in the background, natural soft studio "
        f"lighting, looking friendly at the camera, photorealistic, sharp focus, "
        f"centered, head and shoulders, square composition. No text, no watermark."
    )


def _image_bytes_from_response(resp) -> bytes | None:
    """Handle both URL-based and base64-based image responses."""
    if not getattr(resp, "data", None):
        return None
    item = resp.data[0]
    b64 = getattr(item, "b64_json", None)
    if b64:
        return base64.b64decode(b64)
    url = getattr(item, "url", None)
    if url:
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        return r.content
    return None


def generate_doctor_face(name: str = "", specialization: str = "",
                         persona: str = "", size: str = "1024x1024") -> str | None:
    """
    Generate a doctor face and save it under MEDIA_ROOT/doctor_avatars/ai/.

    Returns the media URL (e.g. ``/media/doctor_avatars/ai/<uuid>.png``) or None
    on failure. Never raises — the copilot draft should succeed regardless.
    """
    if not API_KEY:
        return None
    try:
        from django.conf import settings

        client = OpenAI(api_key=API_KEY, base_url=BASE_URL, timeout=120)
        resp = client.images.generate(
            model=IMAGE_MODEL,
            prompt=build_face_prompt(name, specialization, persona),
            size=size,
        )
        data = _image_bytes_from_response(resp)
        if not data:
            return None

        rel_dir = os.path.join("doctor_avatars", "ai")
        abs_dir = os.path.join(settings.MEDIA_ROOT, rel_dir)
        os.makedirs(abs_dir, exist_ok=True)
        fname = f"{uuid.uuid4().hex}.png"
        with open(os.path.join(abs_dir, fname), "wb") as f:
            f.write(data)

        media_url = settings.MEDIA_URL.rstrip("/")
        return f"{media_url}/{rel_dir.replace(os.sep, '/')}/{fname}"
    except Exception:  # noqa: BLE001 - a failed avatar must not break the draft
        return None
