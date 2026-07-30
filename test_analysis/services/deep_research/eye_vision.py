"""
A1 — Eye Vision Analyst.

Looks at the *actual* conjunctiva crop the segmentation model produced (plus the
original photo) with a vision-language model and describes the clinical signs a
human would look for in suspected anaemia: pallor of the palpebral conjunctiva,
visible vascularity, and whether the image is even good enough to judge.

The result is fused with the classifier's label + confidence and (in the packet)
the lab haemoglobin, so the report can say "the model, the image, and the blood
test agree" — a claim a generic chatbot literally cannot make because it never
sees these pixels.

Best-effort: any failure returns ``None`` and the pipeline continues without it.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from .tools import vision_read

log = logging.getLogger("test_analysis.deep_research")

_SYS = """\
تو یک دستیار چشم‌پزشکی هستی که فقط بر پایهٔ آنچه در تصویر می‌بینی داوری می‌کنی.
دو تصویر دریافت می‌کنی: (۱) عکس اصلی چشم، (۲) برشِ ملتحمهٔ پلکی/فورنیکس که مدل
بخش‌بندی جدا کرده است (پس‌زمینه سیاه است). رنگ و روشنی ملتحمه را ارزیابی کن.
هرگز چیزی فراتر از تصویر ادعا نکن؛ اگر کیفیت پایین است صادقانه بگو.

فقط JSON معتبر با این ساختار بده:
{
 "pallor": "none|mild|marked|uncertain",
 "pallor_fa": "توصیف کوتاه فارسی از رنگ‌پریدگی ملتحمه",
 "vascularity": "normal|reduced|uncertain",
 "image_quality": "good|fair|poor",
 "note_fa": "یک تا دو جملهٔ فارسی از آنچه در تصویر دیده می‌شود (بدون تشخیص قطعی)"
}"""

_USER = ("این دو تصویر چشم را بررسی کن (اولی عکس اصلی، دومی برش ملتحمه). "
         "رنگ‌پریدگی ملتحمه را از نظر نشانه‌های احتمالی کم‌خونی ارزیابی کن و "
         "طبق ساختار JSON پاسخ بده.")


def analyze_eye(packet: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    eye = packet.get("eye")
    if not eye:
        return None
    paths = [p for p in (eye.get("original_path"), eye.get("phase1_path")) if p]
    if not paths:
        return None
    try:
        v = vision_read(paths, _SYS, _USER)
    except Exception as e:  # noqa: BLE001
        log.warning("DR | eye vision failed: %s", e)
        return None

    pallor = str(v.get("pallor", "uncertain")).lower()
    model_positive = eye.get("label") == "positive"
    vision_positive = pallor in ("mild", "marked")
    agrees = (model_positive == vision_positive) if pallor != "uncertain" else None

    out = {
        "pallor": pallor,
        "pallor_fa": str(v.get("pallor_fa", "")).strip(),
        "vascularity": str(v.get("vascularity", "uncertain")).lower(),
        "image_quality": str(v.get("image_quality", "fair")).lower(),
        "note_fa": str(v.get("note_fa", "")).strip(),
        "agrees_with_model": agrees,
    }
    log.info("DR | eye vision → pallor=%s quality=%s agrees_model=%s",
             out["pallor"], out["image_quality"], agrees)
    return out
