"""
A2 — Occupational-Medicine Triage.

Reads the deterministic Evidence Packet summary (+ the eye-vision finding) and
returns a ranked list of the employee's most important health problems. The
prompt forces *cross-modal* thinking: a finding that shows up in the eye AI **and**
a lab flag **and** an occupational exposure is one stronger problem, not three
weak ones.

No arithmetic — every number is already computed in the packet and tagged with a
stable ``S``-id the model must reference in ``evidence``.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List

from .tools import json_llm, REASONING_MODEL

log = logging.getLogger("test_analysis.deep_research")

MAX_PROBLEMS = 6

_SYS = """\
تو یک پزشک متخصص طب کار و داخلی هستی. از روی «پروندهٔ سلامت» یک کارمند،
مهم‌ترین مشکلات و ریسک‌های سلامت او را شناسایی و رتبه‌بندی کن.

قوانین سخت‌گیرانه:
- فقط بر اساس داده‌های ارائه‌شده تصمیم بگیر؛ هیچ عدد یا یافته‌ای از خودت نساز.
- چندوجهی فکر کن: ترکیب چند شاهد (مثلاً کم‌خونی در تصویر چشم + هموگلوبین پایین +
  مواجههٔ شغلی با حلال‌ها) یک مشکلِ واحدِ قوی‌تر است، نه چند مشکل جدا.
- هر مشکل را به شناسه‌های منبع (Sx) که در پرونده آمده گره بزن.
- «mechanism_hint» یک عبارت کوتاه دربارهٔ سازوکار زیستی/شغلیِ محتمل است تا مرحلهٔ
  پژوهش بداند دنبال چه بگردد.
- حداکثر {maxp} مشکلِ واقعاً مهم را برگردان (کیفیت مهم‌تر از کمیت).

خروجی فقط JSON معتبر:
{{"problems":[
  {{"title":"عنوان فارسی مشکل",
    "category":"متابولیک|قلبی-عروقی|خونی|تنفسی/شغلی|اسکلتی-عضلانی|روانی|پوستی|کلیوی-کبدی|سایر",
    "severity":"high|medium|low",
    "evidence":["Sx","Sy"],
    "mechanism_hint":"عبارت کوتاه انگلیسی یا فارسی از سازوکار محتمل"}}
]}}"""


def triage(packet: Dict[str, Any], eye_vision: Dict[str, Any] | None = None) -> List[Dict[str, Any]]:
    summary = packet["summary"]
    if eye_vision and eye_vision.get("note_fa"):
        summary += (f"\n— بررسی تصویری ملتحمه (هوش مصنوعی بینایی): "
                    f"رنگ‌پریدگی={eye_vision.get('pallor')}, {eye_vision.get('note_fa')}")
    msgs = [
        {"role": "system", "content": _SYS.format(maxp=MAX_PROBLEMS)},
        {"role": "user", "content": "پروندهٔ سلامت:\n" + summary},
    ]
    try:
        data = json_llm(msgs, model=REASONING_MODEL, temperature=0.2)
        problems = data.get("problems", []) if isinstance(data, dict) else data
        out: List[Dict[str, Any]] = []
        for p in problems[:MAX_PROBLEMS]:
            if not isinstance(p, dict) or not p.get("title"):
                continue
            sev = p.get("severity")
            out.append({
                "title": str(p["title"]).strip(),
                "category": str(p.get("category", "سایر")).strip(),
                "severity": sev if sev in ("high", "medium", "low") else "medium",
                "evidence": [str(e) for e in (p.get("evidence") or [])],
                "mechanism_hint": str(p.get("mechanism_hint", "")).strip(),
            })
        log.info("DR | triage → %d problem(s): %s", len(out), [p["title"] for p in out])
        return out
    except Exception as e:  # noqa: BLE001
        log.warning("DR | triage failed: %s", e)
        return []
