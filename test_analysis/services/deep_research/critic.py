"""
A7 — Completeness Critic.

Reads the kept dossier + the Evidence Packet and asks the question a senior
reviewer would: *what's missing?* — a lab panel or exam finding never discussed,
a claim left unverified, a guideline threshold not pinned down, an occupational
exposure whose health link was ignored.

It returns up to ``max_gaps`` gap "problems" in the same shape the triage agent
produces, so the runner can push each straight back through
questions → research → author → verify for one more round. Bounded by
``DR_CRITIC_ROUNDS`` in the runner.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List

from .tools import json_llm, SYNTH_MODEL

log = logging.getLogger("test_analysis.deep_research")

_SYS = """\
تو یک بازبینِ ارشدِ بالینی هستی. یک «پروندهٔ سلامت» و «مشکلات پوشش‌داده‌شده» به تو
داده می‌شود. فقط کاستی‌های مهم را پیدا کن: داده یا یافته‌ای که اصلاً بررسی نشده،
مواجههٔ شغلی‌ای که ارتباط سلامتی‌اش نادیده مانده، یا یک ریسکِ مهمِ جامانده.

قوانین:
- فقط کاستیِ واقعی و مهم را گزارش کن؛ اگر چیزی جا نمانده، فهرست خالی برگردان.
- برای هر کاستی یک «مشکل» جدید در همان قالب تعریف کن تا در دور بعد پژوهش شود.
- حداکثر {maxg} کاستی.

خروجی فقط JSON معتبر:
{{"gaps":[
  {{"title":"عنوان فارسی مشکلِ جامانده",
    "category":"متابولیک|قلبی-عروقی|خونی|تنفسی/شغلی|اسکلتی-عضلانی|روانی|کلیوی-کبدی|سایر",
    "severity":"high|medium|low",
    "evidence":["Sx"],
    "mechanism_hint":"عبارت کوتاه",
    "reason":"چرا این مهم است و چرا جا مانده"}}
]}}"""


def find_gaps(packet: Dict[str, Any], problems: List[Dict[str, Any]],
              max_gaps: int = 2) -> List[Dict[str, Any]]:
    covered = "، ".join(p.get("title", "") for p in problems) or "(هیچ)"
    user = (
        f"[خلاصهٔ پرونده]:\n{packet['summary']}\n\n"
        f"[مشکلاتِ پوشش‌داده‌شده]: {covered}\n\n"
        "کاستی‌های مهمِ پوشش‌نداده‌شده را طبق ساختار JSON گزارش کن."
    )
    try:
        data = json_llm([{"role": "system", "content": _SYS.format(maxg=max_gaps)},
                         {"role": "user", "content": user}],
                        model=SYNTH_MODEL, temperature=0.2)
        gaps = data.get("gaps", []) if isinstance(data, dict) else []
        out: List[Dict[str, Any]] = []
        existing = {p.get("title", "").strip() for p in problems}
        for g in gaps[:max_gaps]:
            if not isinstance(g, dict) or not g.get("title"):
                continue
            title = str(g["title"]).strip()
            if title in existing:
                continue
            sev = g.get("severity")
            out.append({
                "title": title,
                "category": str(g.get("category", "سایر")).strip(),
                "severity": sev if sev in ("high", "medium", "low") else "medium",
                "evidence": [str(e) for e in (g.get("evidence") or [])],
                "mechanism_hint": str(g.get("mechanism_hint", "")).strip(),
            })
        if out:
            log.info("DR | critic found %d gap(s): %s", len(out), [g["title"] for g in out])
        return out
    except Exception as e:  # noqa: BLE001
        log.warning("DR | critic failed: %s", e)
        return []
