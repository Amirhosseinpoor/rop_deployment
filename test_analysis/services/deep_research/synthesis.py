"""
A8 — Synthesis & Work-Fitness (one barrier call over all kept problems).

Produces the dossier's global layer — no health score, an editorial executive
brief instead:

  • brief          — 2–4 editorial Persian sentences framing the whole situation
  • recommendations— prioritized + categorized, **lifestyle first-class**
  • red_flags      — "seek care now" symptoms
  • referrals      — specialty + urgency (chatbot finds the doctor)
  • workfitness    — occupational fitness verdict + how problems compound

Uses the strongest tier (``SYNTH_MODEL``). Fully Persian.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List

from .tools import json_llm, SYNTH_MODEL

log = logging.getLogger("test_analysis.deep_research")

_SYS = """\
تو یک پزشک متخصص طب کار هستی که یافته‌های چند مشکل را کنار هم می‌گذاری و لایهٔ
کلانِ یک «گزارش تحقیق عمیق سلامت» را می‌نویسی. کاملاً فارسی. استناد [n] از فهرست
منابع اجباری است؛ عدد نساز. نمرهٔ سلامت نده.

بخش‌ها:
- «brief»: خلاصهٔ اجراییِ ۲ تا ۴ جمله‌ایِ ویراستارانه که کل وضعیت فرد را قاب می‌گیرد
  (بدون نمره)، با استناد [n].
- «recommendations»: توصیه‌های اولویت‌بندی‌شده و عملی. حتماً چند توصیهٔ «سبک زندگی»
  (تغذیه، فعالیت بدنی، خواب، ترک دخانیات، اصلاح ارگونومی) جداگانه بگذار. هر توصیه
  دسته و اولویت خود را مشخص کند.
- «workfitness»: معنای این یافته‌ها برای توانِ کاری و ادامهٔ شغلِ فرد، و این‌که
  مشکلات چگونه روی هم اثر می‌گذارند.
- برای ارجاع فقط تخصص و فوریت را نام ببر.

خروجی فقط JSON معتبر:
{
 "brief": "خلاصهٔ اجرایی چند جمله‌ای [n]",
 "recommendations": [
   {"priority":"urgent|important|suggested",
    "category":"سبک زندگی|دارویی|محیط کار|پیگیری تشخیصی",
    "icon":"emoji","text":"توصیهٔ مشخص و عملی [n]"}
 ],
 "red_flags": ["علامت خطری که نیاز به مراجعهٔ فوری دارد [n]"],
 "referrals": [{"specialty":"تخصص","urgency":"فوری|زودهنگام|روتین","reason":"... [n]"}],
 "workfitness": {"status":"مناسب|مناسب با ملاحظات|نیازمند بررسی",
                 "note":"جمع‌بندی توان کاری [n]",
                 "interactions":["اثر متقابل مشکلات بر یکدیگر [n]"]}
}"""


def synthesize(packet: Dict[str, Any], problems: List[Dict[str, Any]],
               eye_vision: Dict[str, Any] | None, context: str) -> Dict[str, Any]:
    problems_txt = "\n".join(
        f"• «{p.get('title')}» (شدت {p.get('severity')}, اطمینان {p.get('confidence','?')}): "
        f"{p.get('lead','')}"
        for p in problems
    ) or "(مشکلی تأیید نشد)"
    eye_txt = ""
    if eye_vision and eye_vision.get("note_fa"):
        eye_txt = (f"\n[بررسی تصویری ملتحمه]: رنگ‌پریدگی={eye_vision.get('pallor')}, "
                   f"کیفیت={eye_vision.get('image_quality')}, {eye_vision.get('note_fa')}")
    user = (
        f"[خلاصهٔ پرونده]:\n{packet['summary']}{eye_txt}\n\n"
        f"[مشکلات تأییدشده]:\n{problems_txt}\n\n"
        f"[منابع شماره‌دار]:\n{context[:13000] or '(فقط دادهٔ بیمار)'}\n\n"
        "لایهٔ کلانِ گزارش را طبق ساختار JSON بساز و همه‌جا [n] بده."
    )
    try:
        data = json_llm([{"role": "system", "content": _SYS}, {"role": "user", "content": user}],
                        model=SYNTH_MODEL, temperature=0.3)
        if not isinstance(data, dict):
            raise ValueError("synthesis did not return an object")
        return data
    except Exception as e:  # noqa: BLE001
        log.warning("DR | synthesis failed: %s", e)
        return {}
