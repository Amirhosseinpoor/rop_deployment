"""
A5 — Clinical Author (one focused LLM call per problem, run in parallel).

Writes the deep, grounded dossier entry for a single problem in ONE clinical
register — precise, doctor-grade Persian (no plain-language duplicate). Every
factual sentence carries an ``[n]`` citation into the unified numbered source
list; numbers are never invented.

Uses the strongest model tier (``SYNTH_MODEL``).
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List

from .tools import json_llm, SYNTH_MODEL

log = logging.getLogger("test_analysis.deep_research")

_SYS = """\
تو یک پزشک متخصص طب کار و داخلی هستی و برای یک «مشکلِ سلامتِ مشخص» یک تحلیلِ عمیق،
دقیق و کاملاً مستند به زبان فارسی می‌نویسی. لحن حرفه‌ایِ بالینی اما روشن و قابل‌فهم؛
از اصطلاح تخصصی استفاده کن و در صورت لزوم در همان جمله کوتاه توضیحش بده.

استناد (اجباری و سخت‌گیرانه):
- هر جمله‌ای که واقعیت، عدد، یافته، سازوکار یا توصیه‌ای را بیان می‌کند باید در
  انتها نشانگر منبع [n] داشته باشد؛ n فقط از فهرست [منابع] انتخاب شود.
- منابعِ «مطالعه» (مقالات PubMed/Europe PMC) و «رهنمود» را در ادعاهای علمی ترجیح
  بده و شمارهٔ آن‌ها را بیاور.
- عدد جدید نساز؛ فقط از اعداد موجود استفاده کن. اگر داده کافی نیست، صریح بنویس
  «داده کافی نیست» و به مشورت با تیم درمان ارجاع بده.

محتوا:
- کاملاً فارسی. برای دارو یا محل مراجعه توصیه نکن (کارِ دستیار گفتگوست) — فقط تخصص
  و ضرورت پیگیری را ذکر کن.
- «تشخیص افتراقی» را واقعی و مرتبط با همین فرد بنویس (چند احتمال با دلیل).
- «سازوکار و ارتباط شغلی» را با مواجهه‌های شغلیِ همین فرد گره بزن.

خروجی فقط JSON معتبر:
{
 "title":"عنوان مشکل",
 "severity":"high|medium|low",
 "category":"...",
 "lead":"یک جملهٔ سرآمد که یافته را قاب می‌گیرد [n]",
 "sections":[
   {"h":"تعریف و یافته","body":"... [n]"},
   {"h":"تشخیص افتراقی","body":"... [n]"},
   {"h":"سازوکار و ارتباط شغلی","body":"... [n]"},
   {"h":"خطر در صورت بی‌توجهی","body":"... [n]"},
   {"h":"اقدام تشخیصی و درمانی بعدی","body":"... [n]"}
 ],
 "evidence_refs":["Sx"],
 "key_citations":[n]
}"""


def author_problem(packet: Dict[str, Any], problem: Dict[str, Any],
                   questions: List[Dict[str, Any]], context: str) -> Dict[str, Any] | None:
    q_txt = "\n".join(f"- {q['q_fa']}" for q in questions) or "—"
    user = (
        f"[مشکل]: {problem['title']} (شدت: {problem['severity']}، دسته: {problem['category']})\n"
        f"[سازوکار محتمل]: {problem.get('mechanism_hint','—')}\n\n"
        f"[پرسش‌های پژوهشی این مشکل]:\n{q_txt}\n\n"
        f"[خلاصهٔ پرونده]:\n{packet['summary']}\n\n"
        f"[منابع شماره‌دار]:\n{context or '(فقط به دادهٔ بیمار استناد کن)'}\n\n"
        "تحلیل بالینیِ این مشکل را طبق ساختار JSON و با استناد [n] بنویس."
    )
    try:
        data = json_llm(
            [{"role": "system", "content": _SYS}, {"role": "user", "content": user}],
            model=SYNTH_MODEL, temperature=0.32,
        )
        if not isinstance(data, dict) or not data.get("sections"):
            return None
        data.setdefault("title", problem["title"])
        data.setdefault("severity", problem["severity"])
        data.setdefault("category", problem["category"])
        # keep only well-formed sections
        data["sections"] = [s for s in data.get("sections", [])
                            if isinstance(s, dict) and s.get("h") and s.get("body")]
        return data if data["sections"] else None
    except Exception as e:  # noqa: BLE001
        log.warning("DR | author failed for %r: %s", problem.get("title"), e)
        return None
