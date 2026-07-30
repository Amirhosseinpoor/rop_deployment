"""
A3 — Question Strategist  ⭐ (the depth lever).

Instead of jumping straight to a web query, this agent turns each detected
problem into several *specific, patient-anchored* research questions built from
the form, the medical tests, and the eye classification. Each question names the
exact patient datum (``Sx``) it is chasing, so the research that follows is about
*this* person — not a generic template.

Question archetypes it must cover (per problem, whichever apply):
  interpretation · occupational · corroboration · workup · risk · threshold
  · **lifestyle** (diet/activity/sleep/smoking/ergonomic self-care — requested
    explicitly so the report always gives actionable lifestyle guidance)

One batched call for all problems keeps latency + cost low; near-duplicate
questions are merged downstream before spending web calls.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List

from .tools import json_llm, REASONING_MODEL

log = logging.getLogger("test_analysis.deep_research")

QUESTIONS_PER_PROBLEM = 4

_SYS = """\
تو یک راهبرِ پژوهش بالینی هستی. برای هر «مشکل» شناسایی‌شده، چند پرسش پژوهشیِ دقیق
طراحی کن که پاسخشان گزارش را عمیق و مخصوصِ همین فرد کند.

قوانین:
- هر پرسش باید به دادهٔ مشخصِ بیمار (Sx) لنگر بخورد؛ پرسش کلی و بی‌نام ممنوع.
- برای هر مشکل حداکثر {qpp} پرسش. تنوع نوع را رعایت کن.
- انواع پرسش:
  * interpretation  — این عدد/یافته برای همین فرد چه معنایی دارد؟
  * occupational    — مواجههٔ شغلی فرد چگونه با این یافته مرتبط است؟
  * corroboration   — آیا چند داده (چشم/آزمایش/معاینه) یکدیگر را تأیید می‌کنند؟
  * workup          — چه آزمایش/اقدام بعدی برای قطعی‌شدن لازم است؟
  * risk            — اگر رها شود مسیر بیماری برای این فرد چیست؟
  * threshold       — آستانه/هدفِ رهنمودهای روزآمد (۲۰۲۴–۲۰۲۵) برای اعداد او؟
  * lifestyle       — تغییرات سبک زندگی (تغذیه، فعالیت، خواب، ترک دخانیات،
                      اصلاح ارگونومی) که برای همین فرد مؤثر است.
- «search_en» یک عبارت جست‌وجوی وبِ کوتاه و انگلیسی برای یافتن رهنمود روزآمد.
- «lit_query» یک عبارت انگلیسیِ زیست‌پزشکی برای جست‌وجوی مقالات (PubMed / Europe
  PMC) — با واژگان تخصصی (مثلاً «occupational lead exposure anemia mechanism»).
- حداقل یک پرسش از نوع lifestyle برای هر مشکل بگذار.

خروجی فقط JSON معتبر:
{{"per_problem":[
  {{"title":"<دقیقاً همان عنوان مشکل ورودی>",
    "questions":[
      {{"q_fa":"پرسش فارسی دقیق و لنگرزده",
        "search_en":"concise english web query",
        "lit_query":"biomedical literature query",
        "type":"interpretation|occupational|corroboration|workup|risk|threshold|lifestyle",
        "targets":["Sx"]}}
    ]}}
]}}"""


def generate_questions(packet: Dict[str, Any],
                       problems: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    if not problems:
        return {}
    problems_txt = "\n".join(
        f"- «{p['title']}» (شدت: {p['severity']}، دسته: {p['category']}، "
        f"شواهد: {', '.join(p['evidence']) or '—'}، سازوکار: {p.get('mechanism_hint','—')})"
        for p in problems
    )
    user = (
        f"[خلاصهٔ پروندهٔ بیمار]:\n{packet['summary']}\n\n"
        f"[مشکلات شناسایی‌شده]:\n{problems_txt}\n\n"
        "برای هر مشکل، پرسش‌های پژوهشی را طبق ساختار JSON طراحی کن."
    )
    msgs = [
        {"role": "system", "content": _SYS.format(qpp=QUESTIONS_PER_PROBLEM)},
        {"role": "user", "content": user},
    ]
    out: Dict[str, List[Dict[str, Any]]] = {}
    try:
        data = json_llm(msgs, model=REASONING_MODEL, temperature=0.3)
        for entry in (data.get("per_problem", []) if isinstance(data, dict) else []):
            title = str(entry.get("title", "")).strip()
            if not title:
                continue
            qs: List[Dict[str, Any]] = []
            for q in (entry.get("questions") or [])[:QUESTIONS_PER_PROBLEM]:
                if not isinstance(q, dict) or not q.get("q_fa"):
                    continue
                qs.append({
                    "q_fa": str(q["q_fa"]).strip(),
                    "search_en": str(q.get("search_en", "")).strip(),
                    "lit_query": str(q.get("lit_query", "")).strip(),
                    "type": str(q.get("type", "interpretation")).strip(),
                    "targets": [str(t) for t in (q.get("targets") or [])],
                })
            if qs:
                out[title] = qs
        log.info("DR | questions → %d problem(s), %d question(s) total",
                 len(out), sum(len(v) for v in out.values()))
    except Exception as e:  # noqa: BLE001
        log.warning("DR | question generation failed: %s", e)
    return out
