"""
A6 — Adversarial Verifier panel (3 votes, distinct lenses).

Each authored problem is judged by three skeptics, each with a *different* reason
to reject it:

  lens 1  patient-data support  — do the employee's own numbers back this up?
  lens 2  cited-source support  — does a real [n] source say this?
  lens 3  contradiction/conflict — is it contradicted by a medication or another
                                   finding, or overstated?

The three lenses run concurrently (one batched call each covering all problems).
A problem is kept if a majority (≥2) vote to keep; its confidence is the mean of
the votes. Kept-but-shaky problems get downgraded rather than silently trusted.
"""
from __future__ import annotations

import concurrent.futures as cf
import logging
from typing import Any, Dict, List

from .tools import json_llm, REASONING_MODEL

log = logging.getLogger("test_analysis.deep_research")

_LENSES = {
    "patient": "آیا ادعاهای این مشکل با «دادهٔ خودِ بیمار» (اعداد و یافته‌های او) پشتیبانی می‌شود؟",
    "source": "آیا ادعاهای این مشکل به یک «منبع واقعی [n]» در فهرست منابع تکیه دارد؟",
    "conflict": "آیا این مشکل با دارو، یافتهٔ دیگر یا واقعیتی در پرونده در تضاد است یا بیش از حد بزرگ‌نمایی شده؟",
}

_SYS_TMPL = """\
تو یک بازبینِ منتقد و سخت‌گیر هستی. تنها از یک زاویه قضاوت کن:
«{lens}»
برای هر مشکل تصمیم بگیر که با این معیار نگه داشته شود یا نه.

خروجی فقط JSON:
{{"verdicts":[{{"title":"...","keep":true|false,"confidence":0.0..1.0,"issue":"اگر ایرادی هست کوتاه بگو"}}]}}"""


def _judge(lens_key: str, problems_brief: str, context: str) -> Dict[str, Dict[str, Any]]:
    sys = _SYS_TMPL.format(lens=_LENSES[lens_key])
    user = f"[مشکلات و متن آن‌ها]:\n{problems_brief}\n\n[منابع]:\n{context[:9000]}"
    try:
        data = json_llm([{"role": "system", "content": sys}, {"role": "user", "content": user}],
                        model=REASONING_MODEL, temperature=0.1)
        out: Dict[str, Dict[str, Any]] = {}
        for v in (data.get("verdicts", []) if isinstance(data, dict) else []):
            if isinstance(v, dict) and v.get("title"):
                out[str(v["title"]).strip()] = v
        return out
    except Exception as e:  # noqa: BLE001
        log.warning("DR | verify lens %s failed: %s", lens_key, e)
        return {}


def verify(problems: List[Dict[str, Any]], context: str) -> List[Dict[str, Any]]:
    if not problems:
        return problems

    def _brief(p: Dict[str, Any]) -> str:
        bodies = " ".join(s.get("clinical", "")[:300] for s in p.get("sections", []))
        return f"• «{p.get('title')}»: {bodies}"

    brief = "\n".join(_brief(p) for p in problems)

    votes_by_lens: List[Dict[str, Dict[str, Any]]] = []
    with cf.ThreadPoolExecutor(max_workers=3) as ex:
        futures = {ex.submit(_judge, lens, brief, context): lens for lens in _LENSES}
        for fut in cf.as_completed(futures):
            votes_by_lens.append(fut.result())

    kept: List[Dict[str, Any]] = []
    for p in problems:
        title = p.get("title")
        votes = [lens.get(title) for lens in votes_by_lens if lens.get(title)]
        if not votes:
            p["confidence"] = 0.75
            kept.append(p)
            continue
        keeps = sum(1 for v in votes if v.get("keep") is not False)
        confs = [float(v.get("confidence", 0.7)) for v in votes if v.get("confidence") is not None]
        conf = round(sum(confs) / len(confs), 2) if confs else 0.7
        if keeps * 2 < len(votes):  # majority say drop
            issue = next((v.get("issue") for v in votes if v.get("keep") is False and v.get("issue")), "")
            log.info("DR | verify dropped %r (%s)", title, issue)
            continue
        p["confidence"] = conf
        if conf < 0.5:
            p["caveat_fa"] = "این یافته نیازمند بررسی بیشتر است."
        kept.append(p)

    return kept or problems  # never blank the whole report
