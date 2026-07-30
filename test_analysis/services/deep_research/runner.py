"""
Deep Health Research v2 — background orchestration.

Runs the whole agentic pipeline in a daemon thread so it works under plain
``python manage.py runserver`` (no Celery needed for DR itself). Progress is
written to Django's cache and polled by the page; the finished report is stored
as JSON in ``HealthProfile.llm_advice`` behind the ``#DRV1#`` sentinel — no DB
migration. On completion it also flips ``report_ready`` so the existing
processing page (which polls ``report_status``) redirects to the detail view.

Pipeline order (see deepresearch.md):
  0 collect   — packet (deterministic) + eye-vision (VLM)   [waits for eye/lab
                Celery pipelines to finish first, bounded]
  1 triage    — rank cross-modal problems
  2 questions — per-problem patient-anchored research questions (+ lifestyle)
  3 research  — KB + web fan-out → one numbered source list
  4 author    — dual-register grounded problem cards (parallel)
  5 verify    — 3-vote adversarial panel
  6 synthesis — plain summary, recommendations (+lifestyle), red-flags, referrals,
                work-fitness
  7 blocks    — typed block list
  8 persist
"""
from __future__ import annotations

import concurrent.futures as cf
import json
import logging
import os
import threading
from typing import Any, Dict, Optional

from django.core.cache import cache
from django.utils import timezone

from .packet import build_evidence_packet
from . import eye_vision as A1
from . import triage as A2
from . import questions as A3
from . import research as A4
from . import author as A5
from . import verify as A6
from . import critic as A7
from . import synthesis as A8
from . import blocks as A9

log = logging.getLogger("test_analysis.deep_research")

SENTINEL = "#DRV1#\n"
_TTL = 60 * 30
DR_FANOUT_WORKERS = int(os.getenv("DR_FANOUT_WORKERS", "4"))
DR_CRITIC_ROUNDS = int(os.getenv("DR_CRITIC_ROUNDS", "1"))


def _pkey(profile_id: int) -> str:
    return f"deepresearch:progress:{profile_id}"


def set_progress(profile_id: int, *, state: str, stage: str = "", pct: int = 0,
                 label: str = "", error: str = "") -> None:
    cache.set(_pkey(profile_id),
              {"state": state, "stage": stage, "pct": pct, "label": label, "error": error},
              _TTL)


def get_progress(profile_id: int) -> Optional[Dict[str, Any]]:
    return cache.get(_pkey(profile_id))


# --------------------------------------------------------------------------- #
# run the eye-screening + medical-test extraction ourselves, in this thread,
# BEFORE building the packet — so segmentation, the anemia classifier and the
# lab-table extraction are all guaranteed complete before we raise report_ready.
# No Celery worker needed: report + segmentation + classification + extraction
# finish together, so the detail page always has them on redirect.
# --------------------------------------------------------------------------- #
def _process_pending_pipelines(profile_id: int) -> None:
    from django.db import connection
    from test_analysis.models import HealthProfile, MedicalTest
    from test_analysis.services.eye_pipeline import analyze_eye_image
    from test_analysis.services.medical_test_extraction import extract_medical_test

    profile = HealthProfile.objects.get(id=profile_id)
    eye_todo = [ei for ei in profile.eye_images.all()
                if not (getattr(ei, "analysis", None) and ei.analysis.status == "done")]
    mt_todo = list(profile.medical_tests.exclude(status=MedicalTest.STATUS_DONE))

    if not eye_todo and not mt_todo:
        return
    log.info("DR | processing pipelines in-thread for profile=%s | %d eye, %d test(s)",
             profile_id, len(eye_todo), len(mt_todo))

    # Eye screening sequentially — the torch segmentation models are lazily built
    # into a shared cache and are not safe to build from several threads at once.
    for i, ei in enumerate(eye_todo, 1):
        set_progress(profile_id, state="running", stage="collecting",
                     pct=6 + int(6 * i / max(1, len(eye_todo))),
                     label=f"تحلیل تصویر چشم ({i}/{len(eye_todo)}) — بخش‌بندی و طبقه‌بندی")
        try:
            analyze_eye_image(ei)  # best-effort, never raises
        except Exception as e:  # noqa: BLE001
            log.warning("DR | eye analysis failed for image=%s: %s", ei.id, e)

    # Medical-test extraction in parallel (I/O-bound LLM calls). Each worker
    # thread gets its own DB connection and must close it.
    if mt_todo:
        set_progress(profile_id, state="running", stage="collecting", pct=12,
                     label="استخراج نتایج آزمایش‌ها با هوش مصنوعی")

        def _extract(mt):
            try:
                extract_medical_test(mt)  # best-effort, never raises
            finally:
                connection.close()

        with cf.ThreadPoolExecutor(max_workers=min(3, len(mt_todo))) as ex:
            list(ex.map(_extract, mt_todo))


# --------------------------------------------------------------------------- #
# thread worker
# --------------------------------------------------------------------------- #
def _author_all(packet, problems, questions_by_problem, context):
    """Author a set of problems in parallel; preserve their input order."""
    authored: list[Dict[str, Any]] = []
    if not problems:
        return authored
    with cf.ThreadPoolExecutor(max_workers=DR_FANOUT_WORKERS) as ex:
        futs = {ex.submit(A5.author_problem, packet, p,
                          questions_by_problem.get(p["title"], []), context): p
                for p in problems}
        for fut in cf.as_completed(futs):
            res = fut.result()
            if res:
                authored.append(res)
    order = {p["title"]: i for i, p in enumerate(problems)}
    authored.sort(key=lambda a: order.get(a.get("title"), 99))
    return authored


def _worker(profile_id: int) -> None:
    from django.db import connection
    from test_analysis.models import HealthProfile
    try:
        set_progress(profile_id, state="running", stage="collecting", pct=5,
                     label="تحلیل تصویر چشم و استخراج آزمایش‌ها")
        # Screen eyes + extract labs in this thread so they finish before the
        # report — report_ready is raised only when everything is complete.
        _process_pending_pipelines(profile_id)

        profile = HealthProfile.objects.get(id=profile_id)
        packet = build_evidence_packet(profile)

        # Stage 0b — eye vision (VLM looks at the actual conjunctiva crop)
        set_progress(profile_id, state="running", stage="collecting", pct=12,
                     label="بررسی تصویری ملتحمهٔ چشم با هوش مصنوعی بینایی")
        eye_v = A1.analyze_eye(packet)

        # Stage 1 — triage
        set_progress(profile_id, state="running", stage="triage", pct=20,
                     label="شناسایی و رتبه‌بندی مشکلات از روی داده‌ها")
        problems = A2.triage(packet, eye_v)

        # Stage 2 — questions
        set_progress(profile_id, state="running", stage="questions", pct=30,
                     label="طراحی پرسش‌های پژوهشی مخصوص این فرد")
        questions_by_problem = A3.generate_questions(packet, problems)

        # Stage 3 — multi-channel research (KB + web + biomedical literature)
        set_progress(profile_id, state="running", stage="searching", pct=44,
                     label="جست‌وجوی رهنمودها، مقالات علمی و دانش‌نامهٔ داخلی")
        sources, context, src_counts = A4.gather_sources(packet, problems, questions_by_problem)

        # Stage 4 — author (parallel per problem)
        set_progress(profile_id, state="running", stage="analyzing", pct=60,
                     label="نگارش تحلیل بالینیِ مستند برای هر یافته")
        authored = _author_all(packet, problems, questions_by_problem, context)

        # Stage 5 — verify (3-lens adversarial panel)
        set_progress(profile_id, state="running", stage="verifying", pct=74,
                     label="راستی‌آزمایی ادعاها با سه داور مستقل")
        authored = A6.verify(authored, context)

        # Stage 6 — completeness critic → optional extra research+author round
        n_gap_rounds = 0
        if DR_CRITIC_ROUNDS > 0 and authored:
            set_progress(profile_id, state="running", stage="critic", pct=82,
                         label="بازبینی کاستی‌ها و پژوهش تکمیلی")
            gaps = A7.find_gaps(packet, authored, max_gaps=2)
            if gaps:
                n_gap_rounds = 1
                gap_qs = A3.generate_questions(packet, gaps)
                gap_sources, gap_context, gap_counts = A4.gather_sources(packet, gaps, gap_qs)
                # extend the unified source list (renumber the new ones)
                base = len(sources)
                for s in gap_sources:
                    if s["type"] != "patient":  # patient facts already present
                        s = dict(s); s["id"] = len(sources) + 1; sources.append(s)
                for k in src_counts:
                    src_counts[k] = src_counts.get(k, 0) + gap_counts.get(k, 0)
                gap_authored = _author_all(packet, gaps, gap_qs, context + "\n---\n" + gap_context)
                gap_authored = A6.verify(gap_authored, gap_context)
                authored += gap_authored

        # Stage 7 — synthesis (executive brief, recommendations, work-fitness — no score)
        set_progress(profile_id, state="running", stage="synthesizing", pct=90,
                     label="جمع‌بندی، توصیه‌های سبک زندگی و توان کاری")
        synth = A8.synthesize(packet, authored, eye_v, context)

        # Stage 8 — assemble dossier + research trail
        trail = {
            "n_findings": len(authored),
            "n_questions": sum(len(v) for v in questions_by_problem.values()),
            "sources": src_counts,
            "reviewers": 3,
            "critic_rounds": n_gap_rounds,
            "eye_vision": bool(eye_v),
        }
        report = A9.assemble(packet, authored, synth, eye_v, sources, trail)

        # Stage 9 — persist
        profile.llm_advice = SENTINEL + json.dumps(report, ensure_ascii=False)
        profile.report_ready = True
        profile.report_error = None
        profile.save(update_fields=["llm_advice", "report_ready", "report_error"])
        set_progress(profile_id, state="done", stage="done", pct=100, label="گزارش آماده شد")
        log.info("DR | v3 dossier ready for profile=%s | %d block(s), %d source(s), %d finding(s), %d gap-round(s)",
                 profile_id, len(report["blocks"]), len(report["sources"]), len(authored), n_gap_rounds)
    except Exception as e:  # noqa: BLE001
        log.exception("DR | pipeline failed for profile=%s: %s", profile_id, e)
        set_progress(profile_id, state="error", label="خطا در تولید گزارش", error=str(e))
        try:
            from test_analysis.models import HealthProfile
            p = HealthProfile.objects.filter(id=profile_id).first()
            if p:
                p.report_ready = False
                p.report_error = str(e)
                p.save(update_fields=["report_ready", "report_error"])
        except Exception:  # noqa: BLE001
            pass
    finally:
        connection.close()


def start(profile_id: int) -> Dict[str, Any]:
    """Kick off generation unless one is already running for this profile."""
    prog = get_progress(profile_id)
    if prog and prog.get("state") == "running":
        return prog
    set_progress(profile_id, state="running", stage="starting", pct=3,
                 label="در حال آماده‌سازی…")
    threading.Thread(target=_worker, args=(profile_id,), daemon=True).start()
    return get_progress(profile_id)


def load_report(profile) -> Optional[Dict[str, Any]]:
    raw = profile.llm_advice or ""
    if not raw.startswith(SENTINEL):
        return None
    try:
        return json.loads(raw[len(SENTINEL):])
    except Exception as e:  # noqa: BLE001
        log.warning("DR | could not parse stored report for profile=%s: %s", profile.id, e)
        return None
