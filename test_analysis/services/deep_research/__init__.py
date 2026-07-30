"""Deep Health Research v2 — agentic, multimodal, grounded health report.

Public surface:
    build_evidence_packet(profile)  — deterministic Stage 0 (Evidence Packet)
    runner.start(profile_id)        — background generation (daemon thread, no Celery)
    runner.get_progress(profile_id) — poll live status
    runner.load_report(profile)     — parse the stored JSON report

Pipeline stages live in dedicated modules (see deepresearch.md):
    eye_vision (A1) · triage (A2) · questions (A3) · research (A4) ·
    author (A5) · verify (A6) · synthesis (A7) · blocks (A8)
Tools the agents use are in ``tools/`` (json_llm, vision_read, web/kb retrieval).
"""
from .packet import build_evidence_packet
from . import runner

__all__ = ["build_evidence_packet", "runner"]
