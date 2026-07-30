"""
A9 — Dossier Composer: verified findings + packet → the ordered editorial block
list the front-end renders as a clinical research dossier. No health score.

Block types:
  masthead · brief · keyvitals · figure_eye · finding(×N) · recommendations ·
  redflags · workfitness · referrals · labs_pointer · references · trail
(references + trail are rendered from the top-level `sources` / `trail`.)
"""
from __future__ import annotations

from typing import Any, Dict, List

from django.utils import timezone

_SEV_ORDER = {"high": 0, "medium": 1, "low": 2}


def assemble(packet: Dict[str, Any], problems: List[Dict[str, Any]],
             synthesis: Dict[str, Any], eye_vision: Dict[str, Any] | None,
             sources: List[Dict[str, Any]], trail: Dict[str, Any]) -> Dict[str, Any]:
    problems = sorted(problems, key=lambda p: _SEV_ORDER.get(p.get("severity"), 3))
    counts = {"high": 0, "medium": 0, "low": 0}
    for p in problems:
        counts[p.get("severity", "medium")] = counts.get(p.get("severity", "medium"), 0) + 1

    generated_at = timezone.now().isoformat(timespec="seconds")
    blocks: List[Dict[str, Any]] = []

    blocks.append({"type": "masthead", "generated_at": generated_at,
                   "n_sources": len(sources), "n_findings": len(problems)})

    if synthesis.get("brief"):
        blocks.append({"type": "brief", "text": synthesis["brief"], "counts": counts})

    if packet["vitals"]["meters"]:
        blocks.append({"type": "keyvitals", "meters": packet["vitals"]["meters"]})

    if packet.get("eye"):
        e = packet["eye"]
        block = {"type": "figure_eye", "label": e["label"], "label_fa": e["label_fa"],
                 "confidence": e["confidence"], "n_photos": e["n_photos"],
                 "src": e.get("src", ""), "anchor": "#eye-screening",
                 "original_url": e.get("original_url", ""),
                 "phase1_url": e.get("phase1_url", ""),
                 "phase2_url": e.get("phase2_url", "")}
        if eye_vision:
            block["vision"] = {
                "pallor": eye_vision.get("pallor"),
                "pallor_fa": eye_vision.get("pallor_fa", ""),
                "image_quality": eye_vision.get("image_quality"),
                "note_fa": eye_vision.get("note_fa", ""),
                "agrees_with_model": eye_vision.get("agrees_with_model"),
            }
        blocks.append(block)

    for i, p in enumerate(problems):
        blocks.append({"type": "finding", "index": i + 1, **p})

    if synthesis.get("recommendations"):
        blocks.append({"type": "recommendations", "items": synthesis["recommendations"]})
    if synthesis.get("red_flags"):
        blocks.append({"type": "redflags", "items": synthesis["red_flags"]})
    wf = synthesis.get("workfitness")
    if wf and (wf.get("note") or wf.get("interactions")):
        blocks.append({"type": "workfitness", **wf})
    if synthesis.get("referrals"):
        blocks.append({"type": "referrals", "items": synthesis["referrals"]})
    if packet["medical_tests"]["n_reports"]:
        blocks.append({"type": "labs_pointer", "anchor": "#medical-tests",
                       "count": packet["medical_tests"]["n_reports"]})

    return {
        "v": 3,
        "generated_at": generated_at,
        "counts": counts,
        "blocks": blocks,
        "sources": sources,
        "trail": trail,
    }
