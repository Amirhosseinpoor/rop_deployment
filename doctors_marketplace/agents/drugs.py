# doctors_marketplace/agents/drugs.py
"""Drug agents: label lookup, interaction check, contraindication cross-check.

All three use the free openFDA drug-label API (api.fda.gov). openFDA is used for
interactions/contraindications because NLM retired its standalone pairwise Drug
Interaction API in 2024; label sections are the reliable free alternative, so
results are label-derived and the assistant is told to treat them as such.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import requests

from .base import BaseAgent, AgentContext, AgentResult
from .registry import register

log = logging.getLogger(__name__)
_OPENFDA = "https://api.fda.gov/drug/label.json"


def _fetch_label(name: str) -> Optional[Dict[str, Any]]:
    """Return the first openFDA label record for a drug name, or None."""
    name = (name or "").strip()
    if not name:
        return None
    q = f'(openfda.generic_name:"{name}" OR openfda.brand_name:"{name}")'
    try:
        r = requests.get(_OPENFDA, params={"search": q, "limit": 1}, timeout=8)
        if r.status_code == 404:
            return None
        r.raise_for_status()
        results = r.json().get("results") or []
        return results[0] if results else None
    except Exception as e:  # noqa: BLE001
        log.warning("DRUG | openFDA lookup failed for %s: %s", name, e)
        return None


def _sec(label: Dict[str, Any], *fields: str, limit: int = 600) -> str:
    for f in fields:
        val = label.get(f)
        if val:
            text = " ".join(val) if isinstance(val, list) else str(val)
            return text.strip()[:limit]
    return ""


@register
class DrugLookupAgent(BaseAgent):
    key = "drug_lookup"
    name = "Drug lookup"
    icon = "pill"
    category = "Medications"
    description = ("Look up a medication's FDA label: what it's used for, usual dosing, key "
                   "warnings and common side effects. Use when the user asks about a specific drug.")
    input_desc = "name — a generic or brand drug name (e.g. metformin, Tylenol)."
    output_desc = "Indications, dosage & administration, warnings and adverse reactions from the FDA label."
    example = ('User: "What is ibuprofen used for and its warnings?" → name="ibuprofen" → returns '
               'indications, dosing, and warnings, cited to the FDA label (DailyMed).')
    stage_label = "Looking up the drug"
    produces_sources = True
    run_order = 60
    parameters = {
        "type": "object",
        "properties": {"name": {"type": "string", "description": "Generic or brand drug name."}},
        "required": ["name"],
    }

    def run(self, args, ctx):
        name = (args.get("name") or "").strip()
        label = _fetch_label(name)
        if not label:
            return AgentResult(content=f"No FDA label found for '{name}'. Check the spelling.", ok=False)
        ind = _sec(label, "indications_and_usage")
        dose = _sec(label, "dosage_and_administration")
        warn = _sec(label, "warnings", "warnings_and_cautions", "boxed_warning")
        adr = _sec(label, "adverse_reactions")
        openfda = label.get("openfda", {})
        title = (openfda.get("brand_name") or openfda.get("generic_name") or [name])[0]
        body = (f"Indications: {ind or 'n/a'}\nDosage: {dose or 'n/a'}\n"
                f"Warnings: {warn or 'n/a'}\nAdverse reactions: {adr or 'n/a'}")
        source = {"type": "web", "title": f"FDA label: {title}",
                  "url": "https://dailymed.nlm.nih.gov/dailymed/", "domain": "fda.gov",
                  "page_content": body}
        return AgentResult(content=f"FDA label for {title}:\n{body}",
                           sources=[source], display=f"Drug: {title}", ok=True)


@register
class DrugInteractionsAgent(BaseAgent):
    key = "drug_interactions"
    name = "Drug interactions"
    icon = "interaction"
    category = "Medications"
    description = ("Check a list of medications for interaction concerns using each drug's FDA "
                   "label interaction section. Use when the user lists 2+ drugs or asks whether "
                   "drugs are safe together. (Label-derived, not a substitute for a pharmacist.)")
    input_desc = "drugs — an array of 2+ drug names."
    output_desc = "Each drug's documented interaction cautions, so overlapping risks can be flagged."
    example = ('User: "Can I take warfarin with ibuprofen?" → drugs=["warfarin","ibuprofen"] → '
               'returns each label\'s interaction cautions (e.g. bleeding risk).')
    stage_label = "Checking drug interactions"
    produces_sources = True
    run_order = 62
    parameters = {
        "type": "object",
        "properties": {"drugs": {"type": "array", "items": {"type": "string"},
                                 "description": "Two or more drug names."}},
        "required": ["drugs"],
    }

    def run(self, args, ctx):
        drugs: List[str] = [d for d in (args.get("drugs") or []) if d]
        if len(drugs) < 2:
            return AgentResult(content="Provide at least two drugs to compare.", ok=False)
        lines, sources, missing = [], [], []
        for d in drugs[:5]:
            label = _fetch_label(d)
            if not label:
                missing.append(d)
                continue
            inter = _sec(label, "drug_interactions", limit=700) or "no interaction section on label"
            openfda = label.get("openfda", {})
            title = (openfda.get("generic_name") or openfda.get("brand_name") or [d])[0]
            lines.append(f"{title}: {inter}")
            sources.append({"type": "web", "title": f"FDA label interactions: {title}",
                            "url": "https://dailymed.nlm.nih.gov/dailymed/", "domain": "fda.gov",
                            "page_content": f"{title} interactions: {inter}"})
        if not lines:
            return AgentResult(content=f"No FDA labels found for: {', '.join(drugs)}.", ok=False)
        note = ("\n\nIdentify any overlapping/compounding risks between these drugs and advise the "
                "user to confirm with a pharmacist. This is label-derived, not a pairwise database.")
        if missing:
            note += f" (No label found for: {', '.join(missing)}.)"
        return AgentResult(content="Interaction cautions (from FDA labels):\n" + "\n".join(lines) + note,
                           sources=sources, display=f"Interactions: {len(lines)} drug(s)", ok=True)


@register
class CheckContraindicationsAgent(BaseAgent):
    key = "check_contraindications"
    name = "Contraindication check"
    icon = "shield"
    category = "Medications"
    description = ("Cross-check a medication against the patient's allergies and conditions using "
                   "the drug's FDA contraindications and warnings. Use before suggesting or "
                   "confirming a drug when the patient's allergies/conditions are known.")
    input_desc = ("drug (name) and profile = {allergies:[...], conditions:[...], medications:[...]}. "
                  "If the profile is unknown, ask the user first.")
    output_desc = "The drug's contraindications/warnings and any that match the patient's profile."
    example = ('User (asthma, penicillin allergy): "Can I take amoxicillin?" → drug="amoxicillin", '
               'profile={allergies:["penicillin"]} → flags the penicillin-class contraindication.')
    stage_label = "Checking contraindications"
    produces_sources = True
    run_order = 64
    parameters = {
        "type": "object",
        "properties": {
            "drug": {"type": "string", "description": "Drug name to check."},
            "profile": {"type": "object", "description":
                        "Patient profile: {allergies:[], conditions:[], medications:[]}."},
        },
        "required": ["drug"],
    }

    def run(self, args, ctx):
        drug = (args.get("drug") or "").strip()
        profile = args.get("profile") or {}
        terms: List[str] = []
        for key in ("allergies", "conditions", "medications"):
            terms += [str(t).lower() for t in (profile.get(key) or []) if t]
        label = _fetch_label(drug)
        if not label:
            return AgentResult(content=f"No FDA label found for '{drug}'.", ok=False)
        contra = _sec(label, "contraindications", limit=800)
        warn = _sec(label, "warnings", "warnings_and_cautions", "boxed_warning", limit=800)
        blob = f"{contra}\n{warn}".lower()
        matched = sorted({t for t in terms if t and t in blob})
        openfda = label.get("openfda", {})
        title = (openfda.get("generic_name") or openfda.get("brand_name") or [drug])[0]
        body = f"Contraindications: {contra or 'n/a'}\nWarnings: {warn or 'n/a'}"
        if not terms:
            body += "\n\n(No patient allergies/conditions were provided — ask the user for them.)"
        elif matched:
            body += f"\n\n⚠ POTENTIAL MATCH with patient profile: {', '.join(matched)}. Flag this clearly."
        else:
            body += f"\n\nNo direct match found against the patient's profile ({', '.join(terms)})."
        source = {"type": "web", "title": f"FDA label: {title}",
                  "url": "https://dailymed.nlm.nih.gov/dailymed/", "domain": "fda.gov",
                  "page_content": body}
        return AgentResult(content=f"Contraindication check for {title}:\n{body}",
                           sources=[source],
                           display=f"Contraindications: {title}" + (" ⚠" if matched else ""),
                           ok=True)
