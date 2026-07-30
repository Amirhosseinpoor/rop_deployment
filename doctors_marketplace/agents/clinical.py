# doctors_marketplace/agents/clinical.py
"""Clinical agents: calculators, lab interpretation, red-flag safety, triage."""
from __future__ import annotations

import json
import math
import re
from typing import Any, Dict

from .base import BaseAgent, AgentContext, AgentResult
from .registry import register

EMERGENCY_HOTLINE = "115 (Iran emergency) / your local emergency number"


# --------------------------------------------------------------------------- #
# medical_calculator
# --------------------------------------------------------------------------- #
def _num(values: Dict[str, Any], *keys: str) -> float:
    for k in keys:
        if k in values and values[k] not in (None, ""):
            return float(values[k])
    raise KeyError(keys[0])


def _bmi(v, sex):
    kg = _num(v, "weight_kg", "weight"); cm = _num(v, "height_cm", "height")
    bmi = kg / (cm / 100.0) ** 2
    cat = ("underweight" if bmi < 18.5 else "normal" if bmi < 25 else
           "overweight" if bmi < 30 else "obese")
    return f"BMI = {bmi:.1f} kg/m² ({cat})."


def _bsa(v, sex):
    kg = _num(v, "weight_kg", "weight"); cm = _num(v, "height_cm", "height")
    return f"BSA (Mosteller) = {math.sqrt(cm * kg / 3600.0):.2f} m²."


def _egfr(v, sex):
    scr = _num(v, "creatinine_mg_dl", "creatinine", "scr"); age = _num(v, "age")
    female = str(sex).lower().startswith("f")
    k = 0.7 if female else 0.9
    a = -0.241 if female else -0.302
    egfr = (142 * min(scr / k, 1) ** a * max(scr / k, 1) ** -1.200
            * 0.9938 ** age * (1.012 if female else 1.0))
    stage = ("G1 (≥90)" if egfr >= 90 else "G2 (60–89)" if egfr >= 60 else
             "G3a (45–59)" if egfr >= 45 else "G3b (30–44)" if egfr >= 30 else
             "G4 (15–29)" if egfr >= 15 else "G5 (<15)")
    return f"eGFR (CKD-EPI 2021) = {egfr:.0f} mL/min/1.73m² — CKD {stage}."


def _crcl(v, sex):
    age = _num(v, "age"); kg = _num(v, "weight_kg", "weight")
    scr = _num(v, "creatinine_mg_dl", "creatinine", "scr")
    female = str(sex).lower().startswith("f")
    crcl = ((140 - age) * kg * (0.85 if female else 1.0)) / (72 * scr)
    return f"Creatinine clearance (Cockcroft-Gault) = {crcl:.0f} mL/min."


def _chads_vasc(v, sex):
    female = str(sex).lower().startswith("f")
    age = float(v.get("age", 0) or 0)
    score = (int(bool(v.get("chf"))) + int(bool(v.get("hypertension") or v.get("htn")))
             + (2 if age >= 75 else 1 if age >= 65 else 0)
             + int(bool(v.get("diabetes"))) + 2 * int(bool(v.get("stroke") or v.get("tia")))
             + int(bool(v.get("vascular"))) + (1 if female else 0))
    risk = {0: "0.2%", 1: "0.6%", 2: "2.2%", 3: "3.2%", 4: "4.8%",
            5: "7.2%", 6: "9.7%", 7: "11.2%", 8: "10.8%", 9: "12.2%"}.get(score, "high")
    return f"CHA₂DS₂-VASc = {score} (annual stroke risk ≈ {risk}). Anticoagulation usually considered at ≥2 (≥1 in men)."


def _meld(v, sex):
    import math as _m
    bili = max(1.0, _num(v, "bilirubin", "bilirubin_mg_dl"))
    inr = max(1.0, _num(v, "inr"))
    cr = min(4.0, max(1.0, _num(v, "creatinine", "creatinine_mg_dl", "scr")))
    meld = round(3.78 * _m.log(bili) + 11.2 * _m.log(inr) + 9.57 * _m.log(cr) + 6.43)
    return f"MELD = {meld}."


def _anion_gap(v, sex):
    na = _num(v, "na", "sodium"); cl = _num(v, "cl", "chloride"); hco3 = _num(v, "hco3", "bicarbonate")
    ag = na - (cl + hco3)
    flag = "high (metabolic acidosis?)" if ag > 12 else "normal"
    return f"Anion gap = {ag:.0f} mmol/L ({flag}; reference 8–12)."


def _peds_dose(v, sex):
    kg = _num(v, "weight_kg", "weight"); mgkg = _num(v, "mg_per_kg", "dose_mg_per_kg")
    per_day = float(v.get("doses_per_day", 1) or 1)
    total = kg * mgkg
    line = f"Pediatric dose = {total:.1f} mg per dose ({mgkg} mg/kg × {kg} kg)."
    if per_day > 1:
        line += f" Over {per_day:.0f} doses/day = {total * per_day:.1f} mg/day."
    return line


def _iv_rate(v, sex):
    vol = _num(v, "volume_ml", "volume"); t = _num(v, "time_min", "time_minutes")
    drop = _num(v, "drop_factor", "gtt_per_ml")
    return f"IV drip rate = {(vol * drop) / t:.0f} gtt/min ({vol} mL over {t} min, drop factor {drop})."


_CALCS = {
    "bmi": _bmi, "bsa": _bsa, "egfr": _egfr, "creatinine_clearance": _crcl,
    "cockcroft_gault": _crcl, "chads_vasc": _chads_vasc, "cha2ds2_vasc": _chads_vasc,
    "meld": _meld, "anion_gap": _anion_gap, "pediatric_dose": _peds_dose,
    "iv_drip_rate": _iv_rate,
}


@register
class MedicalCalculatorAgent(BaseAgent):
    key = "medical_calculator"
    name = "Medical calculator"
    icon = "calculator"
    category = "Clinical tools"
    description = ("Compute a validated clinical value from numeric inputs: BMI, BSA, "
                   "eGFR (CKD-EPI 2021), creatinine clearance (Cockcroft-Gault), "
                   "CHA₂DS₂-VASc, MELD, anion gap, pediatric weight-based dose, and IV "
                   "drip rate. Deterministic — never guess the numbers.")
    input_desc = ("calc = one of bmi | bsa | egfr | creatinine_clearance | chads_vasc | "
                  "meld | anion_gap | pediatric_dose | iv_drip_rate; values = the numbers "
                  "it needs (e.g. weight_kg, height_cm, creatinine_mg_dl, age); sex when relevant.")
    output_desc = "The computed value with units and a short interpretation/category."
    example = ('User: "BMI for 80 kg, 175 cm?" → calc="bmi", values={weight_kg:80, height_cm:175} '
               '→ "BMI = 26.1 kg/m² (overweight)."')
    stage_label = "Calculating"
    run_order = 40
    parameters = {
        "type": "object",
        "properties": {
            "calc": {"type": "string",
                     "enum": list(dict.fromkeys(_CALCS.keys())),
                     "description": "Which calculation to run."},
            "values": {"type": "object",
                       "description": "Numeric inputs, e.g. {\"weight_kg\":80,\"height_cm\":175}. "
                                      "Keys depend on calc (weight_kg, height_cm, age, "
                                      "creatinine_mg_dl, na, cl, hco3, bilirubin, inr, mg_per_kg, "
                                      "volume_ml, time_min, drop_factor, chf, htn, diabetes, stroke, vascular)."},
            "sex": {"type": "string", "enum": ["male", "female"],
                    "description": "Patient sex (needed for eGFR, CrCl, CHA₂DS₂-VASc)."},
        },
        "required": ["calc", "values"],
    }

    def run(self, args, ctx):
        calc = (args.get("calc") or "").strip().lower()
        values = args.get("values") or {}
        sex = args.get("sex") or values.get("sex") or "male"
        fn = _CALCS.get(calc)
        if not fn:
            return AgentResult(content=f"Unknown calculation '{calc}'.", ok=False)
        try:
            out = fn(values, sex)
            return AgentResult(content=out, display=out, ok=True)
        except KeyError as e:
            return AgentResult(content=f"Missing input for {calc}: {e}. Ask the user for it.", ok=False)
        except Exception as e:  # noqa: BLE001
            return AgentResult(content=f"Could not compute {calc}: {e}", ok=False)


# --------------------------------------------------------------------------- #
# lab_interpreter
# --------------------------------------------------------------------------- #
# (low, high, unit)  — sex-specific entries use a nested {male,female} tuple.
_LAB_RANGES = {
    "glucose_fasting": (70, 99, "mg/dL"),
    "hba1c": (4.0, 5.6, "%"),
    "sodium": (135, 145, "mmol/L"),
    "potassium": (3.5, 5.1, "mmol/L"),
    "chloride": (98, 107, "mmol/L"),
    "creatinine": {"male": (0.74, 1.35, "mg/dL"), "female": (0.59, 1.04, "mg/dL")},
    "bun": (7, 20, "mg/dL"),
    "hemoglobin": {"male": (13.5, 17.5, "g/dL"), "female": (12.0, 15.5, "g/dL")},
    "hematocrit": {"male": (41, 53, "%"), "female": (36, 46, "%")},
    "wbc": (4.0, 11.0, "10^3/µL"),
    "platelets": (150, 450, "10^3/µL"),
    "tsh": (0.4, 4.0, "mIU/L"),
    "ldl": (0, 100, "mg/dL"),
    "hdl": (40, 200, "mg/dL"),
    "triglycerides": (0, 150, "mg/dL"),
    "total_cholesterol": (0, 200, "mg/dL"),
    "alt": (7, 56, "U/L"),
    "ast": (10, 40, "U/L"),
    "crp": (0, 5, "mg/L"),
    "ferritin": {"male": (24, 336, "ng/mL"), "female": (11, 307, "ng/mL")},
    "calcium": (8.6, 10.3, "mg/dL"),
}
_LAB_ALIAS = {
    "fbs": "glucose_fasting", "glucose": "glucose_fasting", "a1c": "hba1c",
    "na": "sodium", "k": "potassium", "cl": "chloride", "hb": "hemoglobin",
    "hgb": "hemoglobin", "hct": "hematocrit", "plt": "platelets", "cr": "creatinine",
    "cholesterol": "total_cholesterol", "tg": "triglycerides",
}


@register
class LabInterpreterAgent(BaseAgent):
    key = "lab_interpreter"
    name = "Lab interpreter"
    icon = "flask"
    category = "Clinical tools"
    description = ("Compare a lab value against age/sex reference ranges and flag it low, "
                   "normal, or high. Covers common chemistry, CBC, lipids, thyroid and "
                   "liver tests. Use it whenever the user reports a specific lab number.")
    input_desc = "test (e.g. hemoglobin, hba1c, ldl, creatinine), value (number), and optional sex, age."
    output_desc = "The reference range and whether the value is LOW / NORMAL / HIGH."
    example = ('User: "My hemoglobin is 10.2, I\'m female." → test="hemoglobin", value=10.2, sex="female" '
               '→ "Hemoglobin 10.2 g/dL is LOW (ref 12.0–15.5) — suggests anemia."')
    stage_label = "Checking reference ranges"
    run_order = 45
    parameters = {
        "type": "object",
        "properties": {
            "test": {"type": "string", "description": "Lab test name, e.g. hemoglobin, hba1c, ldl."},
            "value": {"type": "number", "description": "The measured value."},
            "sex": {"type": "string", "enum": ["male", "female"]},
            "age": {"type": "number"},
        },
        "required": ["test", "value"],
    }

    def run(self, args, ctx):
        raw = (args.get("test") or "").strip().lower().replace(" ", "_")
        test = _LAB_ALIAS.get(raw, raw)
        rng = _LAB_RANGES.get(test)
        if rng is None:
            return AgentResult(content=f"No reference range on file for '{args.get('test')}'.", ok=False)
        try:
            value = float(args.get("value"))
        except (TypeError, ValueError):
            return AgentResult(content="A numeric value is required.", ok=False)
        if isinstance(rng, dict):
            sex = str(args.get("sex") or "male").lower()
            low, high, unit = rng.get("female" if sex.startswith("f") else "male")
        else:
            low, high, unit = rng
        flag = "LOW" if value < low else "HIGH" if value > high else "NORMAL"
        out = (f"{args.get('test')} = {value} {unit} is {flag} "
               f"(reference {low}–{high} {unit}).")
        return AgentResult(content=out, display=out, ok=True)


# --------------------------------------------------------------------------- #
# red_flag_check  (safety pre-pass — runs first, deterministic + fast)
# --------------------------------------------------------------------------- #
_RED_FLAGS = [
    ("chest pain / cardiac", re.compile(
        r"chest pain|crushing chest|pressure in (my )?chest|درد قفسه سینه|درد سینه", re.I)),
    ("stroke (FAST)", re.compile(
        r"face droop|slurred speech|weakness on one side|can't move|numb.*(arm|face)|"
        r"سکته|فلج|کج شدن صورت|لکنت", re.I)),
    ("difficulty breathing", re.compile(
        r"can't breathe|cannot breathe|short(ness)? of breath|choking|"
        r"نفس(م)? (نمی[‌ ]?آید|بالا نمیاد)|تنگی نفس شدید", re.I)),
    ("severe bleeding", re.compile(
        r"heavy bleeding|won'?t stop bleeding|bleeding a lot|خونریزی شدید", re.I)),
    ("anaphylaxis", re.compile(
        r"anaphylaxis|throat closing|swelling.*(throat|tongue)|شوک آنافیلاکسی|تورم گلو", re.I)),
    ("suicidal ideation", re.compile(
        r"suicid|kill myself|end my life|want to die|خودکشی|به زندگی(م)? ادامه ندهم", re.I)),
    ("stroke/loss of consciousness", re.compile(
        r"unconscious|passed out|fainted|بیهوش", re.I)),
]


@register
class RedFlagCheckAgent(BaseAgent):
    key = "red_flag_check"
    name = "Emergency red-flag detector"
    icon = "alert"
    category = "Safety"
    description = ("Scan the patient's message for emergency warning signs (chest pain, "
                   "stroke, breathing trouble, severe bleeding, anaphylaxis, suicidal "
                   "ideation). If found, the app shows an emergency banner and the answer "
                   "must direct the user to emergency care.")
    input_desc = "text — the patient's message (defaults to the current message)."
    output_desc = "Whether an emergency red flag was detected, which one, and a safety directive."
    example = ('User: "I have crushing chest pain radiating to my arm." → detects "chest pain / '
               'cardiac" → emergency banner + advice to call emergency services now.')
    stage_label = "Safety check"
    icon_stage = "alert"
    run_order = 0
    pre_pass = True          # always runs first, before the model
    parameters = {
        "type": "object",
        "properties": {"text": {"type": "string", "description": "Text to scan."}},
    }

    def run(self, args, ctx):
        text = (args.get("text") or ctx.query or "")
        hits = [label for label, rx in _RED_FLAGS if rx.search(text)]
        if not hits:
            return AgentResult(content="No emergency red flags detected.", ok=True)
        label = hits[0]
        banner = {
            "banner": {
                "level": "emergency",
                "title": "This may be a medical emergency",
                "text": (f"Signs of {label} were detected. If this is happening now, call "
                         f"{EMERGENCY_HOTLINE} or go to the nearest emergency department "
                         f"immediately. Do not wait for an online reply."),
            }
        }
        directive = (f"EMERGENCY RED FLAG DETECTED ({label}). Begin your reply by urging the "
                     f"user to seek emergency care immediately ({EMERGENCY_HOTLINE}); keep any "
                     f"other guidance brief and secondary.")
        return AgentResult(content=directive, ui=banner, display=f"⚠ {label}", ok=True)


# --------------------------------------------------------------------------- #
# symptom_triage  (compact LLM sub-call -> structured urgency)
# --------------------------------------------------------------------------- #
@register
class SymptomTriageAgent(BaseAgent):
    key = "symptom_triage"
    name = "Symptom triage"
    icon = "triage"
    category = "Safety"
    description = ("Assess how urgently the described symptoms need care and recommend a "
                   "concrete next step. Returns a structured urgency level so the assistant "
                   "can give consistent, safe guidance.")
    input_desc = "symptoms (free text); optional duration and severity (mild/moderate/severe)."
    output_desc = ("Urgency = emergency | urgent | see_doctor | self_care, with a one-line "
                   "rationale, a recommended action, and a timeframe.")
    example = ('User: "Fever 39°C and a stiff neck for a day." → urgency="urgent", action="Seek '
               'same-day medical assessment (possible meningitis).", timeframe="within hours".')
    stage_label = "Triaging symptoms"
    run_order = 30
    parameters = {
        "type": "object",
        "properties": {
            "symptoms": {"type": "string", "description": "The symptoms to triage."},
            "duration": {"type": "string"},
            "severity": {"type": "string", "enum": ["mild", "moderate", "severe"]},
        },
        "required": ["symptoms"],
    }

    def run(self, args, ctx):
        from ..services.llm import LLMClient
        symptoms = args.get("symptoms") or ctx.query
        extra = f" Duration: {args.get('duration')}." if args.get("duration") else ""
        extra += f" Severity: {args.get('severity')}." if args.get("severity") else ""
        try:
            raw = LLMClient().chat([
                {"role": "system", "content":
                    "You are a medical triage classifier. Given symptoms, reply with ONLY compact "
                    "JSON: {\"urgency\":\"emergency|urgent|see_doctor|self_care\","
                    "\"rationale\":\"...\",\"action\":\"...\",\"timeframe\":\"...\"}. Be cautious: "
                    "when in doubt escalate."},
                {"role": "user", "content": f"Symptoms: {symptoms}.{extra}"},
            ], temperature=0).strip()
            m = re.search(r"\{.*\}", raw, re.S)
            data = json.loads(m.group(0)) if m else {}
        except Exception as e:  # noqa: BLE001
            return AgentResult(content=f"Triage unavailable ({e}).", ok=False)
        urg = data.get("urgency", "see_doctor")
        out = (f"Triage: urgency={urg}; {data.get('rationale','')} "
               f"Recommended: {data.get('action','')} ({data.get('timeframe','')}).")
        return AgentResult(content=out, display=f"Triage: {urg}", ok=True)
