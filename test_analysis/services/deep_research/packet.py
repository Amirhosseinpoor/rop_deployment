"""
Stage 0 of the Deep Health Research pipeline — the **Evidence Packet**.

Everything here is deterministic: it reads a ``HealthProfile`` together with its
eye-AI results (``EyeAnalysis``) and extracted lab reports (``MedicalTest``) and
turns them into

    • ``facts``    — a flat list of clinically-salient facts, each with a stable
                     citation id ("S1", "S2", …) so the LLM can ground every claim
                     in a specific patient datum ("هموگلوبین ۱۰.۹ [S3]").
    • ``sources``  — the same salient facts shaped as *source cards* (type
                     "patient") that the report renders next to web/document
                     sources, exactly like the /rop/ assistant.
    • ``summary``  — a compact Persian/English text digest fed to the LLM.

No arithmetic ever happens in the LLM — BMI class, blood-pressure category,
anaemia severity, spirometry pattern, pack-years and abnormal-analyte detection
are all computed here so the model only interprets, never calculates.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #
def _f(value: Any) -> Optional[float]:
    """Best-effort float parse (labs are stored as free-text CharFields)."""
    if value is None:
        return None
    try:
        m = re.search(r"-?\d+(?:\.\d+)?", str(value).replace("٫", ".").replace(",", ""))
        return float(m.group()) if m else None
    except Exception:
        return None


def _clean(value: Any) -> str:
    return str(value).strip() if value not in (None, "") else ""


class _Facts:
    """Accumulates salient facts and hands out stable S-ids."""

    def __init__(self) -> None:
        self.items: List[Dict[str, Any]] = []

    def add(self, label: str, value: str, *, group: str, status: str = "info",
            ref: str = "", note: str = "") -> str:
        sid = f"S{len(self.items) + 1}"
        self.items.append({
            "id": sid, "label": label, "value": value, "group": group,
            "status": status,        # info | normal | abnormal | warn | high | low
            "ref": ref,              # in-page anchor the source card links to
            "note": note,
        })
        return sid


# --------------------------------------------------------------------------- #
# clinical derivations (deterministic)
# --------------------------------------------------------------------------- #
def _bp_category(sys: Optional[float], dia: Optional[float]) -> tuple[str, str]:
    """ACC/AHA 2017 categories. Returns (persian_label, status)."""
    if sys is None and dia is None:
        return "", "info"
    s, d = sys or 0, dia or 0
    if s >= 180 or d >= 120:
        return "بحران فشار خون", "high"
    if s >= 140 or d >= 90:
        return "فشار خون بالا مرحله ۲", "high"
    if s >= 130 or d >= 80:
        return "فشار خون بالا مرحله ۱", "warn"
    if 120 <= s < 130 and d < 80:
        return "فشار خون افزایش‌یافته", "warn"
    if s and s < 120 and d and d < 80:
        return "طبیعی", "normal"
    return "نامشخص", "info"


def _bmi_category(bmi: Optional[float]) -> tuple[str, str]:
    if not bmi:
        return "", "info"
    if bmi < 18.5:
        return "کمبود وزن", "warn"
    if bmi < 25:
        return "طبیعی", "normal"
    if bmi < 30:
        return "اضافه‌وزن", "warn"
    if bmi < 35:
        return "چاقی درجه ۱", "high"
    return "چاقی درجه ۲ یا بالاتر", "high"


def _glucose_category(fbs: Optional[float]) -> tuple[str, str]:
    if not fbs:
        return "", "info"
    if fbs >= 126:
        return "در محدودهٔ دیابت", "high"
    if fbs >= 100:
        return "پیش‌دیابت", "warn"
    return "طبیعی", "normal"


def _spirometry_pattern(fev1_fvc: Optional[float]) -> tuple[str, str]:
    if fev1_fvc is None:
        return "", "info"
    ratio = fev1_fvc / 100 if fev1_fvc > 1.5 else fev1_fvc
    if ratio < 0.70:
        return "الگوی انسدادی محتمل", "warn"
    return "نسبت FEV1/FVC طبیعی", "normal"


# --------------------------------------------------------------------------- #
# gauges for the vitals dashboard (deterministic)
# --------------------------------------------------------------------------- #
def _zone_from_status(status: str) -> str:
    return {"high": "high", "low": "warn", "warn": "warn", "abnormal": "warn",
            "normal": "good", "info": "info"}.get(status, "info")


def _pct(value: float, lo: float, hi: float) -> int:
    if hi <= lo:
        return 0
    return max(0, min(100, round((value - lo) / (hi - lo) * 100)))


# --------------------------------------------------------------------------- #
# eye-AI aggregation
# --------------------------------------------------------------------------- #
def _img_url(field) -> str:
    """MEDIA url for an ImageField, empty string if missing."""
    try:
        return field.url if field else ""
    except Exception:  # noqa: BLE001
        return ""


def _img_path(field) -> str:
    """Filesystem path for an ImageField (for the vision model), '' if missing."""
    try:
        return field.path if field else ""
    except Exception:  # noqa: BLE001
        return ""


def _eye_summary(profile) -> Optional[Dict[str, Any]]:
    analyses = []
    for ei in profile.eye_images.all():
        a = getattr(ei, "analysis", None)
        if a and a.status == "done" and a.anemia_label:
            analyses.append((ei, a))
    if not analyses:
        return None
    pos = [a for _, a in analyses if a.anemia_label == "positive"]
    # majority vote; confidence = mean of the winning side
    label = "positive" if len(pos) * 2 >= len(analyses) and pos else "negative"
    winners = pos if label == "positive" else [a for _, a in analyses if a.anemia_label != "positive"]
    conf = sum((a.anemia_confidence or 0) for a in winners) / max(1, len(winners))

    # pick a representative image (prefer one on the winning side) for the block +
    # for the vision agent to actually look at.
    rep_ei, rep_a = next(
        ((ei, a) for ei, a in analyses if a.anemia_label == label), analyses[0]
    )
    return {
        "label": label,
        "label_fa": "نشانه‌های کم‌خونی" if label == "positive" else "بدون نشانهٔ کم‌خونی",
        "confidence": round(conf, 3),
        "n_photos": len(analyses),
        "n_positive": len(pos),
        # urls for the report block
        "original_url": _img_url(rep_ei.image),
        "phase1_url": _img_url(rep_a.phase1_overlay),
        "phase2_url": _img_url(rep_a.phase2_overlay),
        # filesystem paths for the vision agent (never sent to the browser)
        "original_path": _img_path(rep_ei.image),
        "phase1_path": _img_path(rep_a.phase1_overlay),
    }


# --------------------------------------------------------------------------- #
# medical-test analyte scan
# --------------------------------------------------------------------------- #
_HB_RE = re.compile(r"\b(h(a?e)?moglobin|hgb|hb)\b", re.I)


def _scan_medical_tests(profile, facts: _Facts) -> Dict[str, Any]:
    """Pull abnormal analytes + hemoglobin from extracted MedicalTest panels."""
    abnormal: List[Dict[str, str]] = []
    hemoglobin: Optional[Dict[str, Any]] = None
    n_reports = 0
    for mt in profile.medical_tests.all():
        if mt.status != "done":
            continue
        n_reports += 1
        for panel in (mt.panels or []):
            for an in panel.get("analytes", []):
                name = _clean(an.get("name"))
                result = _clean(an.get("result"))
                flag = _clean(an.get("flag"))
                unit = _clean(an.get("unit"))
                ref = _clean(an.get("reference"))
                if _HB_RE.search(name) and hemoglobin is None and _f(result) is not None:
                    hemoglobin = {"value": _f(result), "unit": unit, "flag": flag,
                                  "reference": ref, "raw": result}
                if flag:  # only abnormal analytes become citable facts (bounded)
                    abnormal.append({"name": name, "result": result, "flag": flag,
                                     "unit": unit, "reference": ref,
                                     "panel": _clean(panel.get("name"))})
    # register the most salient abnormal analytes as patient sources (cap 12)
    for an in abnormal[:12]:
        status = "low" if an["flag"].upper().startswith("L") else "high"
        facts.add(
            f"{an['name']}", f"{an['result']} {an['unit']}".strip(),
            group="labs", status=status, ref="#medical-tests",
            note=f"پرچم {an['flag']}؛ محدودهٔ مرجع {an['reference']}".strip("؛ "),
        )
    return {"abnormal": abnormal, "hemoglobin": hemoglobin, "n_reports": n_reports}


# --------------------------------------------------------------------------- #
# active occupational hazards
# --------------------------------------------------------------------------- #
_HAZARD_MAP = [
    ("hazard_physical_noise", "سر و صدا", "فیزیکی"),
    ("hazard_physical_vibration", "ارتعاش", "فیزیکی"),
    ("hazard_physical_non_ionizing_radiation", "اشعهٔ غیریونیزان", "فیزیکی"),
    ("hazard_physical_ionizing_radiation", "اشعهٔ یونیزان", "فیزیکی"),
    ("hazard_physical_heat_stress", "استرس حرارتی", "فیزیکی"),
    ("hazard_chemical_dust", "گرد و غبار", "شیمیایی"),
    ("hazard_chemical_metal_fumes", "دمهٔ فلزات", "شیمیایی"),
    ("hazard_chemical_solvents", "حلال‌ها", "شیمیایی"),
    ("hazard_chemical_pesticides", "آفت‌کش‌ها", "شیمیایی"),
    ("hazard_chemical_acids_bases", "اسیدها و بازها", "شیمیایی"),
    ("hazard_chemical_gases", "گازها", "شیمیایی"),
    ("hazard_biological_bites", "گزش", "بیولوژیک"),
    ("hazard_biological_bacteria", "باکتری", "بیولوژیک"),
    ("hazard_biological_virus", "ویروس", "بیولوژیک"),
    ("hazard_biological_parasite", "انگل", "بیولوژیک"),
    ("hazard_ergonomic_prolonged_sitting_standing", "ایستادن/نشستن طولانی", "ارگونومیک"),
    ("hazard_ergonomic_repetitive_work", "کار تکراری", "ارگونومیک"),
    ("hazard_ergonomic_heavy_lifting", "حمل بار سنگین", "ارگونومیک"),
    ("hazard_ergonomic_poor_posture", "وضعیت بدنی نامناسب", "ارگونومیک"),
    ("hazard_psychological_shift_work", "نوبت‌کاری", "روانی-اجتماعی"),
    ("hazard_psychological_stressors", "عوامل استرس‌زای شغلی", "روانی-اجتماعی"),
]


def _active_hazards(profile) -> List[Dict[str, str]]:
    out = []
    for field, label, category in _HAZARD_MAP:
        if getattr(profile, field, False):
            out.append({"label": label, "category": category})
    return out


# --------------------------------------------------------------------------- #
# public entry point
# --------------------------------------------------------------------------- #
def build_evidence_packet(profile) -> Dict[str, Any]:
    facts = _Facts()

    # ---- demographics -----------------------------------------------------
    gender_fa = {"Male": "مرد", "Female": "زن"}.get(_clean(profile.gender), _clean(profile.gender))
    demo = {
        "age": profile.age,
        "gender": gender_fa,
        "job": _clean(profile.current_job_title),
        "smoking": (
            f"سیگاری فعلی ({profile.cigs_per_day or profile.smoking_details or '?'} نخ در روز)"
            if profile.is_currently_smoking else
            "سابقهٔ مصرف سیگار" if profile.has_past_smoking_history else "غیرسیگاری"
        ),
    }

    # ---- vitals + gauges --------------------------------------------------
    sys_bp = float(profile.exam_systolic_bp) if profile.exam_systolic_bp else None
    dia_bp = float(profile.exam_diastolic_bp) if profile.exam_diastolic_bp else None
    if sys_bp is None and _clean(profile.exam_blood_pressure):
        nums = re.findall(r"\d+", profile.exam_blood_pressure)
        if len(nums) >= 2:
            sys_bp, dia_bp = float(nums[0]), float(nums[1])
    bmi = profile.bmi
    if not bmi and profile.exam_weight and profile.exam_height:
        h = profile.exam_height / 100.0
        if h:
            bmi = round(profile.exam_weight / (h * h), 1)
    fbs = _f(profile.lab_fbs) or _f(profile.lab_glucose)
    pulse = float(profile.exam_pulse_rate) if profile.exam_pulse_rate else None
    fev1_fvc = _f(profile.spirometry_fev1_fvc_ratio)

    bp_label, bp_status = _bp_category(sys_bp, dia_bp)
    bmi_label, bmi_status = _bmi_category(bmi)
    glu_label, glu_status = _glucose_category(fbs)
    spir_label, spir_status = _spirometry_pattern(fev1_fvc)

    meters: List[Dict[str, Any]] = []
    if sys_bp:
        sid = facts.add("فشار خون", f"{int(sys_bp)}/{int(dia_bp or 0)} mmHg",
                        group="vitals", status=bp_status, note=bp_label)
        meters.append({"label": "فشار خون", "value": f"{int(sys_bp)}/{int(dia_bp or 0)}",
                       "unit": "mmHg", "zone": _zone_from_status(bp_status),
                       "pct": _pct(sys_bp, 90, 180), "note": bp_label, "src": sid})
    if bmi:
        sid = facts.add("شاخص تودهٔ بدنی (BMI)", f"{bmi}", group="vitals",
                        status=bmi_status, note=bmi_label)
        meters.append({"label": "BMI", "value": f"{bmi}", "unit": "",
                       "zone": _zone_from_status(bmi_status),
                       "pct": _pct(bmi, 15, 40), "note": bmi_label, "src": sid})
    if pulse:
        p_status = "warn" if (pulse < 60 or pulse > 100) else "normal"
        sid = facts.add("نبض", f"{int(pulse)} bpm", group="vitals", status=p_status)
        meters.append({"label": "نبض", "value": f"{int(pulse)}", "unit": "bpm",
                       "zone": _zone_from_status(p_status), "pct": _pct(pulse, 40, 140),
                       "note": "خارج از محدوده" if p_status == "warn" else "طبیعی", "src": sid})
    if fbs:
        sid = facts.add("قند خون ناشتا", f"{int(fbs)} mg/dL", group="vitals",
                        status=glu_status, note=glu_label)
        meters.append({"label": "قند ناشتا", "value": f"{int(fbs)}", "unit": "mg/dL",
                       "zone": _zone_from_status(glu_status), "pct": _pct(fbs, 70, 200),
                       "note": glu_label, "src": sid})
    chol = _f(profile.lab_total_cholesterol)
    if chol:
        c_status = "high" if chol >= 240 else "warn" if chol >= 200 else "normal"
        sid = facts.add("کلسترول تام", f"{int(chol)} mg/dL", group="vitals",
                        status=c_status)
        meters.append({"label": "کلسترول", "value": f"{int(chol)}", "unit": "mg/dL",
                       "zone": _zone_from_status(c_status), "pct": _pct(chol, 120, 300),
                       "note": "بالا" if c_status != "normal" else "طبیعی", "src": sid})

    # ---- eye AI -----------------------------------------------------------
    eye = _eye_summary(profile)
    eye_src = ""
    if eye:
        eye_src = facts.add(
            "غربالگری کم‌خونی از تصویر چشم (EfficientNet-B0)",
            f"{eye['label_fa']} — اطمینان {eye['confidence']:.2f}",
            group="eye", status="high" if eye["label"] == "positive" else "normal",
            ref="#eye-screening", note=f"{eye['n_photos']} تصویر بررسی شد",
        )
        eye["src"] = eye_src

    # ---- medical tests ----------------------------------------------------
    mt = _scan_medical_tests(profile, facts)
    if mt["hemoglobin"] and mt["hemoglobin"]["flag"]:
        facts.add("هموگلوبین", f"{mt['hemoglobin']['raw']} {mt['hemoglobin']['unit']}".strip(),
                  group="labs", status="low", ref="#medical-tests",
                  note=f"محدودهٔ مرجع {mt['hemoglobin']['reference']}")

    # ---- occupational hazards --------------------------------------------
    hazards = _active_hazards(profile)
    if hazards:
        by_cat: Dict[str, List[str]] = {}
        for h in hazards:
            by_cat.setdefault(h["category"], []).append(h["label"])
        for cat, labels in by_cat.items():
            facts.add(f"مواجههٔ شغلی ({cat})", "، ".join(labels),
                      group="hazards", status="warn")

    # ---- history flags ----------------------------------------------------
    history: List[str] = []
    if profile.has_family_cancer_or_chronic_disease:
        history.append(f"سابقهٔ خانوادگی: {_clean(profile.family_disease_details) or 'بله'}")
        facts.add("سابقهٔ خانوادگی بیماری مزمن/سرطان",
                  _clean(profile.family_disease_details) or "بله", group="history", status="warn")
    if profile.has_disease_history:
        history.append(f"سابقهٔ بیماری: {_clean(profile.disease_history_details) or 'بله'}")
    if profile.is_on_medication:
        history.append(f"داروی مصرفی: {_clean(profile.medication_details) or 'بله'}")
        facts.add("داروی مصرفی فعلی", _clean(profile.medication_details) or "بله",
                  group="history", status="info")
    if profile.has_allergies:
        history.append(f"آلرژی: {_clean(profile.allergy_details) or 'بله'}")
    if profile.has_diabetes:
        history.append("سابقهٔ دیابت")
    if profile.on_bp_meds:
        history.append("مصرف داروی فشار خون")
    if profile.is_currently_smoking:
        facts.add("مصرف سیگار", demo["smoking"], group="history", status="warn")

    # ---- paraclinical -----------------------------------------------------
    paraclinical = {
        "spirometry_interpretation": _clean(profile.spirometry_interpretation),
        "spirometry_pattern": spir_label,
        "ecg": _clean(profile.ecg_findings),
        "chest_xray": _clean(profile.chest_xray_findings),
    }
    if spir_label and spir_status != "normal":
        facts.add("اسپیرومتری", spir_label, group="paraclinical", status="warn")

    # ---- sources (patient facts as source cards) --------------------------
    sources = [{
        "id": f["id"], "type": "patient", "title": f"{f['label']}: {f['value']}",
        "domain": None, "url": None, "ref": f["ref"] or "", "status": f["status"],
        "note": f["note"],
    } for f in facts.items]

    # ---- LLM text digest --------------------------------------------------
    summary = _build_summary(demo, meters, eye, mt, hazards, history, paraclinical, facts)

    return {
        "profile_id": profile.id,
        "demographics": demo,
        "vitals": {"meters": meters, "bp_label": bp_label, "bmi_label": bmi_label},
        "eye": eye,
        "medical_tests": mt,
        "hazards": hazards,
        "history": history,
        "paraclinical": paraclinical,
        "facts": facts.items,
        "sources": sources,
        "summary": summary,
    }


def _build_summary(demo, meters, eye, mt, hazards, history, paraclinical, facts) -> str:
    """Compact digest the LLM reads. Each salient datum carries its [S-id] so the
    model can copy the exact marker into its grounded answer."""
    lines: List[str] = []
    lines.append(f"سن: {demo.get('age') or '؟'} | جنسیت: {demo.get('gender') or '؟'} | "
                 f"شغل: {demo.get('job') or '؟'} | دخانیات: {demo.get('smoking')}")
    lines.append("— شاخص‌های حیاتی و آزمایش‌ها (با شناسهٔ منبع):")
    for f in facts.items:
        lines.append(f"  [{f['id']}] {f['label']}: {f['value']}"
                     + (f" ({f['note']})" if f['note'] else ""))
    if eye:
        lines.append(f"— هوش مصنوعی چشم: {eye['label_fa']} با اطمینان {eye['confidence']:.2f} "
                     f"از {eye['n_photos']} تصویر [{eye.get('src','')}]")
    if paraclinical.get("spirometry_interpretation"):
        lines.append(f"— اسپیرومتری: {paraclinical['spirometry_interpretation']}")
    if paraclinical.get("ecg"):
        lines.append(f"— نوار قلب: {paraclinical['ecg']}")
    if paraclinical.get("chest_xray"):
        lines.append(f"— رادیوگرافی قفسهٔ سینه: {paraclinical['chest_xray']}")
    if hazards:
        lines.append("— مواجهه‌های شغلی: " + "، ".join(h["label"] for h in hazards))
    if history:
        lines.append("— سوابق: " + " | ".join(history))
    return "\n".join(lines)
