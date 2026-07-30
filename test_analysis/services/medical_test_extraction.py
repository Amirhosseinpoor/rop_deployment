"""
Medical-test extraction pipeline.

Takes an uploaded medical-test document (PDF or image) and uses an LLM
(gpt-4o-mini, via the same METIS/OpenAI-compatible endpoint the rest of the
app already uses) to pull out the *test results only* — the panels, analytes,
their values, units, flags and reference ranges — into structured JSON that we
can render as tables for every viewer tier.

Patient-identifying information (name, DOB, MRN, ordering provider, address) is
deliberately NOT extracted; the prompt instructs the model to ignore it.

Best-effort: :func:`extract_medical_test` never raises — on failure it records
``status='failed'`` with the error so a web request is never broken.
"""
from __future__ import annotations

import base64
import io
import json
import logging
import os

from django.utils import timezone

logger = logging.getLogger(__name__)

# Reuse the app's existing OpenAI-compatible config. Prefer GAPGPT (the funded
# provider used elsewhere in the app), falling back to the METIS endpoint.
# gpt-4o-mini is multimodal (accepts images) and is the app's default model.
_API_KEY = os.getenv("GAPGPT_API_KEY") or os.getenv("METIS_API_KEY")
_BASE_URL = os.getenv("GAPGPT_BASE_URL") or os.getenv("BASE_URL")
_MODEL = (os.getenv("MEDICAL_TEST_MODEL") or os.getenv("GAPGPT_MODEL")
          or os.getenv("MODEL_NAME_LLM") or "gpt-4o-mini")

# Enough characters of a text PDF to send without wasting tokens on boilerplate.
_MAX_TEXT_CHARS = 24_000

# Keep vision requests small + fast so they don't hit the gateway timeout: a raw
# phone photo can be several MB, which is slow to upload and slow for the model.
# We downscale to a sane long-edge and JPEG-encode; text stays perfectly legible.
_MAX_IMG_SIDE = int(os.getenv("MEDICAL_TEST_MAX_IMG", "1600"))   # px, long edge
_PDF_DPI = int(os.getenv("MEDICAL_TEST_PDF_DPI", "120"))         # render dpi
_TIMEOUT = float(os.getenv("MEDICAL_TEST_TIMEOUT", "180"))       # seconds per request
_MAX_RETRIES = int(os.getenv("MEDICAL_TEST_MAX_RETRIES", "2"))

_SYSTEM_PROMPT = (
    "You are a clinical laboratory data extractor. You are given the raw text or an "
    "image of a medical test report (blood work, urinalysis, addiction/toxicology "
    "screen, hormone panel, etc.). Extract ONLY the test results and return strict "
    "JSON. Rules:\n"
    "- IGNORE all patient-identifying information: name, date of birth, age, sex, "
    "medical record number, accession number, ordering provider, address, phone.\n"
    "- Group results into their panels/sections exactly as printed (e.g. "
    "'Complete Blood Count', 'Lipid Panel', 'Chemical Examination').\n"
    "- For every analyte capture: name, result (the value as printed), flag "
    "(H, L, A, or empty if none/normal), unit, and reference (the reference "
    "interval/range as printed).\n"
    "- Preserve values, units and ranges verbatim; do not convert or invent them.\n"
    "- If a field is absent, use an empty string.\n"
    "- 'summary' must be one or two short plain-language sentences focusing on "
    "any out-of-range values; if everything is normal, say so.\n"
    "Return ONLY a JSON object of this exact shape:\n"
    "{\n"
    '  "report_type": "string (e.g. Complete Blood Count, Urinalysis)",\n'
    '  "lab_name": "string",\n'
    '  "specimen": "string",\n'
    '  "collected_on": "string",\n'
    '  "reported_on": "string",\n'
    '  "panels": [\n'
    '    {"name": "string", "analytes": [\n'
    '      {"name": "string", "result": "string", "flag": "string", '
    '"unit": "string", "reference": "string"}\n'
    "    ]}\n"
    "  ],\n"
    '  "summary": "string"\n'
    "}"
)


# ---- file → LLM input -----------------------------------------------------
def _pdf_text(path: str) -> str:
    """Extract embedded text from a PDF (empty string if it is a scan)."""
    try:
        import fitz  # PyMuPDF
    except Exception:  # pragma: no cover - dependency always present in this app
        return ""
    text_parts = []
    with fitz.open(path) as doc:
        for page in doc:
            text_parts.append(page.get_text("text"))
    return "\n".join(text_parts).strip()


def _pdf_to_image_data_urls(path: str, max_pages: int = 3) -> list[str]:
    """Render the first pages of a PDF to compact JPEG data-URLs for the vision model."""
    import fitz  # PyMuPDF

    urls = []
    with fitz.open(path) as doc:
        for page in doc[:max_pages]:
            pix = page.get_pixmap(dpi=_PDF_DPI)
            try:
                data, mime = pix.tobytes("jpeg"), "jpeg"
            except Exception:  # pragma: no cover - alpha/CMYK edge cases
                data, mime = pix.tobytes("png"), "png"
            b64 = base64.b64encode(data).decode("ascii")
            urls.append(f"data:image/{mime};base64,{b64}")
    return urls


def _image_data_url(path: str) -> str:
    """Downscale + JPEG-encode the image so the vision request stays small/fast."""
    try:
        from PIL import Image
        img = Image.open(path)
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")
        w, h = img.size
        longest = max(w, h)
        if longest > _MAX_IMG_SIDE:
            s = _MAX_IMG_SIDE / float(longest)
            img = img.resize((max(1, int(w * s)), max(1, int(h * s))))
        buf = io.BytesIO()
        img.convert("RGB").save(buf, format="JPEG", quality=85, optimize=True)
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        return f"data:image/jpeg;base64,{b64}"
    except Exception as e:  # noqa: BLE001 - fall back to sending the raw bytes
        logger.warning("medical test image downscale failed (%s) — sending raw", e)
        ext = os.path.splitext(path)[1].lower().lstrip(".") or "png"
        mime = ("image/jpeg" if ext in ("jpg", "jpeg")
                else "image/webp" if ext == "webp" else "image/png")
        with open(path, "rb") as fh:
            b64 = base64.b64encode(fh.read()).decode("ascii")
        return f"data:{mime};base64,{b64}"


def _build_user_content(path: str) -> list[dict]:
    """Build the multimodal user message content for one document."""
    ext = os.path.splitext(path)[1].lower()

    if ext == ".pdf":
        text = _pdf_text(path)
        if len(text) >= 80:  # a real text PDF — cheaper & more reliable than vision
            return [{
                "type": "text",
                "text": "Extract the test results from this report text:\n\n"
                        + text[:_MAX_TEXT_CHARS],
            }]
        # scanned PDF → send page images
        content = [{"type": "text", "text": "Extract the test results from this report:"}]
        for url in _pdf_to_image_data_urls(path):
            content.append({"type": "image_url", "image_url": {"url": url}})
        return content

    if ext in (".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"):
        return [
            {"type": "text", "text": "Extract the test results from this report image:"},
            {"type": "image_url", "image_url": {"url": _image_data_url(path)}},
        ]

    # Unknown type — try to read it as text.
    try:
        with open(path, "r", errors="ignore") as fh:
            return [{"type": "text", "text": fh.read()[:_MAX_TEXT_CHARS]}]
    except Exception:
        raise ValueError(f"unsupported test file type: {ext or 'unknown'}")


# ---- normalisation --------------------------------------------------------
def _clean_str(v) -> str:
    return "" if v is None else str(v).strip()


def _normalise(data: dict) -> dict:
    """Coerce the model's JSON into our stored shape and count abnormals."""
    panels_out = []
    abnormal = 0
    for panel in data.get("panels") or []:
        if not isinstance(panel, dict):
            continue
        analytes_out = []
        for a in panel.get("analytes") or []:
            if not isinstance(a, dict):
                continue
            flag = _clean_str(a.get("flag")).upper()
            if flag in ("N", "NORMAL", "NONE", "-"):
                flag = ""
            if flag:
                abnormal += 1
            name = _clean_str(a.get("name"))
            if not name:
                continue
            analytes_out.append({
                "name": name,
                "result": _clean_str(a.get("result")),
                "flag": flag,
                "unit": _clean_str(a.get("unit")),
                "reference": _clean_str(a.get("reference")),
            })
        if analytes_out:
            panels_out.append({
                "name": _clean_str(panel.get("name")) or "Results",
                "analytes": analytes_out,
            })
    return {
        "report_type": _clean_str(data.get("report_type")),
        "lab_name": _clean_str(data.get("lab_name")),
        "specimen": _clean_str(data.get("specimen")),
        "collected_on": _clean_str(data.get("collected_on")),
        "reported_on": _clean_str(data.get("reported_on")),
        "panels": panels_out,
        "abnormal_count": abnormal,
        "summary": _clean_str(data.get("summary")),
    }


# ---- public entry point ---------------------------------------------------
def extract_medical_test(medical_test) -> "object":
    """Run the LLM extraction for one :class:`MedicalTest` and persist results.

    Best-effort: never raises. Returns the (updated) MedicalTest instance.
    """
    from openai import OpenAI

    mt = medical_test
    try:
        if not _API_KEY or not _BASE_URL:
            raise RuntimeError("LLM credentials (METIS_API_KEY / BASE_URL) are not configured")

        path = mt.file.path
        logger.info("🧪 extracting medical test | id=%s | file=%s | model=%s",
                    mt.id, os.path.basename(path), _MODEL)
        user_content = _build_user_content(path)

        client = OpenAI(api_key=_API_KEY, base_url=_BASE_URL,
                        timeout=_TIMEOUT, max_retries=_MAX_RETRIES)
        resp = client.chat.completions.create(
            model=_MODEL,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
        )
        raw = resp.choices[0].message.content or "{}"
        data = json.loads(raw)
        norm = _normalise(data)

        mt.report_type = norm["report_type"][:255]
        mt.lab_name = norm["lab_name"][:255]
        mt.specimen = norm["specimen"][:255]
        mt.collected_on = norm["collected_on"][:64]
        mt.reported_on = norm["reported_on"][:64]
        mt.panels = norm["panels"]
        mt.abnormal_count = norm["abnormal_count"]
        mt.summary = norm["summary"]
        mt.status = mt.STATUS_DONE
        mt.error = None
        mt.extracted_at = timezone.now()
        mt.save()
        logger.info("✅ medical test extracted | id=%s | panels=%d | abnormal=%d",
                    mt.id, len(norm["panels"]), norm["abnormal_count"])
    except Exception as e:  # noqa: BLE001 - must never break the caller
        logger.exception("🔥 medical test extraction failed | id=%s: %s", getattr(mt, "id", None), e)
        mt.status = mt.STATUS_FAILED
        mt.error = str(e)[:2000]
        try:
            mt.save(update_fields=["status", "error"])
        except Exception:
            pass
    return mt
