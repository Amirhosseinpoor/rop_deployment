"""FastAPI layer for the Test-Analysis (health) service.

Exposes the three AI capabilities. Each endpoint is a thin adapter over
:mod:`service`; the heavy ML/LLM imports happen lazily inside the service layer so
the app boots fast and a missing optional dependency only affects its own feature.

Exports ``router`` (mountable) and ``app`` (runnable via ``uvicorn routes:app``).
"""
from __future__ import annotations

from fastapi import APIRouter, FastAPI, HTTPException, status

from . import service
from .config import get_settings
from .schemas import (
    ChatRequest,
    ChatResponse,
    HypertensionRequest,
    HypertensionResponse,
    ReportRequest,
    ReportResponse,
)

router = APIRouter(tags=["test-analysis"])


@router.get("/health", summary="Liveness probe")
def health() -> dict:
    """Return a simple liveness payload."""
    return {"status": "ok", "service": "test_analysis"}


@router.post(
    "/predict/hypertension",
    response_model=HypertensionResponse,
    summary="Predict hypertension risk from patient vitals",
)
def predict_hypertension(payload: HypertensionRequest) -> HypertensionResponse:
    """Score a patient's hypertension risk with the trained scikit-learn model."""
    try:
        result = service.predict_hypertension(payload.model_dump())
    except Exception as ex:  # noqa: BLE001
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(ex))
    return HypertensionResponse(result=result)


@router.post(
    "/chat",
    response_model=ChatResponse,
    summary="Health chat assistant (finds doctors & medications)",
)
def chat(payload: ChatRequest) -> ChatResponse:
    """Run one turn of the tool-calling health assistant.

    Returns the assistant's reply and, when a finder tool was used, the raw
    doctor/medication results for the frontend to render separately.
    """
    history = [m.model_dump() for m in payload.history]
    try:
        reply, finder_results = service.chat(
            payload.message,
            history,
            disease_results=payload.disease_results,
            personal_information=payload.personal_information,
        )
    except Exception as ex:  # noqa: BLE001
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(ex))
    return ChatResponse(reply=reply, finder_results=finder_results, tool_called=finder_results is not None)


@router.post(
    "/report",
    response_model=ReportResponse,
    summary="Generate a full health-analysis report (RAG pipeline)",
)
def report(payload: ReportRequest) -> ReportResponse:
    """Run the 4-stage pipeline and return the final Persian medical report.

    This call is heavy (multiple LLM round-trips + RAG); expect tens of seconds.
    """
    try:
        text = service.generate_report(payload.profile_text_summary, payload.selected_model)
    except Exception as ex:  # noqa: BLE001
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(ex))
    return ReportResponse(report=text)


app = FastAPI(
    title="Test-Analysis (Health) Service",
    description="Hypertension prediction, health chat assistant, and health-analysis reports.",
    version="1.0.0",
)
app.include_router(router)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=get_settings().port)
