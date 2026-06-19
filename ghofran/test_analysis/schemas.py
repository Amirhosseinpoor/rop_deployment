"""Pydantic request/response models for the Test-Analysis service."""
from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


# --------------------------------------------------------------------------- #
# Hypertension prediction
# --------------------------------------------------------------------------- #
class HypertensionRequest(BaseModel):
    """The 12 model features (plus optional, unused-by-model location fields)."""

    male: int = Field(..., description="1 for male, 0 for female")
    age: int = Field(..., description="Age in years")
    currentSmoker: int = Field(..., description="1 if currently smoking, else 0")
    cigsPerDay: float = Field(..., description="Cigarettes per day")
    BPMeds: int = Field(..., description="1 if on blood-pressure medication, else 0")
    diabetes: int = Field(..., description="1 if diabetic, else 0")
    totChol: float = Field(..., description="Total cholesterol")
    sysBP: float = Field(..., description="Systolic blood pressure")
    diaBP: float = Field(..., description="Diastolic blood pressure")
    BMI: float = Field(..., description="Body mass index")
    heartRate: float = Field(..., description="Heart rate (bpm)")
    glucose: float = Field(..., description="Glucose level")
    city: Optional[str] = None
    region: Optional[str] = None
    insurance: Optional[float] = None


class HypertensionResponse(BaseModel):
    result: str = Field(..., description="Human-readable hypertension-risk statement")


# --------------------------------------------------------------------------- #
# Health chat assistant
# --------------------------------------------------------------------------- #
class ChatMessageItem(BaseModel):
    role: str = Field(..., description="'user' or 'assistant'")
    content: str


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, description="The user's latest message")
    history: list[ChatMessageItem] = Field(default_factory=list, description="Prior turns")
    # Optional patient context; defaults to a demo patient when omitted.
    disease_results: Optional[list[dict[str, Any]]] = Field(
        None, description="Patient disease context (disease, accuracy, refer_to, medications)"
    )
    personal_information: Optional[list[dict[str, Any]]] = Field(
        None, description="Patient profile (name, living_province, neighborhood, ...)"
    )


class ChatResponse(BaseModel):
    reply: str = Field(..., description="The assistant's reply text")
    finder_results: Optional[Any] = Field(
        None, description="Raw doctor/medication tool output for the frontend, if any"
    )
    tool_called: bool = Field(..., description="Whether a finder tool was invoked")


# --------------------------------------------------------------------------- #
# Health analysis report
# --------------------------------------------------------------------------- #
class ReportRequest(BaseModel):
    profile_text_summary: str = Field(
        ..., min_length=1, description="Free-text summary of the patient profile/vitals"
    )
    selected_model: str = Field(
        "cloud_gpt", description="'cloud_gpt' (Metis) or 'local_llama' (Ollama)"
    )


class ReportResponse(BaseModel):
    report: str = Field(..., description="The generated Persian medical report (Markdown)")
