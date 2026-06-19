"""Pydantic models describing the Single-ROP API responses.

The request side is a ``multipart/form-data`` file upload (handled directly by
FastAPI's ``UploadFile`` in the routes), so only *output* schemas live here.
Modelling the output explicitly gives us an OpenAPI contract and validation,
and documents the exact shape the original Django pipeline produced.
"""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class LabelProb(BaseModel):
    """A predicted label together with its (string-formatted) probability."""

    class_name: Optional[str] = Field(None, description="Predicted Plus-disease label")
    class_prob: Optional[str] = Field(None, description="Confidence, formatted to 3 dp")


class StageProb(BaseModel):
    stage_name: Optional[str] = Field(None, description="Predicted ROP stage")
    stage_prob: Optional[str] = Field(None, description="Stage confidence, 3 dp")


class ZoneProb(BaseModel):
    zone_name: Optional[str] = Field(None, description="Predicted ROP zone")
    zone_prob: Optional[str] = Field(None, description="Zone confidence, 3 dp")


class Guidance(BaseModel):
    title: str = Field(..., description="Short headline for the clinical guidance")
    text: str = Field(..., description="Long-form management guidance")


class PredictionResult(BaseModel):
    """Full result for a single fundus image."""

    image_data: Optional[str] = Field(
        None, description="base64 data-URI of the vessel-overlay visualisation"
    )
    original_image_data: Optional[str] = Field(
        None, description="base64 data-URI of the original uploaded image"
    )
    inference_time: str = Field(..., description="Wall-clock pipeline latency, e.g. '1450 ms'")
    file_name: str = Field(..., description="Original file name echoed back")
    predictions: LabelProb
    stage_prediction: StageProb
    zone_prediction: ZoneProb
    final_decision: str = Field(..., description="Aggregate clinical recommendation")
    guidance: Guidance
    diagnostic_context: dict = Field(..., description="Structured summary for downstream LLMs")
    diagnostic_context_text: str
    llm_diagnostic_text: str


class ImageError(BaseModel):
    """Marker returned for an image that failed inside a batch."""

    error: str
    file_name: str


class AggregatedResult(BaseModel):
    """Majority-voted verdict across a batch plus the worst-image visualisation."""

    image_data: Optional[str] = None
    original_image_data: Optional[str] = None
    inference_time: str = ""
    file_name: str = ""
    predictions: LabelProb
    stage_prediction: StageProb
    zone_prediction: ZoneProb
    final_decision: str
    guidance: Guidance
    diagnostic_context: dict
    diagnostic_context_text: str
    llm_diagnostic_text: str
    worst_index: Optional[int] = Field(None, description="Index of the worst image in the batch")


class PredictResponse(BaseModel):
    """Envelope returned by the ``/predict`` endpoint."""

    aggregated: AggregatedResult
    # Each entry is either a full ``PredictionResult`` or an ``ImageError``; we
    # type it loosely as dict to allow both shapes without a discriminated union.
    per_image: list[dict] = Field(..., description="Per-image results or error markers")
