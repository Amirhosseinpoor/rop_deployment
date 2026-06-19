"""Pydantic response models for the Double-ROP service.

Requests are ``multipart/form-data`` uploads (two files), handled directly in
the routes, so only output schemas are defined here.
"""
from __future__ import annotations

from pydantic import BaseModel, Field


class EyePrediction(BaseModel):
    """A single classification head's output."""

    label: str = Field(..., description="Predicted class name")
    probability: str = Field(..., description="Confidence, formatted to 4 dp")


class ImagePreviews(BaseModel):
    """base64 PNG data-URIs of the two uploaded eyes."""

    left: str
    right: str


class PredictResponse(BaseModel):
    """Full response of the ``/predict`` endpoint."""

    left_eye: EyePrediction = Field(..., description="Left-eye classification")
    right_eye: EyePrediction = Field(..., description="Right-eye classification")
    z_class: EyePrediction = Field(..., description="Combined binocular (Z) class")
    image_data: ImagePreviews
    inference_time: float = Field(..., description="Inference duration in seconds")
