"""Pydantic request/response models for the Doctors-Marketplace service."""
from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


# --------------------------------------------------------------------------- #
# Doctors
# --------------------------------------------------------------------------- #
class DoctorOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    slug: str
    specialization: str
    persona: str
    headline: str
    bio: str
    name_fa: str
    specialization_fa: str
    headline_fa: str
    bio_fa: str
    tags_fa: str
    is_active: bool
    created_at: datetime


class DoctorCreate(BaseModel):
    """Fields accepted when creating a doctor in the studio."""

    name: str = Field("", description="English name")
    name_fa: str = Field("", description="Persian name")
    slug: Optional[str] = Field(None, description="Auto-generated from name when omitted")
    specialization: str = ""
    specialization_fa: str = ""
    persona: str = "kind"
    headline: str = ""
    headline_fa: str = ""
    bio: str = ""
    bio_fa: str = ""
    tags_fa: str = ""
    system_prompt: str = ""
    is_active: bool = True


class DoctorUpdate(BaseModel):
    """Partial update — every field optional."""

    name: Optional[str] = None
    name_fa: Optional[str] = None
    specialization: Optional[str] = None
    specialization_fa: Optional[str] = None
    persona: Optional[str] = None
    headline: Optional[str] = None
    headline_fa: Optional[str] = None
    bio: Optional[str] = None
    bio_fa: Optional[str] = None
    tags_fa: Optional[str] = None
    system_prompt: Optional[str] = None
    is_active: Optional[bool] = None


# --------------------------------------------------------------------------- #
# Chat
# --------------------------------------------------------------------------- #
class SessionCreate(BaseModel):
    user_id: str = Field(..., description="Stable identifier of the chatting user")
    slug: str = Field(..., description="Slug of the doctor to chat with")


class SessionOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    user_id: str
    doctor_id: int
    title: str
    created_at: datetime


class MessageOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    role: str
    content: str
    created_at: datetime


class SendMessageRequest(BaseModel):
    user_id: str = Field(..., description="Must own the session")
    message: str = Field(..., min_length=1)


class SendMessageResponse(BaseModel):
    ok: bool = True
    reply: str


# --------------------------------------------------------------------------- #
# Knowledge base
# --------------------------------------------------------------------------- #
class KnowledgeOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    title: str
    file_path: str
    created_at: datetime


class SeedResponse(BaseModel):
    created: int = Field(..., description="Number of doctors created from the catalogue")
