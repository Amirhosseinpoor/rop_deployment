"""Pydantic request/response models for the USAC service."""
from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict, EmailStr, Field


# --------------------------------------------------------------------------- #
# Auth
# --------------------------------------------------------------------------- #
class LoginRequest(BaseModel):
    username: str = Field(..., min_length=1)
    password: str = Field(..., min_length=1)


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    role: Optional[str] = Field(None, description="The authenticated user's role, if any")
    redirect: str = Field(..., description="Suggested landing route based on role")


# --------------------------------------------------------------------------- #
# Registration
# --------------------------------------------------------------------------- #
class ManagerSignupRequest(BaseModel):
    """Payload to register a manager and create their company."""

    username: str = Field(..., min_length=1)
    email: EmailStr
    password: str = Field(..., min_length=8)
    phone: Optional[str] = None

    company_name: str = Field(..., min_length=1)
    company_address: str
    company_email: EmailStr
    company_phone: str


class StaffSignupRequest(BaseModel):
    """Payload to register a doctor/employee against an invitation."""

    username: str = Field(..., min_length=1)
    email: EmailStr
    password: str = Field(..., min_length=8)
    national_code: str = Field(..., min_length=10, max_length=10, description="10-digit national code")
    phone: Optional[str] = None


# --------------------------------------------------------------------------- #
# Output models (ORM-backed via from_attributes)
# --------------------------------------------------------------------------- #
class UserOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    username: str
    email: str
    is_active: bool


class CompanyOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    address: str
    email: str
    phone: str


class InvitationRequest(BaseModel):
    national_code: str = Field(..., min_length=10, max_length=10)
    role: str = Field(..., description="Role to grant: manager / doctor / employee")


class InvitationOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    national_code: str
    role: str
    note: str
    used_by_id: Optional[int] = None
    created_at: datetime
    used_at: Optional[datetime] = None


class MemberOut(BaseModel):
    """A company member (user) with their role, for dashboards."""

    id: int
    username: str
    email: str
    role: Optional[str] = None


class ManagerDashboardOut(BaseModel):
    """The membership view a manager sees (analytics handled by other services)."""

    company: Optional[CompanyOut] = None
    doctors: list[MemberOut]
    employees: list[MemberOut]
    invitations: list[InvitationOut]
