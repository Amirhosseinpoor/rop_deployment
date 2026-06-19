"""FastAPI layer for the USAC service.

Maps the Django views/forms onto a JSON+JWT API:

* ``POST /auth/login``                  -> obtain a JWT (replaces ``custom_login``)
* ``GET  /auth/me``                     -> current user info
* ``GET  /auth/redirect``               -> role-based landing route
* ``POST /signup/manager``              -> manager + company registration
* ``POST /signup/doctor`` / ``/employee`` -> invitation-gated staff registration
* ``POST /manager/invitations``         -> issue/re-target an invitation
* ``GET  /manager/dashboard``           -> company membership + invitations

Exports ``router`` (mountable) and ``app`` (runnable via ``uvicorn routes:app``).
"""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import APIRouter, Depends, FastAPI, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from . import models, service
from .config import get_settings
from .database import get_db, init_db
from .schemas import (
    CompanyOut,
    InvitationOut,
    InvitationRequest,
    LoginRequest,
    ManagerDashboardOut,
    ManagerSignupRequest,
    MemberOut,
    StaffSignupRequest,
    TokenResponse,
    UserOut,
)
from .security import create_access_token, decode_access_token
from .service import ServiceError

router = APIRouter(tags=["usac"])

# Bearer-token scheme; ``auto_error=True`` makes missing tokens a 403 automatically.
_bearer = HTTPBearer()

# Map a domain error code to an HTTP status. Centralised so every handler stays
# free of status-code bookkeeping.
_CODE_TO_STATUS = {
    "auth": status.HTTP_401_UNAUTHORIZED,
    "conflict": status.HTTP_409_CONFLICT,
    "not_found": status.HTTP_404_NOT_FOUND,
    "bad_request": status.HTTP_400_BAD_REQUEST,
}


def _raise(err: ServiceError) -> None:
    """Translate a :class:`ServiceError` into an :class:`HTTPException`."""
    raise HTTPException(
        status_code=_CODE_TO_STATUS.get(err.code, status.HTTP_400_BAD_REQUEST),
        detail=err.message,
    )


def _redirect_for(role: str | None) -> str:
    """Suggested landing route after login, mirroring ``role_based_redirect``."""
    if role == models.ROLE_MANAGER:
        return "manager_dashboard"
    if role == models.ROLE_DOCTOR:
        return "doctor_dashboard"
    return "dilemma"


def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(_bearer),
    db: Session = Depends(get_db),
) -> models.User:
    """Resolve the bearer token to a live ``User`` row, or 401."""
    try:
        payload = decode_access_token(credentials.credentials)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=str(e))
    username = payload.get("sub")
    user = service.get_user_by_username(db, username) if username else None
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="User no longer exists."
        )
    return user


def require_manager(
    user: models.User = Depends(get_current_user),
) -> models.User:
    """Dependency that allows only manager-role users through."""
    if service.role_of(user) != models.ROLE_MANAGER:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Manager role required."
        )
    return user


# --------------------------------------------------------------------------- #
# Health
# --------------------------------------------------------------------------- #
@router.get("/health", summary="Liveness probe")
def health() -> dict:
    """Return a simple liveness payload."""
    return {"status": "ok", "service": "usac"}


# --------------------------------------------------------------------------- #
# Auth
# --------------------------------------------------------------------------- #
@router.post("/auth/login", response_model=TokenResponse, summary="Authenticate and get a JWT")
def login(payload: LoginRequest, db: Session = Depends(get_db)) -> TokenResponse:
    """Verify credentials and return a signed access token plus a role hint."""
    try:
        user = service.authenticate(db, payload.username, payload.password)
    except ServiceError as e:
        _raise(e)
    role = service.role_of(user)
    token = create_access_token(user.username, extra_claims={"role": role})
    return TokenResponse(access_token=token, role=role, redirect=_redirect_for(role))


@router.get("/auth/me", response_model=UserOut, summary="Current authenticated user")
def me(user: models.User = Depends(get_current_user)) -> UserOut:
    """Return the profile of the user the bearer token belongs to."""
    return UserOut.model_validate(user)


@router.get("/auth/redirect", summary="Role-based landing route")
def redirect(user: models.User = Depends(get_current_user)) -> dict:
    """Return the route the frontend should navigate to for this user's role."""
    role = service.role_of(user)
    return {"role": role, "redirect": _redirect_for(role)}


# --------------------------------------------------------------------------- #
# Registration
# --------------------------------------------------------------------------- #
@router.post(
    "/signup/manager",
    response_model=UserOut,
    status_code=status.HTTP_201_CREATED,
    summary="Register a manager and create their company",
)
def signup_manager(payload: ManagerSignupRequest, db: Session = Depends(get_db)) -> UserOut:
    """Create a manager account together with the company they will own."""
    try:
        user = service.signup_manager(
            db,
            username=payload.username,
            email=payload.email,
            password=payload.password,
            phone=payload.phone,
            company_name=payload.company_name,
            company_address=payload.company_address,
            company_email=payload.company_email,
            company_phone=payload.company_phone,
        )
    except ServiceError as e:
        _raise(e)
    return UserOut.model_validate(user)


@router.post(
    "/signup/doctor",
    response_model=UserOut,
    status_code=status.HTTP_201_CREATED,
    summary="Register a doctor (requires an active invitation)",
)
def signup_doctor(payload: StaffSignupRequest, db: Session = Depends(get_db)) -> UserOut:
    """Register a doctor against a matching, unused company invitation."""
    return _signup_staff(payload, models.ROLE_DOCTOR, db)


@router.post(
    "/signup/employee",
    response_model=UserOut,
    status_code=status.HTTP_201_CREATED,
    summary="Register an employee (requires an active invitation)",
)
def signup_employee(payload: StaffSignupRequest, db: Session = Depends(get_db)) -> UserOut:
    """Register an employee against a matching, unused company invitation."""
    return _signup_staff(payload, models.ROLE_EMPLOYEE, db)


def _signup_staff(payload: StaffSignupRequest, role: str, db: Session) -> UserOut:
    """Shared body for the doctor/employee signup endpoints."""
    try:
        user = service.signup_staff(
            db,
            role=role,
            username=payload.username,
            email=payload.email,
            password=payload.password,
            national_code=payload.national_code,
            phone=payload.phone,
        )
    except ServiceError as e:
        _raise(e)
    return UserOut.model_validate(user)


# --------------------------------------------------------------------------- #
# Manager area
# --------------------------------------------------------------------------- #
@router.post(
    "/manager/invitations",
    response_model=InvitationOut,
    summary="Issue or re-target a staff invitation",
)
def create_invitation(
    payload: InvitationRequest,
    manager: models.User = Depends(require_manager),
    db: Session = Depends(get_db),
) -> InvitationOut:
    """Create (or re-target) an invitation scoped to the manager's company."""
    try:
        inv = service.create_or_update_invitation(
            db, manager=manager, national_code=payload.national_code, role=payload.role
        )
    except ServiceError as e:
        _raise(e)
    return InvitationOut.model_validate(inv)


@router.get(
    "/manager/dashboard",
    response_model=ManagerDashboardOut,
    summary="Company membership and invitations",
)
def manager_dashboard(
    manager: models.User = Depends(require_manager),
    db: Session = Depends(get_db),
) -> ManagerDashboardOut:
    """Return the manager's company, its doctors/employees, and invitations.

    Note: the original Django dashboard also computed health analytics (BMI,
    risks, opinions) from ``test_analysis`` data. That belongs to a different
    microservice; compose it at the API-gateway/frontend layer.
    """
    company = service.get_user_company(db, manager)
    if company is None:
        return ManagerDashboardOut(company=None, doctors=[], employees=[], invitations=[])

    def _members(role: str) -> list[MemberOut]:
        return [
            MemberOut(id=u.id, username=u.username, email=u.email, role=role)
            for u in service.list_company_members(db, company, role)
        ]

    return ManagerDashboardOut(
        company=CompanyOut.model_validate(company),
        doctors=_members(models.ROLE_DOCTOR),
        employees=_members(models.ROLE_EMPLOYEE),
        invitations=[
            InvitationOut.model_validate(i)
            for i in service.list_company_invitations(db, company)
        ],
    )


@asynccontextmanager
async def _lifespan(_: FastAPI):
    """Create database tables at start-up so the service is usable immediately."""
    init_db()
    yield


app = FastAPI(
    title="USAC Service",
    description="Users, companies, roles and invitations for the SurgiNote ecosystem.",
    version="1.0.0",
    lifespan=_lifespan,
)
app.include_router(router)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=get_settings().port)
