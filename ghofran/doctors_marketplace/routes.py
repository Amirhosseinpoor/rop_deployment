"""FastAPI layer for the Doctors-Marketplace service.

Public endpoints: browse doctors, open a chat session, send messages.
Studio endpoints (doctor CRUD + KB management): protected by an ``X-Studio-Key``
header that must equal ``DM_STUDIO_API_KEY`` (the standalone replacement for
Django's ``superuser`` gate). Studio is disabled if the key isn't configured.

Exports ``router`` (mountable) and ``app`` (runnable via ``uvicorn routes:app``).
"""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import (
    APIRouter,
    Depends,
    FastAPI,
    File,
    Form,
    Header,
    HTTPException,
    UploadFile,
    status,
)
from sqlalchemy.orm import Session

from . import service
from .config import get_settings
from .database import get_db, init_db
from .schemas import (
    DoctorCreate,
    DoctorOut,
    DoctorUpdate,
    KnowledgeOut,
    MessageOut,
    SeedResponse,
    SendMessageRequest,
    SendMessageResponse,
    SessionCreate,
    SessionOut,
)
from .service import ServiceError

router = APIRouter(tags=["doctors-marketplace"])

_CODE_TO_STATUS = {
    "not_found": status.HTTP_404_NOT_FOUND,
    "bad_request": status.HTTP_400_BAD_REQUEST,
}


def _raise(err: ServiceError) -> None:
    """Translate a domain error into an HTTPException."""
    raise HTTPException(
        status_code=_CODE_TO_STATUS.get(err.code, status.HTTP_400_BAD_REQUEST),
        detail=err.message,
    )


def require_studio_key(x_studio_key: str | None = Header(default=None)) -> None:
    """Gate studio endpoints behind the configured admin key (fail closed)."""
    configured = get_settings().studio_api_key
    if not configured:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Studio is disabled: set DM_STUDIO_API_KEY to enable admin endpoints.",
        )
    if x_studio_key != configured:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid studio key."
        )


# --------------------------------------------------------------------------- #
# Health
# --------------------------------------------------------------------------- #
@router.get("/health", summary="Liveness probe")
def health() -> dict:
    """Return a simple liveness payload."""
    return {"status": "ok", "service": "doctors_marketplace"}


# --------------------------------------------------------------------------- #
# Public: browse + chat
# --------------------------------------------------------------------------- #
@router.get("/doctors", response_model=list[DoctorOut], summary="List active doctors")
def list_doctors(db: Session = Depends(get_db)) -> list[DoctorOut]:
    """Return all active doctors (the public marketplace listing)."""
    return [DoctorOut.model_validate(d) for d in service.list_active_doctors(db)]


@router.get("/doctors/{slug}", response_model=DoctorOut, summary="Get one doctor")
def get_doctor(slug: str, db: Session = Depends(get_db)) -> DoctorOut:
    """Return a single active doctor by slug."""
    try:
        return DoctorOut.model_validate(service.get_doctor_by_slug(db, slug))
    except ServiceError as e:
        _raise(e)


@router.post(
    "/sessions",
    response_model=SessionOut,
    status_code=status.HTTP_201_CREATED,
    summary="Open a chat session with a doctor",
)
def open_session(payload: SessionCreate, db: Session = Depends(get_db)) -> SessionOut:
    """Create a session and seed it with the doctor's system prompt."""
    try:
        return SessionOut.model_validate(
            service.create_session(db, user_id=payload.user_id, slug=payload.slug)
        )
    except ServiceError as e:
        _raise(e)


@router.get(
    "/sessions/{session_id}/messages",
    response_model=list[MessageOut],
    summary="List messages in a session",
)
def list_messages(
    session_id: str,
    user_id: str,
    db: Session = Depends(get_db),
) -> list[MessageOut]:
    """Return the full message history of a session owned by ``user_id``."""
    try:
        session = service.get_session(db, session_id=session_id, user_id=user_id)
    except ServiceError as e:
        _raise(e)
    return [MessageOut.model_validate(m) for m in session.messages]


@router.post(
    "/sessions/{session_id}/send",
    response_model=SendMessageResponse,
    summary="Send a message and get the assistant's reply",
)
def send_message(
    session_id: str,
    payload: SendMessageRequest,
    db: Session = Depends(get_db),
) -> SendMessageResponse:
    """Persist the user's message, run the RAG+LLM turn, and return the reply."""
    try:
        reply = service.send_message(
            db, session_id=session_id, user_id=payload.user_id, user_text=payload.message
        )
    except ServiceError as e:
        _raise(e)
    return SendMessageResponse(ok=True, reply=reply)


# --------------------------------------------------------------------------- #
# Studio (admin) — protected by X-Studio-Key
# --------------------------------------------------------------------------- #
@router.post(
    "/studio/seed",
    response_model=SeedResponse,
    dependencies=[Depends(require_studio_key)],
    summary="Seed the built-in doctor catalogue",
)
def seed(db: Session = Depends(get_db)) -> SeedResponse:
    """Create any missing doctors from the bundled catalogue (idempotent)."""
    return SeedResponse(created=service.seed_doctors(db))


@router.get(
    "/studio/doctors",
    response_model=list[DoctorOut],
    dependencies=[Depends(require_studio_key)],
    summary="List all doctors (active or not)",
)
def studio_list(db: Session = Depends(get_db)) -> list[DoctorOut]:
    """Return every doctor for administration."""
    return [DoctorOut.model_validate(d) for d in service.list_all_doctors(db)]


@router.post(
    "/studio/doctors",
    response_model=DoctorOut,
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(require_studio_key)],
    summary="Create a doctor",
)
def studio_create(payload: DoctorCreate, db: Session = Depends(get_db)) -> DoctorOut:
    """Create a new doctor (slug auto-generated from name if omitted)."""
    try:
        return DoctorOut.model_validate(service.create_doctor(db, payload.model_dump()))
    except ServiceError as e:
        _raise(e)


@router.patch(
    "/studio/doctors/{slug}",
    response_model=DoctorOut,
    dependencies=[Depends(require_studio_key)],
    summary="Update a doctor",
)
def studio_update(slug: str, payload: DoctorUpdate, db: Session = Depends(get_db)) -> DoctorOut:
    """Update mutable fields of a doctor."""
    try:
        return DoctorOut.model_validate(
            service.update_doctor(db, slug, payload.model_dump(exclude_unset=True))
        )
    except ServiceError as e:
        _raise(e)


@router.delete(
    "/studio/doctors/{slug}",
    status_code=status.HTTP_204_NO_CONTENT,
    dependencies=[Depends(require_studio_key)],
    summary="Delete a doctor",
)
def studio_delete(slug: str, db: Session = Depends(get_db)) -> None:
    """Delete a doctor and all of their sessions/knowledge."""
    try:
        service.delete_doctor(db, slug)
    except ServiceError as e:
        _raise(e)


@router.get(
    "/studio/doctors/{slug}/knowledge",
    response_model=list[KnowledgeOut],
    dependencies=[Depends(require_studio_key)],
    summary="List a doctor's knowledge files",
)
def studio_kb_list(slug: str, db: Session = Depends(get_db)) -> list[KnowledgeOut]:
    """Return the doctor's uploaded knowledge files."""
    try:
        return [KnowledgeOut.model_validate(i) for i in service.list_knowledge(db, slug)]
    except ServiceError as e:
        _raise(e)


@router.post(
    "/studio/doctors/{slug}/knowledge",
    response_model=KnowledgeOut,
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(require_studio_key)],
    summary="Upload & index a knowledge file",
)
async def studio_kb_upload(
    slug: str,
    title: str = Form(...),
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
) -> KnowledgeOut:
    """Save an uploaded file and (synchronously) index it into the doctor's RAG."""
    content = await file.read()
    if not content:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Uploaded file is empty."
        )
    try:
        item = service.add_knowledge(
            db, slug=slug, title=title, filename=file.filename or "upload", content=content
        )
    except ServiceError as e:
        _raise(e)
    return KnowledgeOut.model_validate(item)


@router.delete(
    "/studio/doctors/{slug}/knowledge/{item_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    dependencies=[Depends(require_studio_key)],
    summary="Delete a knowledge file",
)
def studio_kb_delete(slug: str, item_id: int, db: Session = Depends(get_db)) -> None:
    """Delete a knowledge row and its on-disk file."""
    try:
        service.delete_knowledge(db, slug=slug, item_id=item_id)
    except ServiceError as e:
        _raise(e)


@asynccontextmanager
async def _lifespan(_: FastAPI):
    """Create tables on start-up."""
    init_db()
    yield


app = FastAPI(
    title="Doctors-Marketplace Service",
    description="RAG-augmented specialised medical chat assistants.",
    version="1.0.0",
    lifespan=_lifespan,
)
app.include_router(router)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=get_settings().port)
