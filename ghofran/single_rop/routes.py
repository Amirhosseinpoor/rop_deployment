"""FastAPI layer for the Single-ROP service.

This module is deliberately thin: it converts HTTP uploads into bytes, calls the
framework-free :mod:`service` layer, and serialises the result. All clinical
logic lives in ``service.py``.

Two objects are exported:

* ``router`` — an :class:`APIRouter` you can mount into a larger application.
* ``app``    — a ready-to-run :class:`FastAPI` instance with ``router`` already
  included, so ``uvicorn routes:app`` just works.
"""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import APIRouter, FastAPI, File, HTTPException, UploadFile, status

from . import service
from .config import get_settings
from .schemas import PredictResponse

router = APIRouter(tags=["single-rop"])


async def _read_uploads(files: list[UploadFile]) -> list[tuple[bytes, str]]:
    """Read every upload into memory as ``(bytes, name)`` tuples.

    Reading here (rather than in the service) keeps the service layer decoupled
    from the framework's file abstraction.
    """
    out: list[tuple[bytes, str]] = []
    for f in files:
        data = await f.read()
        if not data:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Uploaded file '{f.filename}' is empty.",
            )
        out.append((data, f.filename or "uploaded_image.jpg"))
    return out


@router.get("/health", summary="Liveness probe")
def health() -> dict:
    """Return a simple liveness payload. Does not touch the heavy models."""
    return {"status": "ok", "service": "single_rop"}


@router.post(
    "/predict",
    response_model=PredictResponse,
    summary="Run the ROP pipeline on one or more fundus images",
)
async def predict(files: list[UploadFile] = File(..., description="One or more fundus images")) -> PredictResponse:
    """Analyse uploaded retinal fundus images for Retinopathy of Prematurity.

    Accepts a ``multipart/form-data`` body with one or more files under the
    field name ``files``. Returns the aggregated (majority-voted) verdict and a
    per-image breakdown, mirroring the original Django ``/predict`` contract.
    """
    if not files:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No files received. Use multipart/form-data with field name 'files'.",
        )

    images = await _read_uploads(files)
    try:
        aggregated, per_image = service.predict_many_images(images)
    except Exception as ex:  # noqa: BLE001 - surface pipeline failures as 400
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(ex))

    return PredictResponse(aggregated=aggregated, per_image=per_image)


@asynccontextmanager
async def _lifespan(_: FastAPI):
    """Warm the models at start-up so the first request isn't slow.

    Failures are swallowed (logged via print) so the service can still start in
    environments where weights are missing — endpoints will then error per-call,
    which is easier to diagnose than a crash-on-boot.
    """
    try:
        service.warmup_models()
    except Exception as ex:  # noqa: BLE001
        print(f"[single_rop] model warm-up skipped: {ex}")
    yield


app = FastAPI(
    title="Single-ROP Service",
    description="Automated detection & classification of Retinopathy of Prematurity.",
    version="1.0.0",
    lifespan=_lifespan,
)
app.include_router(router)


if __name__ == "__main__":
    # Convenience launcher: `python -m single_rop.routes` or `python routes.py`.
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=get_settings().port)
