"""FastAPI layer for the Double-ROP service.

Thin adapter: reads the two uploaded eye images, times the call, delegates to
:mod:`service`, and serialises the result.

Exports ``router`` (mountable) and ``app`` (runnable via ``uvicorn routes:app``).
"""
from __future__ import annotations

import time
from contextlib import asynccontextmanager

from fastapi import APIRouter, FastAPI, File, HTTPException, UploadFile, status

from . import service
from .config import get_settings
from .schemas import PredictResponse

router = APIRouter(tags=["double-rop"])


@router.get("/health", summary="Liveness probe")
def health() -> dict:
    """Return a simple liveness payload without loading the model."""
    return {"status": "ok", "service": "double_rop"}


@router.post(
    "/predict",
    response_model=PredictResponse,
    summary="Classify a left/right eye image pair",
)
async def predict(
    left_file: UploadFile = File(..., description="Left-eye image"),
    right_file: UploadFile = File(..., description="Right-eye image"),
) -> PredictResponse:
    """Run binocular keratoconus/corneal classification on a pair of eye images.

    Accepts ``multipart/form-data`` with two files named ``left_file`` and
    ``right_file``. Both are required (matching the original Django contract).
    """
    left_bytes = await left_file.read()
    right_bytes = await right_file.read()
    if not left_bytes or not right_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Both left and right eye images are required and must be non-empty.",
        )

    start = time.time()
    try:
        result = service.get_prediction(left_bytes, right_bytes)
    except ValueError as ex:  # malformed image
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(ex))
    except Exception as ex:  # noqa: BLE001 - inference failure
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(ex)
        )

    result["inference_time"] = time.time() - start
    return PredictResponse(**result)


@asynccontextmanager
async def _lifespan(_: FastAPI):
    """Warm the model at start-up; tolerate missing weights so boot never fails."""
    try:
        service.warmup_model()
    except Exception as ex:  # noqa: BLE001
        print(f"[double_rop] model warm-up skipped: {ex}")
    yield


app = FastAPI(
    title="Double-ROP Service",
    description="Binocular keratoconus / corneal classification from a pair of eye images.",
    version="1.0.0",
    lifespan=_lifespan,
)
app.include_router(router)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=get_settings().port)
