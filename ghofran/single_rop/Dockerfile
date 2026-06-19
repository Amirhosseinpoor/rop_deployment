# Single-ROP service image.
# Build context = this directory (ghofran/single_rop). The package is copied to
# /app/single_rop so the import path `single_rop.routes:app` resolves with /app
# as the working directory.
FROM python:3.11-slim

# Fail fast and keep logs unbuffered; never write .pyc into the image layers.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# OpenCV (cv2) needs these shared libraries at runtime.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install CPU-only torch first (much smaller than the default CUDA wheels and
# needs no GPU runtime). Override by editing requirements / using a CUDA base
# image if you want GPU inference.
RUN pip install --upgrade pip \
    && pip install --index-url https://download.pytorch.org/whl/cpu \
        torch==2.6.0 torchvision==0.21.0

# Dependencies are installed before copying the code so the layer is cached
# across code changes.
COPY requirements.txt ./single_rop/requirements.txt
RUN pip install -r ./single_rop/requirements.txt

# Copy the rest of the service (code + bundled model/ + templates/).
COPY . ./single_rop/

EXPOSE 8001

# Liveness check used by Docker/compose.
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8001/health')" || exit 1

CMD ["uvicorn", "single_rop.routes:app", "--host", "0.0.0.0", "--port", "8001"]
