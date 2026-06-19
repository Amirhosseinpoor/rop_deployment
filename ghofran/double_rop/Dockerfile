# Double-ROP service image.
# Build context = this directory (ghofran/double_rop).
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# CPU-only torch (smaller, no GPU runtime needed). Uses Pillow, not OpenCV, so no
# extra system libraries are required.
RUN pip install --upgrade pip \
    && pip install --index-url https://download.pytorch.org/whl/cpu \
        torch==2.6.0 torchvision==0.21.0

COPY requirements.txt ./double_rop/requirements.txt
RUN pip install -r ./double_rop/requirements.txt

# Copy code + bundled model/ (EyeNet checkpoint) + templates/.
COPY . ./double_rop/

EXPOSE 8002

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8002/health')" || exit 1

CMD ["uvicorn", "double_rop.routes:app", "--host", "0.0.0.0", "--port", "8002"]
