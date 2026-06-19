# Doctors-Marketplace service image (LLM + per-doctor FAISS RAG).
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# sentence-transformers pulls in torch; install the CPU-only build first.
RUN pip install --upgrade pip \
    && pip install --index-url https://download.pytorch.org/whl/cpu torch==2.6.0

COPY requirements.txt ./doctors_marketplace/requirements.txt
RUN pip install -r ./doctors_marketplace/requirements.txt

COPY . ./doctors_marketplace/

EXPOSE 8004

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8004/health')" || exit 1

CMD ["uvicorn", "doctors_marketplace.routes:app", "--host", "0.0.0.0", "--port", "8004"]
