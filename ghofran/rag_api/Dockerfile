# ROP-RAG service image (hybrid local + web RAG).
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# sentence-transformers + torch (CPU-only build).
RUN pip install --upgrade pip \
    && pip install --index-url https://download.pytorch.org/whl/cpu torch==2.6.0

COPY requirements.txt ./rag_api/requirements.txt
RUN pip install -r ./rag_api/requirements.txt

# Copy code + the bundled rag_documents/ corpus.
COPY . ./rag_api/

EXPOSE 8005

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8005/health')" || exit 1

CMD ["uvicorn", "rag_api.routes:app", "--host", "0.0.0.0", "--port", "8005"]
