# USAC (auth / users / companies / invitations) service image.
# Lightweight: no ML stack, pure web + SQLAlchemy.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt ./usac/requirements.txt
RUN pip install --upgrade pip && pip install -r ./usac/requirements.txt

COPY . ./usac/

EXPOSE 8003

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8003/health')" || exit 1

CMD ["uvicorn", "usac.routes:app", "--host", "0.0.0.0", "--port", "8003"]
