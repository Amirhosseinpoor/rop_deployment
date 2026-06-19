# Test-Analysis (health) service image.
# Bundles the hypertension model; the report pipeline's PDF corpus and drugstore
# CSV are mounted at runtime (see docker-compose.yml).
#
# Note: the chat assistant's `medications_finder_tool` needs Chrome + Selenium,
# which are intentionally NOT installed here (large, and only one optional tool
# uses them). Doctor finding and the other features work without them.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# sentence-transformers (report pipeline) pulls torch; CPU-only build.
RUN pip install --upgrade pip \
    && pip install --index-url https://download.pytorch.org/whl/cpu torch==2.6.0

COPY requirements.txt ./test_analysis/requirements.txt
RUN pip install -r ./test_analysis/requirements.txt

# Copy code + bundled model/ (hypertension joblib) + templates/.
COPY . ./test_analysis/

EXPOSE 8006

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8006/health')" || exit 1

CMD ["uvicorn", "test_analysis.routes:app", "--host", "0.0.0.0", "--port", "8006"]
