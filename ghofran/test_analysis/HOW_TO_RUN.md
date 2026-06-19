# How to Run Test-Analysis (Health) Service

Three AI capabilities extracted from the original Django `test_analysis` app:

1. **`POST /predict/hypertension`** — scikit-learn hypertension-risk scoring.
2. **`POST /chat`** — a tool-calling health assistant that finds doctors and
   medications/pharmacies (web scrapers as tools).
3. **`POST /report`** — a 4-stage RAG pipeline that produces a full Persian
   medical report (disease → doctors → drugs → report).

> The original app's health-profile forms, doctor/manager dashboards and role
> checks are **out of scope** here — they are presentation/role logic that
> belongs with the USAC service. This service is the AI engine those screens call.

## Prerequisites
- Python 3.10+
- **Feature 1:** the `best_rf_hypertension_model.joblib` file — bundled in this
  service's `model/` directory and loaded automatically (override with
  `HYPERTENSION_MODEL_PATH`/`MODEL_DIR`).
- **Feature 2 (chat):** a GapGPT / OpenAI-compatible API key. The
  `medications_finder_tool` additionally needs Chrome + Chromedriver (Selenium);
  doctor finding works with plain HTTP.
- **Feature 3 (report):** a Metis / OpenAI-compatible API key, a running Ollama
  (only if you pass `selected_model=local_llama`), a `knowledge_base/` folder of
  PDFs, and `data_csv/drugstores.csv`. First run downloads the embedding model.

Each feature is independent: you can run and use feature 1 without configuring 2 or 3.

## Setup
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

pip install -r requirements.txt
# Optional, only for the chat medications finder:
# pip install selenium==4.27.1
```

## Configuration (environment variables)

| Variable                  | Default                                  | Used by | Description                              |
|---------------------------|------------------------------------------|---------|------------------------------------------|
| `HYPERTENSION_MODEL_PATH` | `${MODEL_DIR}/best_rf_hypertension_model.joblib` | 1 | Path to the joblib model.       |
| `MODEL_DIR`               | `model`                                  | 1       | Used to build the default model path.    |
| `GAPGPT_API_KEY`          | *(required for chat)*                    | 2       | Chat assistant LLM key.                  |
| `GAPGPT_BASE_URL`         | *(empty = OpenAI default)*               | 2       | Chat LLM base URL.                       |
| `GAPGPT_MODEL`            | `gpt-5-nano`                             | 2       | Chat model name.                         |
| `METIS_API_KEY`           | *(required for report)*                  | 3       | Pipeline LLM key.                        |
| `BASE_URL`                | *(empty = OpenAI default)*               | 3       | Pipeline LLM base URL.                   |
| `MODEL_NAME_LLM`          | `gpt-4o-mini`                            | 3       | Pipeline model name.                     |
| `OLLAMA_API_URL`          | `http://localhost:11434`                | 3       | Used when `selected_model=local_llama`.  |
| `LOCAL_MODEL_NAME`        | `llama3`                                 | 3       | Local model name.                        |
| `HEALTH_EMBEDDING_MODEL`  | `BAAI/bge-small-en-v1.5`                | 3       | Embedding model id/path.                 |
| `KNOWLEDGE_BASE_DIR`      | `knowledge_base`                        | 3       | PDF corpus folder.                       |
| `DRUGSTORES_CSV`          | `data_csv/drugstores.csv`               | 3       | Drugstore CSV path.                      |
| `HEALTH_PORT`             | `8006`                                   | —       | Port for the convenience launcher.       |

```bash
export MODEL_DIR=/home/amir/Desktop/rop/model
export GAPGPT_API_KEY=...          # for /chat
export METIS_API_KEY=...           # for /report
export KNOWLEDGE_BASE_DIR=/home/amir/Desktop/rop/knowledge_base
export DRUGSTORES_CSV=/home/amir/Desktop/rop/data_csv/drugstores.csv
```

## Run
From inside `ghofran/`:
```bash
cd ..
uvicorn test_analysis.routes:app --host 0.0.0.0 --port 8006
```
Or self-launch (honours `HEALTH_PORT`):
```bash
cd test_analysis
python routes.py
```
Interactive docs: <http://localhost:8006/docs>

## Test

**1. Liveness**
```bash
curl http://localhost:8006/health
```

**2. Hypertension prediction (feature 1 — no API keys needed)**
```bash
curl -X POST http://localhost:8006/predict/hypertension -H 'Content-Type: application/json' -d '{
  "male": 1, "age": 52, "currentSmoker": 0, "cigsPerDay": 0, "BPMeds": 1, "diabetes": 0,
  "totChol": 220, "sysBP": 140, "diaBP": 90, "BMI": 27.5, "heartRate": 75, "glucose": 80
}'
# -> {"result": "⚠️ Based on the model, the patient has a 74.5% probability of **having hypertension**."}
```

**3. Health chat assistant (feature 2 — needs GAPGPT_API_KEY)**
```bash
curl -X POST http://localhost:8006/chat -H 'Content-Type: application/json' -d '{
  "message": "سلام، می‌خواهم یک پزشک قلب در محله ابوذر پیدا کنم.",
  "history": []
}'
```

**4. Health-analysis report (feature 3 — needs METIS_API_KEY + corpus; slow)**
```bash
curl -X POST http://localhost:8006/report -H 'Content-Type: application/json' -d '{
  "profile_text_summary": "Male, 52, sysBP 140, diaBP 90, BMI 27.5, lives in tehran, region abuzar.",
  "selected_model": "cloud_gpt"
}'
```

## Notes
- Web scrapers (doctor/medication finders) target third-party sites
  (nobat.ir, doctoreto.com, darooyab.ir) and are inherently brittle if those
  sites change their markup — this matches the original behaviour.
- `/report` is intentionally synchronous here (the Django app ran it via Celery).
  It can take tens of seconds; run it behind a queue/gateway if you need async.
```
