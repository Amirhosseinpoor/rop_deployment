# SurgiNote — FastAPI Microservices (`ghofran/`)

This directory contains the original Django apps refactored into **independent,
self-contained FastAPI microservices**. Nothing here imports from the original
Django project; each service owns its config, persistence and dependencies and
can be installed and run on its own.

## Services & ports

| Service                | Port | What it does                                                        |
|------------------------|------|---------------------------------------------------------------------|
| `single_rop`           | 8001 | ROP detection from a single fundus image (segmentation + plus/stage/zone). |
| `double_rop`           | 8002 | Binocular keratoconus/corneal classification from a left/right eye pair.    |
| `usac`                 | 8003 | Users, companies, roles & invitations (JWT auth). Identity backbone.        |
| `doctors_marketplace`  | 8004 | RAG-augmented specialised medical chat assistants ("doctors").             |
| `rag_api`              | 8005 | Hybrid (local BM25+FAISS) + web RAG agentic API for the ROP assistant.    |
| `test_analysis`        | 8006 | Health AI: hypertension prediction, chat assistant, full report pipeline.  |

Every service exposes `GET /health` and interactive docs at `/docs`.

## Layout (per service)

```
<service>/
├── __init__.py
├── routes.py        # FastAPI APIRouter + a ready-to-run `app`
├── schemas.py       # Pydantic input/output models
├── service.py       # core business logic (no FastAPI dependency)
├── config.py        # env-driven settings (no hard-coded secrets)
├── requirements.txt # dependencies specific to this service
├── HOW_TO_RUN.md    # setup, configuration, run & test instructions
└── (extra modules only where strictly necessary, e.g. model.py, database.py)
```

Each service's `HOW_TO_RUN.md` is the authoritative guide for that service.

## Running with Docker (all services at once)

Each service has its own `Dockerfile`; `docker-compose.yml` builds and runs all
six (plus an Ollama embedding server used by `rag_api` and the report pipeline).

```bash
cd ghofran
cp .env.example .env          # fill in API keys & set USAC_JWT_SECRET
docker compose up --build     # build & start everything

# one-time: pull the embedding model Ollama serves
docker compose exec ollama ollama pull nomic-embed-text
```

Run a subset (e.g. just the imaging services, which need no keys):

```bash
docker compose up --build single_rop double_rop
```

Notes:
- **Images bake in the bundled `model/`, `templates/` and `rag_documents/`** so
  imaging services run with no extra setup. torch is installed CPU-only to keep
  images smaller and GPU-free; switch to a CUDA base image for GPU inference.
- **Persisted data** uses named volumes: `usac_data`, `dm_data` (SQLite),
  `ollama_data` (Ollama models), and `hf_cache` (downloaded HuggingFace models,
  shared so each is downloaded once).
- **The report corpus** is mounted read-only from the repo root into
  `test_analysis` (`../knowledge_base`, `../data_csv`) — run compose from
  `ghofran/` so those relative paths resolve.
- Each container exposes `GET /health`; compose uses it as a healthcheck.

## Running a service (without Docker)

Install in an isolated virtualenv, then launch from **inside `ghofran/`** so the
package import resolves (services are Python packages):

```bash
cd ghofran
uvicorn single_rop.routes:app --host 0.0.0.0 --port 8001
```

Each `routes.py` is also self-launching (`python <service>/routes.py`) and honours
its `*_PORT` env var. Because every service is on a distinct port, all six can run
simultaneously.

## Design notes

- **Independence over DRY.** Shared concepts (e.g. patient identity) are not
  imported across services; instead, services accept identifiers/inputs and can
  be composed at an API gateway or frontend.
- **Secrets via environment only.** Real keys that were hard-coded in the original
  code have been removed; missing optional secrets degrade gracefully.
- **Heavy ML/LLM imports are lazy** where possible, so services boot fast and a
  missing optional dependency only affects the feature that needs it.
- **Model weights & templates are bundled per service.** Each ML service ships
  its own `model/` directory (loaded by default) and each web-facing service ships
  a `templates/` directory copied from the original app. Override model locations
  with the documented env vars (`MODEL_DIR`, `KC_MODEL_PATH`, ...) if you prefer a
  shared location. Larger corpora (`KNOWLEDGE_BASE_DIR`, drugstore CSV) still point
  at the repo's assets via env vars.

### Bundled assets per service

| Service          | `model/`                                   | `templates/`                          |
|------------------|--------------------------------------------|---------------------------------------|
| `single_rop`     | 4 checkpoints (seg + plus + stage + zone)  | `index2.html`, `index.html`           |
| `double_rop`     | `best_model.pth` (EyeNet)                  | `index_double2.html`, `index_double.html` |
| `test_analysis`  | `best_rf_hypertension_model.joblib`        | `test_analysis/*` (profile, dashboard, ...) |
| `usac`           | —                                          | `usac/*`, `socialaccount/*`           |
| `doctors_marketplace` | —                                     | `doctors_marketplace/*` (public + studio) |
| `rag_api`        | — (downloads HF reranker)                  | — (pure API)                          |

> The templates are copied verbatim for reference / future server-rendered use;
> the FastAPI services themselves return JSON and do not render them.

## Out of scope (intentionally)

The Django apps also contained presentation/role plumbing — HTML templates,
manager/doctor dashboards, CSV-export views, profile CRUD forms — that read across
several apps. These overlap with `usac` and are UI concerns, so they are not
reproduced here. The microservices provide the underlying capabilities those
screens consumed; compose them at the gateway/frontend layer.
```
