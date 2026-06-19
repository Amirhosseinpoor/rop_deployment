# How to Run Doctors-Marketplace Service

A marketplace of specialised, RAG-augmented medical chat assistants. Patients
browse doctors, open a chat session, and converse; each reply is grounded in the
doctor's private knowledge base (FAISS) when relevant. A protected "studio" lets
admins manage doctors and upload knowledge files.

## Prerequisites
- Python 3.10+
- An **OpenAI-compatible API key** for chat replies.
- First chat/upload downloads the embedding model (`BAAI/bge-small-en-v1.5` by
  default, ~130 MB) — allow internet access or pre-cache it.
- Database: defaults to local **SQLite** (auto-created). Use Postgres in prod.

## Setup
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

pip install -r requirements.txt
```

## Configuration (environment variables)

| Variable             | Default                       | Description                                             |
|----------------------|-------------------------------|---------------------------------------------------------|
| `DM_DATABASE_URL`    | `sqlite:///./doctors_marketplace.db` | SQLAlchemy DB URL.                               |
| `OPENAI_API_KEY`     | *(required for chat)*         | OpenAI-compatible API key.                              |
| `OPENAI_MODEL`       | `gpt-4o-mini`                 | Chat model name.                                        |
| `BASE_URL`           | *(empty = OpenAI default)*    | Optional OpenAI-compatible base URL (gateway).          |
| `DM_EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5`      | HuggingFace embedding model id or local path.           |
| `DM_EMBEDDING_DEVICE`| `cpu`                         | `cpu` or `cuda`.                                        |
| `VECTOR_ROOT`        | `./dm_doctor_vectors`         | Root dir for per-doctor FAISS indexes.                  |
| `DM_KNOWLEDGE_ROOT`  | `./dm_doctor_knowledge`       | Root dir for uploaded knowledge files.                  |
| `DM_STUDIO_API_KEY`  | *(empty = studio disabled)*   | Admin key required in the `X-Studio-Key` header.        |
| `DM_PORT`            | `8004`                        | Port for the convenience launcher.                      |

```bash
export OPENAI_API_KEY=sk-...
export DM_STUDIO_API_KEY="$(python -c 'import secrets;print(secrets.token_urlsafe(24))')"
```

## Run
From inside `ghofran/`:
```bash
cd ..
uvicorn doctors_marketplace.routes:app --host 0.0.0.0 --port 8004
```
Or self-launch (honours `DM_PORT`):
```bash
cd doctors_marketplace
python routes.py
```
Interactive docs: <http://localhost:8004/docs>

## Test

**1. Liveness**
```bash
curl http://localhost:8004/health
```

**2. Seed the built-in doctor catalogue (studio)**
```bash
curl -X POST http://localhost:8004/studio/seed -H "X-Studio-Key: $DM_STUDIO_API_KEY"
# -> {"created": 8}
```

**3. Browse doctors**
```bash
curl http://localhost:8004/doctors
```

**4. Open a chat session with a doctor**
```bash
SID=$(curl -s -X POST http://localhost:8004/sessions -H 'Content-Type: application/json' \
  -d '{"user_id":"patient-1","slug":"diabetes"}' | python -c 'import sys,json;print(json.load(sys.stdin)["id"])')
echo "$SID"
```

**5. Send a message (RAG-grounded LLM reply)**
```bash
curl -X POST "http://localhost:8004/sessions/$SID/send" -H 'Content-Type: application/json' \
  -d '{"user_id":"patient-1","message":"سلام، قند ناشتای من ۱۸۰ است. چه کنم؟"}'
```

**6. Upload a knowledge file for a doctor (studio)**
```bash
curl -X POST "http://localhost:8004/studio/doctors/diabetes/knowledge" \
  -H "X-Studio-Key: $DM_STUDIO_API_KEY" \
  -F "title=Diabetes Guidelines" -F "file=@/path/to/guidelines.pdf"
```

## Integration notes
- `user_id` is any stable string the caller controls. To integrate with USAC,
  pass the `sub` (username) from a verified USAC JWT.
- The studio gate is a shared admin key (`X-Studio-Key`); put this service behind
  a gateway that enforces the original "superuser only" policy if needed.
```
