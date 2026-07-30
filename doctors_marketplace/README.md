# Doctors Marketplace: Clinician Portal & Knowledge RAG

This module manages the professional marketplace and "Studio" environment, where doctors can maintain personalized **Retrieval-Augmented Generation (RAG)** knowledge bases.

---

### 1. Models (`models.py`)
- **`Doctor`**:
    - Stores metadata including `specialization` and `persona` (Kind, Analytical, etc.).
    - **`system_prompt`**: A large-text field used to seed and customize the behavior of the doctor's AI assistant.
- **`DoctorKnowledge`**:
    - **`file`**: Stores PDF/TXT uploads specific to a doctor's expertise.
    - **`vector_dir()`**: Dynamic method that computes the path for the doctor's private FAISS vector store.
- **`ChatSession` & `ChatMessage`**:
    - Persists conversational history between users and doctor-assistants with role tracking (System, User, Assistant).

---

### 2. Core Logic & AI Pipeline (`services/`)
#### **RAG Pipeline (`rag.py`):**
1.  **Ingestion**: `index_file_for_doctor` reads PDFs via **PyMuPDF** (much better Persian/Arabic handling than PyPDF2) and **NFKC-normalises** the text so presentation-form glyphs match user queries. Chunking is **token-aware** (tiktoken `RecursiveCharacterTextSplitter`, ~320 tokens / 60 overlap) with Persian-aware separators.
2.  **Embeddings**: GAPGPT **`text-embedding-3-large`** (3072-dim) over the OpenAI-compatible API — no local model / torch needed, so `runserver` stays light.
3.  **Indexing**: Builds or updates a per-doctor **FAISS** store under `media/doctor_vectors/<slug>/`. Indexing runs in a **background thread** on upload (no Celery/Redis required); set `DM_USE_CELERY=1` to offload to Celery instead.
4.  **Retrieval**: `retrieve_context` uses **MMR** (relevance + diversity) and is dimension-safe — it returns `[]` (chat degrades gracefully) if an index is missing or was built with an older embedding model.

> After changing embedding models, rebuild stores with `python manage.py reindex_kb` (or the **«بازسازی ایندکس»** button in the KB studio).

#### **LLM Integration (`llm.py`):**
- `LLMClient` wraps the **GAPGPT** chat API (`gpt-5-nano`, a vision-language model). It accepts standard OpenAI messages, including **multimodal content** (`image_url` parts), and retries transient rate limits. Configured via `GAPGPT_*` env vars (falls back to `OPENAI_*` / `BASE_URL`).

---

### 3. Views & Handlers (`views.py`)
- **`api_send_message`**:
    - The core business handler for the chat interface.
    - **Flow**: Saves the user message + any **attachments** (`ChatAttachment`) → extracts text from attached **documents** and encodes attached **images** as base64 `image_url` parts → retrieves RAG context → calls the vision-capable `LLMClient` → saves and returns the assistant reply.
    - Accepts `multipart/form-data` with a `message` field and zero or more `attachments` files (images and/or documents).
- **`studio_kb`**:
    - Administrative view for doctors (or superusers) to upload and manage their medical knowledge files.
    - Triggers immediate re-indexing on file upload.

---

### 4. Routing (`urls.py`)
| Endpoint | View | Logic |
| :--- | :--- | :--- |
| `/` | `market_index` | Public listing of active doctors. |
| `/doctor/<slug>/` | `doctor_detail` | Profile view and chat initialization. |
| `/api/chat/<uuid>/send/` | `api_send_message` | RAG-augmented message processing. |
| `/studio/` | `studio_index` | Dashboard for doctor profile/KB management. |

---
*Note: This module handles its corresponding frontend views via template rendering in the `templates/doctors_marketplace/` directory.*
