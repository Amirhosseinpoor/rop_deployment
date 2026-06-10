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
1.  **Ingestion**: `index_file_for_doctor` reads PDFs (via `PdfReader`), chunks text using `RecursiveCharacterTextSplitter`, and generates embeddings using the local **BGE-Small** model.
2.  **Indexing**: Builds or updates a localized **FAISS** index stored in the doctor's media directory.
3.  **Retrieval**: `retrieve_context` performing similarity searches to find the top `k` relevant snippets from the doctor's private library.

#### **LLM Integration (`llm.py`):**
- A wrapper for the OpenAI API (`LLMClient`) that manages system prompt injection and temperature settings for medical consistency.

---

### 3. Views & Handlers (`views.py`)
- **`api_send_message`**:
    - The core business handler for the chat interface.
    - **Flow**: Saves User message → Retrieves RAG context from the doctor's FAISS index → Injects context as a "system" block into the message history → Calls `LLMClient` → Saves and returns the Assistant's reply.
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
