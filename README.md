# Mediverse AI (Aras AI): Enterprise Ophthalmic Diagnostics & Agentic RAG Platform

**Mediverse AI** is a production-grade, hybrid medical technology platform specializing in automated diagnostics for **Retinopathy of Prematurity (ROP)** and **Keratoconus (KC)**. The system integrates high-performance Computer Vision (CV) pipelines with a stateful, agentic Retrieval-Augmented Generation (RAG) service to support clinical decision-making.

---

## 🏗 Core Architecture & Engineering Philosophy

The platform is engineered as a **distributed monorepo**, utilizing a dual-framework approach to optimize for both administrative complexity and high-performance asynchronous AI orchestration.

### 1. Django: The Orchestration & Governance Layer
- **Role**: Handles the "system of record" logic, including User Identity (RBAC), multi-tenant Company structures, and persistent medical records.
- **ORM & Data Persistence**: Manages complex relational schemas in PostgreSQL, ensuring audit trails for every AI diagnostic session.
- **Async Execution**: Leverages **Celery** with **Redis** to offload long-running deep learning inference tasks, preventing request blocking in clinical environments.

### 2. FastAPI: The Agentic RAG Service (`/rag_api`)
- **Role**: Operates as a decoupled, high-throughput microservice focused on stateful medical knowledge retrieval.
- **Retrieval Engine**: Implements a **Hybrid Retrieval** strategy (BM25 + Dense FAISS) coupled with a **Cross-Encoder Reranker** for high precision in medical literature grounding.
- **Agentic Capability**: Features a web-crawling agent (Serper + Selenium) to augment local knowledge with real-time clinical updates.

---

## 🗺 System Architecture Diagram

```ascii
      [ Clinical Frontend ]
               |
               ▼
    +-----------------------+          +--------------------------+
    |    Django Monolith    | <------> |   FastAPI RAG Service    |
    | (Auth, ORM, Admin)    |  [REST]  | (Agentic Pipeline / RAG) |
    +-----------+-----------+          +-------------+------------+
                |                                    |
        +-------+-------+                    +-------+-------+
        |               |                    |               |
  [ PostgreSQL ]    [ Redis ]          [ Local PDFs ]  [ Web Search ]
  (Medical Logs)    (Broker)           (Vector Store)  (Serper API)
        |               |
        +-------+-------+
                |
       +--------+--------+
       | Celery Workers  | <--- [ AI Models ]
       | (CV Inference)  |      (ROP: Unet++/EffNet | KC: EyeNet)
       +-----------------+
```

---

## 🛠 Technical Deep Dive: AI Domains

### Retinopathy of Prematurity (ROP) Diagnostic Pipeline
- **Modules**: `single_rop`, `double_rop`.
- **Diagnostics**: Multi-stage classification of retinal fundus images.
- **Workflow**: Segmentation of vascular tree → Plus disease classification → Disease Staging (0-5) → Zonal identification (1-3) → Automated clinical guidance generation.

### Keratoconus (KC) Eye Disease Diagnostics
- **Modules**: `double_rop`.
- **Diagnostics**: Pairwise analysis of left/right eye corneal topography images.
- **Workflow**: Feature extraction via Siamese-style architecture (EyeNet) → Symmetry evaluation → Severity classification (ATN, EIr, etc.).

---

## 📂 Application Directory Map

| Application Name | Path | Core Logic & Domain |
| :--- | :--- | :--- |
| **Config** | `/config` | Global orchestration, Auth gateways, and middleware. |
| **Single ROP** | `/single_rop` | 4-stage CV pipeline for single retinal image analysis. |
| **Double ROP** | `/double_rop` | Pairwise KC diagnostics and multi-image ROP analysis. |
| **Test Analysis** | `/test_analysis` | Agentic health profile auditing & clinical report generation. |
| **Marketplace** | `/doctors_marketplace` | Clinician-specific Knowledge RAG & Chat Studio. |
| **RAG API** | `/rag_api` | Global FastAPI service for Hybrid & Web RAG. |
| **USAC** | `/usac` | Identity layer, Role-based access, and Company management. |

---

## 🚀 Global Deployment & Setup

### Environment Requirements
- **Hardware**: GPU with CUDA support recommended (8GB+ VRAM) for local inference.
- **Models**: Pre-trained weights for EyeNet, Unet++, and EfficientNet must be placed in `model/`.
- **APIs**: Requires `OPENAI_API_KEY`, `SERPER_API_KEY`, and `METIS_API_KEY`.

### Quick Start
1.  **Initialize Environment**: `pip install -r requirements.txt`
2.  **Apply Migrations**: `python manage.py migrate`
3.  **Start Django**: `python manage.py runserver`
4.  **Start RAG API**: `python -m rag_api.app.main`
5.  **Start Workers**: `celery -A config worker --loglevel=info`

---
*Note: This project strictly separates backend logic from frontend presentation. Local `templates/` and `static/` directories handle UI rendering via Django's template engine.*
