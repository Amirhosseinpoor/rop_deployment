# Test Analysis: Agentic Health Auditing & RAG

The `test_analysis` application is the platform's most advanced analytical engine, implementing an **Agentic Health Audit** pipeline to evaluate patient profiles and generate professional medical reports.

---

### 1. Models (`models.py`)
The data model is centered around the **`HealthProfile`**, a comprehensive digital representation of a patient's medical and occupational history.

- **`HealthProfile` Schema**:
    - **Personal & Occupational**: 50+ fields covering demographics, hazards (Physical, Chemical, Ergonomic), and smoking history.
    - **Clinical Data**: Lab results (Cholesterol, Glucose, CBC), Vital signs (BP, Pulse), and Spirometry metrics.
    - **AI Outputs**: `llm_advice` (persisted Markdown report) and `report_task_id` (Celery tracking).
- **Relational Data**:
    - **`PreviousJob` (FK)**: Tracking occupational exposure over time.
    - **`Referral` (FK)**: Storing specialist recommendations.

---

### 2. Core Logic & AI Pipelines (`ai_pipeline.py`)
The module orchestrates a 4-stage **Agentic Graph** to transform raw health data into clinical insights.

#### **Pipeline Trace (`run_health_analysis_pipeline`):**
1.  **Stage 1: Disease Prediction (`predict_hypertension_risk`)**:
    - **Model**: Random Forest classifier (`best_rf_hypertension_model.joblib`).
    - **Logic**: Evaluates 12 biometric features (Age, BP, Glucose, etc.) to predict hypertension probability.
2.  **Stage 2: Specialist Matching (`scrape_doctors`)**:
    - **Tool**: Web scraper for `doctoreto.com`.
    - **Logic**: Uses the LLM to identify the required specialty (e.g., "Cardiologist") and scrapes local providers based on the patient's city/region.
3.  **Stage 3: Pharmaceutical RAG (`chat_with_drugs_llm`)**:
    - **Logic**: Uses a specialized agent to search a local CSV vector store (`drugstores.csv`) and the `mokamelkhoone.com` website to find relevant supplements (e.g., Magnesium for hypertension).
4.  **Stage 4: Synthesis & Grounding (`chat_with_rag_llm`)**:
    - **Logic**: A `ConversationalRetrievalChain` pulls context from the authoritative PDF knowledge base (`knowledge_base/`) and synthesizes all previous stages into a cohesive Persian/English medical report.

---

### 3. Views & Handlers (`views.py`)
- **`create_or_update_health_profile`**:
    - Collects 100+ form inputs.
    - Triggers the **`generate_health_report`** Celery task asynchronously.
    - Implements an "Edit Lock" rule: Employees cannot edit their profiles once a doctor has appended clinical notes.
- **`profile_detail_view`**:
    - Implements strict **Role-Based Access Control (RBAC)**.
    - Doctors and Managers can view and append `DoctorNotesForm` and `ReferralFormSet` to an employee's profile.
- **`health_chat_api`**:
    - A stateless endpoint for real-time conversation with the Health Assistant agent.

---

### 4. Routing (`urls.py`)
| Endpoint | View | Logic |
| :--- | :--- | :--- |
| `/profile/` | `create_or_update_health_profile` | Profile submission and report triggering. |
| `/profile/detail/<int:user_id>/` | `profile_detail_view` | Medical report and clinician notes. |
| `/chat/health-chat/` | `health_chat_api` | Agentic chat interface. |
| `/processing/status/` | `report_status` | Polling endpoint for async report readiness. |

---
*Note: This module handles its corresponding frontend views via template rendering in the `templates/test_analysis/` directory.*
