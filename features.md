# Mediverse AI (Aras AI) — Platform Services & Clinical Capabilities Catalogue

---

## 🏛 EXECUTIVE SUMMARY

**Mediverse AI** is an enterprise medical technology platform providing automated ophthalmic diagnostics, occupational health auditing, agentic clinical research, and AI-driven virtual medical consultations.

The platform delivers services across four primary user personas:
* **Patients & Employees**: Accessing diagnostic screenings, health dossiers, lab interpretations, and AI doctor consultations.
* **Clinicians & Examining Doctors**: Utilizing AI diagnostic assistance, expert review/override portals, decision support tools, and clinical calculators.
* **Corporate Managers**: Overseeing employee health compliance, workplace hazard assessments, work-fitness certifications, and staff invitations.
* **Healthcare Administrators**: Building custom virtual AI medical specialists, uploading proprietary knowledge bases, and managing organizational health pipelines.

---

## 1. OPHTHALMIC AI DIAGNOSTIC SERVICES

### A. Retinopathy of Prematurity (ROP) Screening & Diagnostics
* **Automated Infant Retinal Fundus Screening**: AI-powered analysis of retinal fundus photographs from premature infants to detect ROP.
* **4-Stage Pathology Detection**:
  * **Vessel & Disk Segmentation**: Automated extraction of retinal vascular networks, optic disk, and macula using deep segmentation models (U-Net, U-Net++, Attention ResUNet, U2-Net).
  * **Plus Disease Classification**: Classification of retinal vessel tortuosity and dilation into *Normal*, *Pre-Plus*, or *Plus Disease* using EfficientNet-B4.
  * **Disease Severity Staging**: Precise grading across *Stage 0* through *Stage 5*.
  * **Zonal Mapping**: Automated localization of disease activity relative to the optic nerve (*Zone I*, *Zone II*, *Zone III*).
* **Multi-Image Batch Diagnostics**: Simultaneous upload and cross-analysis of multiple fundus photos per infant, synthesizing worst-case risk indicators into a single diagnosis.
* **ICROP 3rd Edition Clinical Decision Support**: Automated recommendations based on international clinical guidelines (e.g., immediate laser photocoagulation / anti-VEGF intervention vs. 48-hour follow-up).
* **Doctor Review & Ground-Truth Verification**: Dedicated portal for eye specialists to inspect fundus images, override machine labels, and record official clinical notes.

---

### B. Keratoconus (KC) & Corneal Topography Diagnostics
* **Dual-Eye Corneal Topography Analysis**: Pairwise AI analysis of Left Eye (OD) and Right Eye (OS) corneal topography maps using the EyeNet Siamese deep learning architecture.
* **5-Tier Severity Staging**: Grading of corneal ectasia into *Normal*, *ATN* (Atypical Topography Normal), *NEIr* (Non-Ectatic Irregularity), *EIr* (Ectatic Irregularity), or *eKCN* (Early Keratoconus).
* **Bilateral Structural Symmetry Scoring**: Pairwise evaluation of corneal structural symmetry (*SfRS* — Symmetrical / Favorable Refractive Status vs. *NSfRS* — Non-Symmetrical / Unfavorable Refractive Status).
* **Ocular-Linked Cardiovascular Risk Estimation**: AI estimation of systemic hypertension and cardiorespiratory risk derived from ocular topography and vascular features via pre-trained Random Forest models.

---

## 2. OCCUPATIONAL HEALTH & DEEP HEALTH RESEARCH SERVICES

### A. Comprehensive Employee Profile Detail Page Capabilities (`/health/profile/detail/`)
The Employee Profile Detail page is an all-in-one clinical command center for employee health management:

1. **Hero Profile & Demographic Command**:
   * Displays full personal identity: Name, National ID, DOB/Age, Gender, Marital Status, Children count, Military service & exemption reasons.
   * Contact & Placement: Province, Neighborhood, Insurance type, Work address, Work phone.
   * Dynamic hazard chips highlighting active occupational risk exposures.
2. **Non-Invasive Conjunctiva Anemia Screening (`.eye-unit`)**:
   * Interactive 3-stage visual frame displaying raw photo, Phase-1 conjunctiva crop, Phase-2 palpebral segmentation mask, and VLM pallor verdict (*positive* / *negative* with confidence percentage).
   * Modal form allowing employees to upload new smartphone eye photos for instant re-screening.
3. **Extracted Medical Test & Laboratory Analytics (`MedicalTest`)**:
   * Rendered tabular view of extracted lab panels (CBC, Metabolic Panel, Lipid Panel, Urinalysis, Addiction Screen).
   * Displays analyte names, measured values, units, reference ranges, specimen details, collection dates, and abnormal out-of-range flag badges.
   * File viewer for original uploaded lab report documents.
4. **Physical Examination & Vitals Matrix**:
   * Measured vitals: Weight (kg), Height (cm), BMI, Blood Pressure (sys/dia mmHg), Pulse rate (bpm).
   * 12-System Organ Exam Notes: General, Eye, Skin/Hair/Nails, ENT/Mouth, Head/Neck, Lung/Respiratory, Cardiovascular, Abdomen/Pelvis, Urinary, Musculoskeletal, Nervous System, Mental Health.
5. **Occupational History & 5-Domain Hazard Breakdown**:
   * Timeline of Previous Employment (*PreviousJob*): Title, duties, start/end dates, reason for leaving.
   * Detailed breakdown across Physical, Chemical, Biological, Ergonomic, and Psychological hazard categories.
   * Health history: Illness history, symptom variations at work/holidays, colleague symptoms, allergies, hospitalizations, surgeries, family disease, smoking pack-years, cigs/day, absence >3 days, medical commission referrals.
6. **Paraclinical & Spirometry Analytics**:
   * Pulmonary function test values: FVC, FEV1, FEV1/FVC ratio, FEF 25-75%, PEF, and clinical pattern interpretation (*Normal*, *Obstructive*, *Restrictive*, *Mixed*).
   * Radiography, ECG, and ultrasound clinical findings.
7. **Work Fitness Certification & Referral Center**:
   * High-impact Work Fitness Banners: *Fit (بلامانع)* in green, *Fit with Conditions (مشروط)* in amber with required condition details, *Unfit (عدم صلاحیت)* in rose red with explicit medical disqualification reasons.
   * Medical Recommendations callout box.
   * Referral Cards: Tracking required specialty, urgency date, referral reason, and specialist evaluation results.
8. **Role-Restricted Doctor & Manager Editing Suite**:
   * Interactive *Doctor Notes Form* for examining clinicians/managers to edit organ exam notes, vitals, medical recommendations, and fit/unfit opinions.
   * Dynamic *Referral FormSet* to add, update, or remove specialist referrals.
9. **Interactive Co-Pilot Sidebar & Retro Minigame**:
   * Embedded assistant sidebar featuring pure CSS 3D Minecraft Steve animation (`.steve-scene`) connected to patient/doctor consultation chatbots.
   * One-click link to `/health/play/` launching retro Pacman minigame while long-running Deep Research reports compile in the background.

---

### B. Deep Health Research v3 Clinical Dossier Engine (The 10 Autonomous Agents)
The Deep Health Research v3 engine replaces simple health scores with an autonomous 10-agent clinical research dossier. Below is the explicit breakdown of all 10 agents:

| Agent ID | Agent Name | Core Role & Functionality | Underlying Model / Tool |
| :--- | :--- | :--- | :--- |
| **A0** | **Evidence Packet Agent** | Deterministically processes raw patient data (eye AI crops, extracted lab panels, vitals, hazards, occupational history) into structured, stable cited facts (`S`-ids). Calculates BP/BMI/glucose classes, eGFR, anemia severity, pack-years, and spirometry patterns in code without math errors. | Python Deterministic Code (`packet.py`) |
| **A1** | **Eye Vision Analyst** | Inspects palpebral & forniceal conjunctiva crops for pallor, quality, conjunctival redness, and cross-checks VLM findings against laboratory hemoglobin levels. | Vision-Language Model (`DR_VISION_MODEL` / `gpt-4o-mini`) |
| **A2** | **Triage Agent** | Ranks cross-modal patient problems by severity (*high*, *medium*, *low*), synthesizing findings from ocular AI, lab analyte flags, vitals, and workplace hazard exposures. | Reasoning Model (`DR_REASONING_MODEL`) |
| **A3** | **Question Strategist** | Formulates patient-anchored research questions across 7 archetypes (*interpretation*, *occupational*, *corroboration*, *workup*, *risk*, *threshold*, mandatory *lifestyle*), outputting biomedical PubMed queries (`lit_query`). | Reasoning Model (`DR_REASONING_MODEL`) |
| **A4** | **Research Planner & Multi-Hop Agent** | Plans retrieval channels (*Internal KB*, *Web RAG*, *NCBI PubMed E-Utilities*, *Europe PMC REST API*). Executes multi-hop queries and extracts evidence tuples (`claim`, `source_n`, `quote`, `tier`). | Multi-Channel APIs + Reasoning Model |
| **A5** | **Clinical Author** | Authors deep clinical dossier entries in single doctor-grade Persian register. Enforces sentence-level evidence citations `[n]` for every factual claim. | Synthesis Model (`DR_SYNTH_MODEL` / `gpt-4o`) |
| **A6** | **Adversarial Verifier** | Evaluates triaged problems against 3 skeptic lenses (Patient Data Support, Cited Source Support, Contradiction/Overstatement Check) via majority vote. | 3 Skeptic Lenses (`DR_REASONING_MODEL`) |
| **A7** | **Completeness Critic** | Audits the generated dossier for unaddressed lab modalities, missing threshold numbers, or gaps, triggering a targeted 2nd round of A4 research & A5 authoring. | Synthesis Model (`DR_SYNTH_MODEL`) |
| **A8** | **Synthesis & Work-Fitness Agent** | Authors executive brief (no numerical score), evaluates drug-disease interactions, formulates categorized recommendations (clinical & mandatory lifestyle), red-flag callouts, referrals, and work-fitness verdicts. | Synthesis Model (`DR_SYNTH_MODEL`) |
| **A9** | **Dossier Composer** | Assembles clean reading-column editorial blocks and compiles the signature **Agentic Research Trail** detailing problems triaged, questions asked, source counts by channel, and skeptic votes. | Python Orchestrator (`blocks.py`) |

---

## 3. VIRTUAL DOCTOR MARKETPLACE & AI SPECIALIST CONSULTATION

### A. AI Medical Specialist Directory
* **Specialized Virtual Clinicians**: Access to specialized AI virtual doctors across key medical fields (Nephrology/Urology, Cardiology, Endocrinology, Hematology, Hypertension, Family Counseling).
* **Bilingual Persona Styling**: Virtual doctors equipped with specific communication personas (*Kind & Patient*, *Efficient & To-the-Point*, *Calm & Reassuring*, *Analytical*) operating in Persian and English.

---

### B. Interactive Consultation Chat Studio
* **24/7 AI Medical Consultation**: Real-time streaming conversations with specialized virtual doctors.
* **Multi-Modal Consultation**: Patients can upload lab reports, medical documents (PDF/DOCX/TXT), or clinical images directly into their chat session for instant AI analysis.
* **Voice-Enabled Consultation**: Voice message input via Whisper speech-to-text with auto Persian/English detection, coupled with Text-to-Speech (TTS) audio playback of doctor advice.
* **Rich Chat Management**: Session pinning, renaming, message editing, quote-replying, thumbs up/down response rating, and full-text session search.

---

### C. Automated WhatsApp Patient Care & Reminders
* **WhatsApp Medication Reminders**: Automated delivery of medication schedules and dosage alerts directly to patient WhatsApp numbers.
* **Clinical Follow-Up Scheduling**: Automated follow-up messages sent to patients at designated intervals after diagnostic assessments.

---

## 4. CLINICIAN STUDIO & AI DOCTOR CREATION (FOR HEALTHCARE PROVIDERS)

### A. Custom Virtual Doctor Studio
* **AI Specialist Construction**: Tools for healthcare organizations and administrators to design and deploy custom virtual AI doctors.
* **AI Copilot System Prompt Builder**: Automated prompt engineering assistant that constructs doctor personas and clinical behavioral guidelines.
* **Generative Doctor Avatar Creation**: Generative AI creation of photo-realistic medical doctor profile avatars customized by specialty and persona.

---

### B. Custom Knowledge Base (RAG) Builder
* **Proprietary Document Indexing**: Uploading custom clinical guidelines, textbook chapters, or institutional protocols (PDF/TXT) to create doctor-specific vector knowledge bases.
* **Automated Re-Indexing**: On-demand vector store re-indexing for seamless updating of clinical reference materials.

---

### C. Specialized Clinical Tool Assignment
Administrators can equip custom virtual doctors with active clinical tools:
* **Medical Calculators**: BMI, BSA, eGFR, CrCl, MELD Score, CHA₂DS₂-VASc Score, Anion Gap, Pediatric Dosage Calculator, IV Flow Rate Calculator.
* **Diagnostic & Pharmacology Agents**: Lab Interpreter, Emergency Red-Flag Scanner, Drug Monograph Lookup, Multi-Drug Interaction Checker, Medical Literature Search (PubMed).

---

## 5. CORPORATE MANAGEMENT & GOVERNANCE SERVICES

### A. Multi-Tenant Corporate Management
* **Company Registration & Administration**: Registration of corporate entities, managing departments, and tracking employee counts.
* **National ID Invitation System**: Secure onboarding of staff and employees linked by National Identification numbers.
* **Role-Tailored Dashboards**: Customized web portals tailored specifically for Corporate Managers, Examining Clinicians, and Personnel.

---

### B. Audit & Quality Control
* **Centralized Diagnostic Audit Trail**: Full historical logging of all ROP screenings, KC topographies, health profiles, and AI consultations across the organization.
* **Misclassification Data Export**: Admin tools to export misclassified diagnostic cases to CSV for quality control and AI continuous model refinement.

---

## 6. DETAILED CATALOGUE OF AI CHATBOTS ACROSS ALL PAGES

Mediverse AI integrates specialized AI chatbots across every application page. Each chatbot is customized to its specific page domain, user role, and clinical task:

* **Chatbot 1: ROP Diagnostic Assistant (`/rop/`)**: Context-aware ROP decision support, dual doctor/patient response modes, SSE streaming, document uploads (PDF/DOCX/TXT/MD), Whisper voice STT, and hybrid RAG.
* **Chatbot 2: Keratoconus (KC) Assistant (`/kc/`)**: Corneal topography context awareness (EyeNet OD/OS staging & Z-class symmetry), CXL & lens fitting guidance, file uploads, and voice STT.
* **Chatbot 3: Patient Health Consultation Assistant (`/health/profile/detail/`)**: Translates lab flags and health audit data into plain-language Persian advice and workplace hazard PPE recommendations.
* **Chatbot 4: Doctor Clinical Assistance Assistant (`/health/doctor-dashboard/`)**: Assists examining clinicians with work-fitness evaluations (*Fit*, *Fit with Conditions*, *Unfit*), workplace exposure limits, and referral guidance.
* **Chatbot 5: Doctor Deep Research Assistant (`/health/deep-research/`)**: Interrogates the multi-agent research dossier, performing live PubMed/Europe PMC literature searches and displaying evidence-tier badges.
* **Chatbot 6: AI Doctor Marketplace Chat Studio (`/market/chat/`)**: Multi-specialty virtual doctors with custom personas executing 9 clinical tools (calculators, lab interpreters, drug interaction checkers, PubMed search), private RAG, voice TTS, and WhatsApp reminder scheduling.
* **Chatbot 7: FastAPI Global Agentic RAG Microservice (`/rag_api/`)**: High-throughput RAG API endpoint featuring hybrid BM25 + FAISS + Cross-Encoder reranking and live web scraping.

---

## 7. SYSTEM INTEGRATION & DISTRIBUTED DATA FLOW

```ascii
                      [ Web & Mobile Frontends ]
                                  |
                                  ▼
      +-------------------------------------------------------+
      |                Django Monolith (Port 8000)            |
      |   Auth (RBAC)  ·  PostgreSQL System of Record  ·  Admin  |
      +------------+------------------------------+------------+
                   |                              |
                   ▼                              ▼
      +------------------------+      +-----------------------+
      | FastAPI RAG Service    |      | Redis Broker & Celery |
      | (Port 8001)            |      | Task Queue Workers    |
      +-----------+------------+      +-----------+-----------+
                  |                               |
       +----------+----------+         +----------+----------+
       |                     |         |                     |
[ Hybrid Local KB ]    [ Web Search ] [ Ophthalmic CV ]  [ Deep Research ]
 (FAISS + BM25)         (Serper)     (U-Net++ / EyeNet)   (PubMed / EPMC)
```

1. **Django Core**: Handles user session authentication, corporate tenant isolation, database persistence, and synchronous HTTP template rendering.
2. **FastAPI Microservice**: Decoupled asynchronous RAG backend servicing document vector search, cross-encoder reranking, and dynamic web scraping.
3. **Celery & Redis Worker Infrastructure**: Offloads heavy, long-running neural network inference and multi-agent research dossier synthesis to background workers.
4. **WAHA Gateway**: Interconnects scheduled background tasks with WhatsApp API endpoints for automated patient reminder delivery.

---

## 8. PRE-TRAINED DEEP LEARNING MODEL CHECKPOINTS

The platform relies on proprietary pre-trained model weights located in the `model/` directory:

| Model Checkpoint File | Target Pathology / Task | Model Architecture |
| :--- | :--- | :--- |
| `model_efficentnet_b4_plus.pth` | ROP Plus Disease Classification | EfficientNet-B4 |
| `best_model.pth` | Keratoconus Topography Staging | EyeNet (ResNet50/CBAM/MHSA) |
| `best_rf_hypertension_model.joblib` | Ocular-Linked Hypertension Risk | Random Forest Ensemble |
| `phase2_UNet_PlusPlus_fold0.pth` | Conjunctiva Tissue Segmentation | U-Net++ |
| `phase2_Attention_ResUNet_fold0.pth` | Anatomical Feature Segmentation | Attention ResUNet |
| `phase2_U2_Net_fold0.pth` | Fine Vessel Tree Segmentation | U2-Net |
| `best_resnet18_palpebral.pth` | Palpebral Tissue Extraction | ResNet18 |
| `best_efficientnet_b0_palpebral.pth` | Conjunctiva Pallor Evaluation | EfficientNet-B0 |

---

## 9. DATA SECURITY, PRIVACY & PII COMPLIANCE

1. **Automated PII Redaction**: Document extraction engines (`MedicalTest`) process lab reports using strict instruction boundaries that extract only numerical analytes, units, and flags while discarding patient names, national IDs, DOBs, and ordering physician details.
2. **Multi-Tenant Corporate Isolation**: Database queries enforce strict company scoping (`company_id`). Corporate managers and doctors are strictly constrained to reviewing personnel belonging to their own registered corporate entity.
3. **View-Locking Enforcement**: Non-clinician employees are locked to their own diagnostic records, preventing unauthorized access to peer records.
4. **Audit Trail Logging**: Every diagnostic prediction, clinician label override, and case note is permanently timestamped and linked to the acting user ID in PostgreSQL.

---

## 10. COMPREHENSIVE ENDPOINT & SERVICE URL REFERENCE MAP

| URL Path Pattern | Target View / Service Function | Authorized Persona |
| :--- | :--- | :--- |
| `/` | Landing Page & Portal Entry | Public |
| `/login/` | Role-Aware Authentication Gateway | Public |
| `/signup/choose-role/` | Role Selection Interface | Public |
| `/signup/manager/` | Company Registration & Manager Signup | Public |
| `/signup/doctor/` | Clinician Signup with National Code Verification | Invited Doctors |
| `/signup/employee/` | Employee Signup with National Code Verification | Invited Employees |
| `/managing/` | Corporate Management Dashboard | Managers |
| `/dilemma/` | Diagnostic Dilemma & Ethical Case Module | All Users |
| `/history/` | Centralized Diagnostic Audit History | Authenticated Users |
| `/history/export/` | Export History to CSV | Authenticated Users |
| `/export/rop/` | Export Misclassified ROP Diagnoses to CSV | Admins / Doctors |
| `/export/kc/` | Export Misclassified KC Diagnoses to CSV | Admins / Doctors |
| `/rop/` | ROP Fundus Single & Batch AI Screening | All Users / Reviewers |
| `/rop/predict/` | Anonymous ROP Multi-Image Screening API | All Users / API Clients |
| `/rop/feedback/` | Expert ROP Prediction Override Submission | Doctors / Managers |
| `/rop/notes/` | Save ROP Doctor Case Notes | Doctors / Managers |
| `/kc/` | Keratoconus Topography AI Screening | All Users / Reviewers |
| `/kc/predict/` | KC Topography Prediction API | All Users / API Clients |
| `/kc/notes/` | Save KC Doctor Case Notes | Doctors / Managers |
| `/health/profile/` | Create or Edit Employee Health Profile Form | Employees / Managers |
| `/health/profile/detail/` | Employee Profile Command Center | Profile Owner / Clinicians |
| `/health/profile/eye-scan/` | Conjunctiva Anemia Photo Upload | Profile Owner |
| `/health/doctor-dashboard/` | Clinical Audit & Health Risk Dashboard | Doctors / Managers |
| `/health/deep-research/start/` | Trigger Deep Health Research v3 Loop | Authenticated Users |
| `/health/play/` | Retro Pacman Patient Minigame | All Patients / Employees |
| `/market/` | AI Doctor Directory & Marketplace Index | Public |
| `/market/doctor/<slug>/` | Bilingual Doctor Profile & Persona Detail | Public |
| `/market/chat/<session_id>/` | Interactive Virtual Specialist Chat Studio | Authenticated Users |
| `/market/studio/` | AI Doctor Builder & Management Studio | Admins / Superusers |
| `/market/studio/copilot/` | System Prompt Engineering Copilot | Admins / Superusers |
| `/market/studio/copilot/avatar/` | Generative DALL-E Doctor Avatar Builder | Admins / Superusers |
| `:8001/query_rag` | Global FastAPI RAG Query Endpoint | Microservice Clients |
| `:8001/health` | RAG API Health Monitoring Endpoint | Monitoring Services |
