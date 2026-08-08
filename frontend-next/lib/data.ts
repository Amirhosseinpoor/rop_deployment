// Content model for the Mediverse AI landing — sourced from the platform catalogue.

export const STATS = [
  { n: 8, suffix: "", label: "Pretrained CV\nmodel checkpoints" },
  { n: 10, suffix: "", label: "Autonomous\nresearch agents" },
  { n: 5, suffix: "", label: "Zone / stage ROP\ngrading pipeline" },
  { n: 7, suffix: "", label: "In-context\nclinical assistants" },
];

export const PERSONAS = [
  {
    tag: "01",
    who: "Patients & Employees",
    lead: "Your health, read and explained.",
    body: "Instant AI screenings, a living health dossier, plain-language lab interpretation, and 24/7 consultations with specialist AI doctors — in Persian and English.",
    routes: [
      ["Consult an AI doctor", "/market/"],
      ["View your health dossier", "/health/profile/detail/"],
      ["Run an eye / anemia scan", "/health/profile/eye-scan/"],
      ["ROP fundus screening", "/rop/"],
    ],
  },
  {
    tag: "02",
    who: "Clinicians",
    lead: "Diagnose faster, with a second reader.",
    body: "AI diagnostic assistance, expert override portals, decision-support calculators, and an agentic research engine — inside your review workflow.",
    routes: [
      ["Doctor clinical dashboard", "/health/doctor-dashboard/"],
      ["Review & override ROP", "/rop/"],
      ["Deep health research", "/health/deep-research/start/"],
      ["Keratoconus staging", "/kc/"],
    ],
  },
  {
    tag: "03",
    who: "Managers",
    lead: "Keep your workforce fit and compliant.",
    body: "Oversee employee health compliance, workplace hazard assessments, work-fitness certifications, and onboard staff by national ID — scoped to your company.",
    routes: [
      ["Corporate management dashboard", "/managing/"],
      ["Register your company", "/signup/manager/"],
      ["Invite staff by national ID", "/signup/employee/"],
      ["Diagnostic audit history", "/history/"],
    ],
  },
  {
    tag: "04",
    who: "Administrators",
    lead: "Ship custom AI specialists.",
    body: "Design and deploy custom virtual AI doctors, upload proprietary knowledge bases, assign clinical tools, and manage organizational health pipelines.",
    routes: [
      ["Open the AI doctor studio", "/market/studio/"],
      ["System-prompt copilot", "/market/studio/copilot/"],
      ["Generate doctor avatars", "/market/studio/copilot/avatar/"],
      ["Export QC data", "/history/export/"],
    ],
  },
];

export const AGENTS = [
  ["A0", "Evidence Packet", "packet.py · deterministic", "Turns raw eye crops, lab panels, vitals and hazards into stable, cited facts — computing BP/BMI classes, eGFR, anemia severity and spirometry patterns in code, without math errors."],
  ["A1", "Eye Vision Analyst", "VLM · gpt-4o-mini", "Inspects palpebral & forniceal conjunctiva crops for pallor and quality, cross-checking the finding against laboratory hemoglobin."],
  ["A2", "Triage Agent", "reasoning model", "Ranks cross-modal problems by severity — high, medium, low — across ocular AI, lab flags, vitals and workplace hazard exposures."],
  ["A3", "Question Strategist", "reasoning model", "Formulates patient-anchored research questions across 7 archetypes and outputs biomedical PubMed queries."],
  ["A4", "Research Planner", "multi-channel APIs", "Plans retrieval across internal KB, web RAG, NCBI PubMed and Europe PMC — running multi-hop queries and extracting evidence tuples with source tiers."],
  ["A5", "Clinical Author", "synth · gpt-4o", "Authors doctor-grade dossier entries in a single Persian register, enforcing sentence-level [n] citations for every claim."],
  ["A6", "Adversarial Verifier", "3 skeptic lenses", "Evaluates each triaged problem against data-support, source-support and overstatement lenses via majority vote."],
  ["A7", "Completeness Critic", "synth model", "Audits the dossier for unaddressed modalities or missing thresholds, triggering a targeted second research round."],
  ["A8", "Synthesis & Fitness", "synth model", "Writes the executive brief, checks drug–disease interactions, and issues recommendations, red-flags, referrals and work-fitness verdicts."],
  ["A9", "Dossier Composer", "blocks.py · orchestrator", "Assembles reading-column blocks and compiles the Agentic Research Trail — problems, questions, sources by channel, skeptic votes."],
] as const;

export const MODELS = [
  ["EfficientNet-B4", "ROP Plus Disease", "model_efficentnet_b4_plus.pth"],
  ["EyeNet · CBAM · MHSA", "Keratoconus Staging", "best_model.pth"],
  ["Random Forest", "Hypertension Risk", "best_rf_hypertension_model.joblib"],
  ["U-Net++", "Conjunctiva Segmentation", "phase2_UNet_PlusPlus_fold0.pth"],
  ["Attention ResUNet", "Anatomical Features", "phase2_Attention_ResUNet_fold0.pth"],
  ["U²-Net", "Fine Vessel Tree", "phase2_U2_Net_fold0.pth"],
  ["ResNet18", "Palpebral Extraction", "best_resnet18_palpebral.pth"],
  ["EfficientNet-B0", "Pallor Evaluation", "best_efficientnet_b0_palpebral.pth"],
];

export const ASSISTANTS = [
  ["01", "ROP Diagnostic Assistant", "/rop/", "Context-aware ROP decision support with doctor/patient modes, SSE streaming, uploads and hybrid RAG."],
  ["02", "Keratoconus Assistant", "/kc/", "OD/OS staging awareness, CXL & lens-fitting guidance, file uploads and voice STT."],
  ["03", "Patient Health Consultation", "/health/profile/detail/", "Turns lab flags and audit data into plain-language Persian advice and PPE recommendations."],
  ["04", "Doctor Clinical Assistant", "/health/doctor-dashboard/", "Assists work-fitness evaluations, exposure limits and referral guidance."],
  ["05", "Deep Research Assistant", "/health/deep-research/", "Interrogates the multi-agent dossier with live PubMed / Europe PMC search and evidence-tier badges."],
  ["06", "Marketplace Chat Studio", "/market/chat/", "Multi-specialty doctors executing 9 clinical tools, private RAG, voice TTS and WhatsApp reminders."],
  ["07", "Global Agentic RAG API", ":8001/query_rag", "High-throughput endpoint with hybrid BM25 + FAISS + cross-encoder rerank and live web scraping."],
];

export const ARCH = [
  ["DJ", "Django Monolith", "Auth · RBAC · PostgreSQL system of record", ":8000"],
  ["FA", "FastAPI RAG service", "Hybrid BM25 + FAISS + cross-encoder rerank", ":8001"],
  ["CR", "Celery + Redis", "Neural inference & multi-agent synthesis", "workers"],
  ["WA", "WAHA gateway", "Scheduled WhatsApp medication reminders", "async"],
];
