"""Test-Analysis (health) microservice.

Exposes the three AI capabilities of the original Django ``test_analysis`` app:

1. **Hypertension risk prediction** — a scikit-learn model scores patient vitals.
2. **Health chat assistant** — a tool-calling LLM agent that finds doctors and
   medications/pharmacies for a patient (web scrapers as tools).
3. **Health analysis report** — a multi-stage RAG pipeline that predicts disease,
   finds doctors and drugs, and generates a full Persian medical report.

The Django web plumbing (health-profile CRUD forms, doctor/manager dashboards,
role checks) is intentionally excluded: it overlaps with the USAC service and is
presentation logic, not an AI capability. See ``HOW_TO_RUN.md``.

Module layout:
* :mod:`disease_models` — the hypertension predictor.
* :mod:`scrapers` / :mod:`handle_tools` / :mod:`prompts` — pipeline tool support.
* :mod:`finders`  — scrapers + tool-calling chat agent.
* :mod:`pipeline` — the report-generation RAG pipeline.
* :mod:`service`  — thin orchestration used by the routes.
"""
