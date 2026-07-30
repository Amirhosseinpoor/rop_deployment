# Deep Health Research v3 — Clinical Dossier Engine

A ground-up rebuild of the **تحلیل و توصیه‌های هوش مصنوعی** engine on
`/health/profile/detail/`. v3 drops the "dashboard of widgets" look and the
dual-register text and instead produces a **rigorous clinical research dossier**,
fully in Persian, that reads like the deep-research output of a serious lab
(OpenAI / Claude / Perplexity deep-research) rather than a health app.

The report ingests everything about one employee — the eye-AI crops, every
extracted medical-test analyte, and the full occupational-health form — and
reasons over it with a **multi-channel, multi-hop, multi-agent research loop**
whose every clinical claim is graded and cited to a real source.

> Runs under plain `python manage.py runserver` (daemon thread + bounded
> `ThreadPoolExecutor`). No new migration — the JSON dossier is stored in
> `HealthProfile.llm_advice` behind the `#DRV1#` sentinel.

---

## 0. What changed from v2 (the directives that drive v3)

| Directive | v3 |
| --- | --- |
| ❌ Remove **زبان ساده** (plain register) | Single **clinical register** only — the v2 clinical prompt was good, so it becomes *the* voice: precise, Persian, doctor-grade but readable. |
| ❌ Remove **نمرهٔ سلامت** (health score) | No score, no gauge. The dossier opens with an **editorial executive brief**, not a number. |
| "quality still too low" | **Three retrieval channels** (internal KB + web + **biomedical literature APIs**), a **real per-problem multi-hop research agent** (plan → retrieve → read → follow-up → extract), **evidence grading**, and a **completeness critic** that fills gaps in a second round. |
| "completely in Persian" | Every user-visible string is Persian. English is used only *internally* for search queries; it never appears in the output. |
| "use different APIs / agents" | Adds **NCBI PubMed E-utilities** + **Europe PMC** literature APIs, a **model tier** (stronger model for authoring/synthesis), a **Research Planner** agent, and a **Completeness Critic** agent. |
| "world-class, not a kid's dashboard" | New **editorial dossier** front-end language (see §11): reading column, typographic hierarchy, a signature **agentic research trail**, a references apparatus with evidence-tier badges. Zero reuse of the old `.dr-*` component styles. |

---

## 1. Principles

1. **Compute in code, reason in the LLM.** All arithmetic (BP/BMI/glucose class,
   anemia severity, pack-years, spirometry pattern, abnormal-flag scan, eGFR when
   Cr present) stays deterministic in `packet.py`. The model interprets, links,
   grades — never calculates.
2. **No claim without a graded citation.** Every clinical sentence carries an
   `[n]` pointing at a real source, and every source carries an **evidence tier**
   (`رهنمود` guideline · `مطالعه` study/review · `دانش‌نامه` internal KB ·
   `دادهٔ بیمار` patient datum). A claim's strength = the best tier that supports it.
3. **Cross-modal by construction.** A finding present in the eye AI *and* a lab
   flag *and* an occupational exposure is one stronger, better-cited problem.
4. **Adversarially verified.** Each finding survives a 3-lens skeptic panel; a
   completeness critic then hunts for what's missing and triggers one more round.
5. **Persian, clinical, un-intimidating.** Doctor-grade terminology, defined in
   place when needed. RTL. No score, no gimmicks.
6. **Un-clonable.** Private multimodal data (eye crops) + this person's exact
   numbers + occupational exposure + *fresh cited biomedical literature* is a
   combination a generic chatbot cannot reproduce.

---

## 2. Data sources

Unchanged from v2 (all of them): `EyeAnalysis` (label, confidence, **crops**),
every `MedicalTest.panels` analyte, all vitals + inline labs, occupational
`hazard_*`, history, `PreviousJob` timeline, clinician notes, demographics.
Built into the deterministic **Evidence Packet** (`packet.py`) with stable
`S`-ids so every downstream claim can cite a specific patient datum.

---

## 3. Tools / APIs (the agents' hands)

| Tool | Source | Role |
| --- | --- | --- |
| **json_llm** | `tools/llm_json.py` (raw GAPGPT client, JSON mode) | Structured reasoning calls. |
| **vision_read** | `tools/vision.py` (`DR_VISION_MODEL`) | VLM reads the conjunctiva crops (A1). |
| **web_search / read_and_rank** | `tools/web.py` → Serper + scrape + FAISS | Current guideline pages. |
| **kb_retrieve** | `tools/web.py` → internal RAG KB | Curated offline clinical/occupational knowledge. |
| **lit_search** ⭐ NEW | `tools/literature.py` → **NCBI PubMed E-utilities** (`esearch`+`esummary`) **+ Europe PMC REST** | Real biomedical citations (guidelines, reviews, primary studies). Free, no key. Best-effort + short timeouts (geo-degrades gracefully to KB+web). |
| **embed** | `rag.get_embeddings` (`text-embedding-3-large`, 3072-d) | Dedup/rank questions, sources. |

**Model tier** (all env-configurable, GAPGPT):
- `DR_SYNTH_MODEL` — authoring + synthesis + critic (strongest; default `gpt-4o-mini`, set to `gpt-4o` if the account has it).
- `DR_REASONING_MODEL` — triage, questions, planning, verify (default `gpt-4o-mini`).
- `DR_VISION_MODEL` — eye vision (default `gpt-4o-mini`).

Knobs: `DR_QUESTIONS_PER_PROBLEM=4`, `DR_MAX_PROBLEMS=6`, `DR_LIT_PER_PROBLEM=3`,
`DR_WEB_PAGES=8`, `DR_VERIFY_VOTES=3`, `DR_CRITIC_ROUNDS=1`, `DR_FANOUT_WORKERS=4`.

---

## 4. Agents

| # | Agent | Model | Job |
| --- | --- | --- | --- |
| **A0** | Evidence Packet (deterministic) | — | Facts + gauges + exposure timeline + `S`-ids. |
| **A1** | Eye Vision Analyst | vision | Reads the actual conjunctiva crop → pallor/quality/agreement. |
| **A2** | Triage | reasoning | Ranked cross-modal problems `{title, category, severity, evidence_refs, mechanism_hint}`. |
| **A3** | Question Strategist | reasoning | Per problem → patient-anchored questions across 7 archetypes incl. **lifestyle**. |
| **A4** | **Research Planner + Agent** (per problem, parallel) | reasoning | Plans channels per question, runs **multi-hop** KB + web + **literature**, extracts graded `evidence[]`. |
| **A5** | **Clinical Author** (per problem, parallel) | synth | Deep dossier entry, single clinical register, every sentence `[n]`, plus a differential + occupational linkage. |
| **A6** | Adversarial Verifier (3 lenses) | reasoning | Keep/drop/downgrade per problem; sets confidence. |
| **A7** | **Completeness Critic** ⭐ | synth | "What's missing — an unread modality, an unverified claim, a gap?" → triggers one targeted research+author round. |
| **A8** | Synthesis & Work-Fitness | synth | Executive brief (no score), interactions, recommendations (**incl. lifestyle**), red-flags, referrals, work-fitness. |
| **A9** | Dossier Composer | code | Assembles editorial blocks + the **research trail** (agentic process metadata). |

---

## 5. Execution order

```
0 COLLECT     A0 packet (det.)  ‖  A1 eye-vision (VLM)      [waits for eye/lab pipelines]
1 TRIAGE      A2 → ranked cross-modal problems
2 QUESTIONS   A3 → patient-anchored research questions (+lifestyle), deduped
3 RESEARCH    A4 per problem (parallel): plan → KB + web + PubMed/EuropePMC
                 → read → one follow-up hop → graded evidence  ➜ unified source list
4 AUTHOR      A5 per problem (parallel): clinical dossier entry, [n] everywhere
5 VERIFY      A6 3-lens panel → keep/drop/downgrade + confidence
6 CRITIC      A7 gap scan → (if gaps) one more research+author round on the gap
7 SYNTHESIZE  A8 exec brief + interactions + recommendations + red-flags +
                 referrals + work-fitness
8 COMPOSE     A9 editorial blocks + research trail
9 PERSIST     llm_advice = "#DRV1#" + json(dossier); report_ready = True
```

Progress → Django cache per stage; the page animates a live **research trail**.

---

## 6. Question generation (Stage 2) — depth, per person

A3 turns each problem into specific, `S`-anchored questions across seven
archetypes; **lifestyle is mandatory** (≥1 per problem):
`interpretation · occupational · corroboration · workup · risk · threshold ·
lifestyle`. Each carries `search_en` (web) and a distinct `lit_query` (biomedical
phrasing for PubMed/Europe PMC). Near-duplicate questions across problems are
embedded-deduped so a shared question is researched once and cited by many.

---

## 7. Multi-hop research + evidence grading (Stage 3)

A4 is a **real agent loop** per problem (not just pooling):

1. **Plan** — for each question decide channels `{kb, web, literature}` and the
   exact queries; skip channels unlikely to help (keeps latency sane).
2. **Retrieve** — `kb_retrieve` + `read_and_rank(web_search(...))` +
   `lit_search(...)` (PubMed esearch→esummary + Europe PMC). Literature results
   carry title, journal, year, PMID/DOI, abstract snippet.
3. **Follow-up hop (once)** — if a needed number/threshold is still unsupported,
   issue one more targeted query on the weakest channel.
4. **Extract + grade** — return `evidence[] = {claim, source_n, quote, tier}`
   where `tier ∈ {رهنمود, مطالعه, دانش‌نامه, دادهٔ بیمار}`.

Everything pools into ONE numbered source list (patient facts 1..P, then KB, then
web, then literature), reused by every downstream agent so `[7]` is stable.

---

## 8. Clinical authoring (Stage 4) — the single, stronger register

A5 keeps the v2 clinical prompt (the one the user approved) as **the** voice and
deepens it. Per problem it emits a dossier entry:

```jsonc
{ "title":"…", "severity":"high|medium|low", "category":"…", "confidence":0.0..1,
  "lead":"یک جملهٔ سرآمد که یافته را قاب می‌گیرد [n]",
  "sections":[
    {"h":"تعریف و یافته","body":"… [n]"},
    {"h":"تشخیص افتراقی","body":"… [n]"},
    {"h":"سازوکار و ارتباط شغلی","body":"… [n]"},
    {"h":"خطر در صورت بی‌توجهی","body":"… [n]"},
    {"h":"اقدام تشخیصی و درمانی بعدی","body":"… [n]"}
  ],
  "evidence_refs":["Sx"], "key_citations":[n,…] }
```
Rules enforced: numbers only from sources; occupational linkage explicit; no
drug/where-to-go (that's the chatbot); fully Persian; each factual sentence `[n]`.

---

## 9. Verification + completeness critic (Stages 5–6)

- **A6** — 3 skeptic lenses (patient-data support · cited-source support ·
  contradiction/overstatement), run concurrently over all problems; majority
  keeps, else drop/downgrade; confidence = mean.
- **A7 Completeness Critic** — reads the kept dossier + packet and asks: is a
  modality unread (e.g. a lab panel never discussed)? a claim unverified? a
  guideline threshold missing? It returns a short list of gaps; for the top gap
  it spawns one more A4→A5 round, then re-verifies. Bounded by `DR_CRITIC_ROUNDS`.

---

## 10. Report structure — editorial dossier blocks (no score)

`dossier = { blocks:[…], sources:[…], trail:{…}, generated_at }`. Block types:

| type | content |
| --- | --- |
| `masthead` | title, generated date, evidence-base count, "reviewed by 3 lenses" line. |
| `brief` | **executive brief** — 2–4 editorial Persian sentences framing the situation (no score); a quiet severity ledger (n high/medium/low as small labels). |
| `keyvitals` | a restrained clinical readout strip (value · unit · status dot) — editorial, not big colored cards. |
| `figure_eye` | the analyzed conjunctiva crop as a **captioned medical figure** + vision findings + model↔image↔lab agreement. |
| `finding` (×N) | dossier entry: numbered heading, severity label, the clinical sections, evidence chips, confidence, key citations. |
| `recommendations` | prioritized, categorized (incl. **lifestyle**). |
| `redflags` | serious but quiet "seek-care" callout. |
| `workfitness` | occupational fitness verdict + problem interactions. |
| `referrals` | specialty + urgency + CTA → chat assistant. |
| `labs_pointer` | cite/link into the existing `#medical-tests` table. |
| `references` | the **sources apparatus**: numbered, grouped, each with an evidence-tier badge; web/lit are links (PMID/DOI shown). |
| `trail` | the **agentic research trail** (signature): problems triaged, questions asked, sources by channel (KB/web/PubMed/EuropePMC), reviewers, critic rounds. |

---

## 11. Front-end direction — the editorial dossier (world-class)

Design language = **a serious research document**, not an app dashboard. What the
top labs' deep-research UIs share, and what we adopt:

- **Reading column.** A single, generous measure (max ~72ch), large line-height,
  strong typographic hierarchy. Whitespace does the work; almost no boxes.
- **Masthead + meta.** An editorial title block with a hairline rule and a mono
  meta row (date · N sources · reviewed). Feels like a published report.
- **Section craft.** Eyebrow label (mono, tracked) + large Persian heading +
  hairline. Findings are numbered like a dossier (`یافتهٔ ۰۱`).
- **Signature: the research trail.** An elegant, collapsible "چگونه این گزارش
  ساخته شد" rail that shows the agentic pipeline and per-channel source counts —
  the thing that signals real deep research. This is the one bold element.
- **Restraint in color.** Ink-forward and quiet; the page's purple only for
  citations/links/markers; clinical status via *small* dots/labels, never big
  red/green fills. A monospace face for data, citations, PMIDs, and meta.
- **Citations as apparatus.** Inline `[n]` are refined superscript chips that
  link to a proper **references** section with evidence-tier badges — like a
  journal article, not a chat bubble.
- **Motion, sparingly.** Fade-up on scroll, citation hover, one subtle trail
  draw-in. Reduced-motion respected. RTL, light/dark, mobile.

New CSS/JS namespace `.hr-*` (health-research). The old `.dr-*` block styles and
renderer are removed entirely.

---

## 12. File map

```
test_analysis/services/deep_research/
  packet.py            # A0 (unchanged core; keeps eye crop urls/paths)
  tools/
    llm_json.py        # + model tiers (SYNTH/REASONING/VISION)
    vision.py          # A1 tool
    web.py             # web + KB
    literature.py      # ⭐ NEW  PubMed E-utilities + Europe PMC
  eye_vision.py        # A1
  triage.py            # A2
  questions.py         # A3  (+ lit_query, lifestyle mandatory)
  research.py          # A4  planner + multi-hop + evidence grading
  author.py            # A5  single clinical register (no plain_fa)
  verify.py            # A6
  critic.py            # ⭐ NEW  A7 completeness critic
  synthesis.py         # A8  (no score; exec brief)
  blocks.py            # A9  editorial dossier blocks + trail
  runner.py            # orchestration (adds critic stage, trail, no score)
templates/test_analysis/
  profile_detail2.html # NEW .hr-* editorial styles + renderer (old .dr-* removed)
```

---

## 13. Build order

1. `literature.py` (PubMed + Europe PMC, best-effort) + model tiers in `llm_json`.
2. `research.py` → real planner + multi-hop + graded evidence + literature channel.
3. `author.py` → single clinical register (drop `plain_fa`); deepen sections.
4. `critic.py` (A7) + wire the extra round into `runner.py`.
5. `synthesis.py` → remove score, add executive brief.
6. `blocks.py` → editorial block set + research trail; `runner.py` trail assembly.
7. Front-end: brand-new `.hr-*` editorial dossier CSS + renderer; delete `.dr-*`.
8. Verify: `runserver`-only run, `makemigrations --check → No changes`,
   screenshots in light + dark.

---

## 14. Why v3 is stronger (summary)

Private multimodal eye reading **+** this person's exact numbers **+** occupational
exposure linkage **+** *fresh, graded biomedical literature* (PubMed/Europe PMC)
**+** adversarial verification **+** a completeness critic that closes gaps —
presented as a rigorous, fully-Persian clinical dossier. No generic model handed
the same text summary can match it, and it now *looks* like the deep-research
product it is.
