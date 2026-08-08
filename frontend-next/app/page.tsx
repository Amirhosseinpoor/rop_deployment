"use client";
import { useEffect, useRef, useState } from "react";
import dynamic from "next/dynamic";
import { gsap } from "gsap";
import { ScrollTrigger } from "gsap/ScrollTrigger";
import SmoothScroll from "@/components/SmoothScroll";
import { STATS, PERSONAS, AGENTS, MODELS, ASSISTANTS, ARCH } from "@/lib/data";

const HeroScene = dynamic(() => import("@/components/HeroScene"), { ssr: false });

const S = "/static/landing/media"; // asset root (served by Django)
const APP = `${S}/app`;

function Arrow() {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2.2}>
      <path d="M5 12h14M13 6l6 6-6 6" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

function Reveal({ children, className = "", delay = 0 }: { children: React.ReactNode; className?: string; delay?: number }) {
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const io = new IntersectionObserver(
      ([e]) => {
        if (e.isIntersecting) {
          el.style.transitionDelay = `${delay}ms`;
          el.classList.add("in");
          io.unobserve(el);
        }
      },
      { threshold: 0.14, rootMargin: "0px 0px -7% 0px" }
    );
    io.observe(el);
    return () => io.disconnect();
  }, [delay]);
  return (
    <div ref={ref} className={`reveal ${className}`}>
      {children}
    </div>
  );
}

function Counter({ to }: { to: number }) {
  const ref = useRef<HTMLSpanElement>(null);
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const io = new IntersectionObserver(
      ([e]) => {
        if (!e.isIntersecting) return;
        io.disconnect();
        let start: number | null = null;
        const step = (t: number) => {
          if (start === null) start = t;
          const p = Math.min((t - start) / 1100, 1);
          el.textContent = String(Math.round(to * (1 - Math.pow(1 - p, 3))));
          if (p < 1) requestAnimationFrame(step);
        };
        requestAnimationFrame(step);
      },
      { threshold: 0.6 }
    );
    io.observe(el);
    return () => io.disconnect();
  }, [to]);
  return <span ref={ref}>0</span>;
}

export default function Home() {
  const [persona, setPersona] = useState(0);
  const [agent, setAgent] = useState(0);

  useEffect(() => {
    const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    if (reduce) return;
    gsap.registerPlugin(ScrollTrigger);

    const ctx = gsap.context(() => {
      // Hero parallax + fade on scroll-away
      gsap.to(".hero-copy", {
        yPercent: -18,
        opacity: 0,
        ease: "none",
        scrollTrigger: { trigger: ".hero", start: "top top", end: "bottom top", scrub: true },
      });
      gsap.to(".hero-scene", {
        yPercent: 12,
        ease: "none",
        scrollTrigger: { trigger: ".hero", start: "top top", end: "bottom top", scrub: true },
      });

      // Pinned horizontal gallery of real app screens
      const track = document.querySelector<HTMLElement>(".gallery-track");
      const gallery = document.querySelector<HTMLElement>(".gallery-section");
      if (track && gallery) {
        const amount = () => track.scrollWidth - window.innerWidth + 80;
        gsap.to(track, {
          x: () => -amount(),
          ease: "none",
          scrollTrigger: {
            trigger: gallery,
            start: "top top",
            end: () => "+=" + amount(),
            scrub: 0.6,
            pin: true,
            anticipatePin: 1,
            invalidateOnRefresh: true,
          },
        });
      }

      // Agent pipeline: step through A0..A9 as it scrolls
      ScrollTrigger.create({
        trigger: ".agents-section",
        start: "top 62%",
        end: "bottom 60%",
        onUpdate: (self) => {
          const i = Math.max(0, Math.min(AGENTS.length - 1, Math.floor(self.progress * AGENTS.length)));
          setAgent(i);
        },
      });

      // Parallax for floating frames
      gsap.utils.toArray<HTMLElement>(".para").forEach((el) => {
        const d = parseFloat(el.dataset.depth || "10");
        gsap.to(el, {
          yPercent: d,
          ease: "none",
          scrollTrigger: { trigger: el, start: "top bottom", end: "bottom top", scrub: true },
        });
      });
    });

    const t = setTimeout(() => ScrollTrigger.refresh(), 500);
    return () => {
      clearTimeout(t);
      ctx.revert();
    };
  }, []);

  return (
    <SmoothScroll>
      {/* ===== NAV ===== */}
      <header className="nav">
        <div className="nav-inner">
          <a className="brand" href="/">
            <span className="brand-mark" />
            Mediverse<span className="grad-text">AI</span>
          </a>
          <nav className="nav-links">
            <a href="#diagnostics">Diagnostics</a>
            <a href="#platform-inside">Platform</a>
            <a href="#marketplace">AI Doctors</a>
            <a href="#studio">Studio</a>
            <a href="#engine">Engine</a>
          </nav>
          <div className="nav-cta">
            <a href="/login/" className="btn nav-signin">Sign in</a>
            <a href="/signup/choose-role/" className="btn btn-accent">
              Request access <Arrow />
            </a>
          </div>
        </div>
      </header>

      {/* ===== HERO ===== */}
      <section className="hero">
        <div className="hero-fluid" style={{ backgroundImage: `url(${S}/abstract-fluid-hero-01.jpg)` }} />
        <div className="hero-scene">
          <HeroScene />
        </div>
        <div className="hero-veil" />
        <div className="grid-lines" />
        <div className="wrap hero-copy">
          <div className="eyebrow">Mediverse AI · Clinical intelligence platform</div>
          <h1 className="display hero-title">
            See the<br />
            <span className="grad-text">whole patient.</span>
          </h1>
          <p className="hero-lede">
            From an infant&apos;s retinal fundus to a workforce&apos;s lab panels, Mediverse AI reads medical images,
            documents and workplace exposures — then returns <b>cited, doctor-grade</b> diagnostics and work-fitness
            decisions in minutes.
          </p>
          <div className="hero-cta">
            <a href="/login/" className="btn btn-accent">
              Enter the platform <Arrow />
            </a>
            <a href="/market/" className="btn btn-ghost">Meet the AI doctors</a>
          </div>
          <div className="hero-stats">
            {STATS.map((s, i) => (
              <div className="hstat" key={i}>
                <div className="hstat-n display">
                  <Counter to={s.n} />
                </div>
                <div className="hstat-l mono">{s.label}</div>
              </div>
            ))}
          </div>
        </div>
        <div className="scroll-cue mono">
          <span>SCROLL</span>
          <i />
        </div>
      </section>

      {/* ===== MARQUEE ===== */}
      <div className="marquee">
        <div className="marquee-row">
          {[...Array(2)].map((_, k) => (
            <div className="marquee-inner" key={k} aria-hidden={k === 1}>
              {["EfficientNet-B4", "EyeNet · CBAM · MHSA", "U-Net++", "U²-Net", "Attention ResUNet", "Random Forest", "ResNet18", "EfficientNet-B0", "ICROP 3rd Ed."].map((m, i) => (
                <span key={i}>
                  {m} <b>✦</b>
                </span>
              ))}
            </div>
          ))}
        </div>
      </div>

      {/* ===== CINEMATIC: THE EYE ===== */}
      <section className="cine">
        <video className="cine-vid" autoPlay muted loop playsInline preload="metadata" poster={`${S}/eye-macro-01.jpg`}>
          <source src={`${S}/eye-macro-loop.mp4`} type="video/mp4" />
        </video>
        <div className="cine-veil" />
        <div className="cine-inner">
          <div className="eyebrow">The signal is in the tissue</div>
          <h2 className="display">
            Everything begins <span className="grad-text">with the eye.</span>
          </h2>
          <p>
            Vessels, pallor, ectasia, tortuosity — the eye is the one window where a camera reaches living
            microvasculature. Mediverse reads it at the pixel, then reasons across the whole body.
          </p>
        </div>
      </section>

      {/* ===== PERSONAS ===== */}
      <section className="band" id="personas">
        <div className="wrap">
          <Reveal className="sec-head center">
            <div className="eyebrow" style={{ justifyContent: "center" }}>One platform · four vantage points</div>
            <h2>Built for everyone who touches the chart.</h2>
            <p>Patients, clinicians, managers and administrators each get a portal tuned to their decisions — sharing one system of record, one audit trail, and one AI core.</p>
          </Reveal>

          <Reveal className="persona-tabs">
            {PERSONAS.map((p, i) => (
              <button key={i} className={`ptab ${persona === i ? "active" : ""}`} onClick={() => setPersona(i)}>
                <span className="mono">{p.tag}</span> {p.who}
              </button>
            ))}
          </Reveal>

          <Reveal className="persona-panel card">
            <div className="persona-copy">
              <div className="eyebrow iris">{PERSONAS[persona].who}</div>
              <h3 className="display">{PERSONAS[persona].lead}</h3>
              <p>{PERSONAS[persona].body}</p>
            </div>
            <div className="persona-routes">
              {PERSONAS[persona].routes.map(([label, href], i) => (
                <a key={i} className="route" href={href}>
                  <span className="rt">{label}</span>
                  <span className="rg"><Arrow /></span>
                </a>
              ))}
            </div>
          </Reveal>
        </div>
      </section>

      {/* ===== DIAGNOSTICS ===== */}
      <section className="band" id="diagnostics">
        <div className="glow" style={{ width: 520, height: 520, background: "#7c5cff", top: 100, left: -120 }} />
        <div className="wrap">
          <Reveal className="sec-head">
            <div className="eyebrow">01 — Ophthalmic AI diagnostics</div>
            <h2>Reading the eye at the resolution of a subspecialist.</h2>
            <p>Two deep-learning pipelines turn fundus photographs and corneal maps into staged, zoned, guideline-anchored diagnoses — with an expert override portal built in.</p>
          </Reveal>

          {/* ROP */}
          <div className="feature">
            <Reveal className="fc">
              <div className="tag"><i style={{ background: "var(--cyan)" }} /> Retinopathy of Prematurity</div>
              <h3 className="display">Infant retinal screening, staged and zoned automatically.</h3>
              <p className="fmuted">Upload one or many fundus photos per infant. Mediverse segments the vessel tree, disk and macula, then grades disease and localizes it to the optic nerve.</p>
              <ul className="flist">
                <li><b>Plus disease</b> — Normal · Pre-Plus · Plus via EfficientNet-B4.</li>
                <li><b>Severity &amp; zonal mapping</b> — Stage 0–5, localized to Zone I / II / III.</li>
                <li><b>ICROP 3rd-edition support</b> — laser / anti-VEGF vs. 48-hour follow-up.</li>
                <li><b>Doctor review portal</b> — override machine labels, record notes.</li>
              </ul>
              <a href="/rop/" className="btn btn-ghost">Open ROP screening <Arrow /></a>
            </Reveal>
            <Reveal className="fv" delay={80}>
              <div className="fv-back bl"><img src={`${S}/eye-macro-02.jpg`} alt="" /></div>
              <div className="frame para" data-depth="-6">
                <div className="bar"><i /><i /><i /><span className="u">mediverse.ai/rop</span></div>
                <div className="shot"><img src={`${APP}/rop.png`} alt="ROP screening interface" /></div>
              </div>
              <div className="chip-float cf-a"><span className="mono">Zone II</span> Plus · Stage 2</div>
            </Reveal>
          </div>

          {/* KC */}
          <div className="feature rev">
            <Reveal className="fc">
              <div className="tag"><i style={{ background: "var(--magenta)" }} /> Keratoconus &amp; Corneal Topography</div>
              <h3 className="display">Dual-eye ectasia staging with a Siamese network.</h3>
              <p className="fmuted">The EyeNet architecture compares left (OD) and right (OS) corneal maps pairwise, grading ectasia and scoring structural symmetry between the eyes.</p>
              <ul className="flist">
                <li><b>5-tier severity</b> — Normal · ATN · NEIr · EIr · eKCN.</li>
                <li><b>Bilateral symmetry</b> — SfRS vs. NSfRS refractive status.</li>
                <li><b>Ocular-linked risk</b> — hypertension &amp; cardiorespiratory estimates.</li>
              </ul>
              <a href="/kc/" className="btn btn-ghost">Open KC topography <Arrow /></a>
            </Reveal>
            <Reveal className="fv" delay={80}>
              <div className="fv-back br"><img src={`${S}/eye-macro-03.jpg`} alt="" /></div>
              <div className="frame para" data-depth="6">
                <div className="bar"><i /><i /><i /><span className="u">mediverse.ai/kc</span></div>
                <div className="shot"><img src={`${APP}/kc.png`} alt="Keratoconus topography interface" /></div>
              </div>
              <div className="chip-float cf-b"><span className="mono">Max K</span> 54.2 D · eKCN</div>
            </Reveal>
          </div>
        </div>
      </section>

      {/* ===== REAL APP GALLERY (pinned horizontal) ===== */}
      <section className="gallery-section" id="platform-inside">
        <div className="gallery-head wrap">
          <div className="eyebrow">Inside the platform</div>
          <h2 className="display">Real product. Not a render.</h2>
        </div>
        <div className="gallery-track">
          {[
            ["market.png", "Doctor Marketplace", "/market/", "24/7 specialist AI directory"],
            ["studio.png", "Clinician Studio", "/market/studio/", "Build & manage AI doctors"],
            ["managing.png", "Manager Console", "/managing/", "Company modules & staff"],
            ["market_doctor.png", "Specialist Profile", "/market/doctor/heart-disease/", "Bilingual persona + tools"],
            ["history.png", "Diagnostic Audit", "/history/", "Every prediction, logged"],
          ].map(([img, title, href, sub], i) => (
            <a className="gcard" href={href as string} key={i}>
              <div className="frame">
                <div className="bar"><i /><i /><i /><span className="u">mediverse.ai{href}</span></div>
                <div className="shot"><img src={`${APP}/${img}`} alt={title as string} /></div>
              </div>
              <div className="gcap">
                <div className="gt display">{title}</div>
                <div className="gs mono">{sub}</div>
              </div>
            </a>
          ))}
        </div>
      </section>

      {/* ===== OCCUPATIONAL + AGENT ENGINE ===== */}
      <section className="band agents-section" id="engine">
        <video className="engine-vid" autoPlay muted loop playsInline preload="metadata" poster={`${S}/particles-blue-dark-01.jpg`}>
          <source src={`${S}/network-plexus-loop.mp4`} type="video/mp4" />
        </video>
        <div className="glow" style={{ width: 600, height: 600, background: "#7c5cff", top: -80, right: -160, opacity: 0.35 }} />
        <div className="wrap">
          <Reveal className="sec-head">
            <div className="eyebrow iris">02 — Deep Health Research v3 · signature engine</div>
            <h2>Ten autonomous agents compile a cited clinical dossier.</h2>
            <p>No health score. Sentence-level evidence [n] citations across PubMed, Europe PMC, web RAG and your internal knowledge base — with an adversarial verifier voting on every claim.</p>
          </Reveal>

          <div className="agent-stage">
            <div className="agent-rail">
              {AGENTS.map((a, i) => (
                <button key={i} className={`arow ${agent === i ? "on" : ""} ${i < agent ? "done" : ""}`} onMouseEnter={() => setAgent(i)}>
                  <span className="aid mono">{a[0]}</span>
                  <span className="an">{a[1]}</span>
                  <span className="abar"><span style={{ width: agent >= i ? "100%" : "0%" }} /></span>
                </button>
              ))}
            </div>
            <div className="agent-detail card">
              <div className="ad-id display grad-text">{AGENTS[agent][0]}</div>
              <div className="ad-name display">{AGENTS[agent][1]}</div>
              <div className="ad-model mono">{AGENTS[agent][2]}</div>
              <p className="ad-body">{AGENTS[agent][3]}</p>
              <div className="ad-trail mono">
                <span>PROBLEMS TRIAGED</span><span>QUESTIONS ASKED</span><span>SOURCES BY CHANNEL</span><span>SKEPTIC VOTES</span>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* ===== MARKETPLACE ===== */}
      <section className="band split" id="marketplace">
        <div className="wrap split-grid">
          <Reveal className="fc">
            <div className="eyebrow">03 — Virtual doctor marketplace</div>
            <h2 className="display split-h">Specialist AI doctors, on call 24/7.</h2>
            <p className="fmuted">Streaming, multi-modal consultations — each doctor with its own persona, private knowledge base, clinical tools, voice, and WhatsApp follow-up.</p>
            <div className="chips">
              {["Nephrology", "Cardiology", "Endocrinology", "Hematology", "Hypertension", "Family"].map((c) => (
                <span className="chip" key={c}><i /> {c}</span>
              ))}
            </div>
            <div className="badges">
              {["Whisper voice + TTS", "Inline [n] citations", "Quote-reply & edit-fork", "WhatsApp reminders"].map((b) => (
                <span className="badge mono" key={b}>{b}</span>
              ))}
            </div>
            <a href="/market/" className="btn btn-accent">Browse AI doctors <Arrow /></a>
          </Reveal>
          <Reveal className="fv" delay={80}>
            <div className="frame para" data-depth="8">
              <div className="bar"><i /><i /><i /><span className="u">mediverse.ai/market</span></div>
              <div className="shot"><img src={`${APP}/market.png`} alt="AI doctor marketplace" /></div>
            </div>
          </Reveal>
        </div>
      </section>

      {/* ===== STUDIO ===== */}
      <section className="band split" id="studio">
        <div className="wrap split-grid rev">
          <Reveal className="fc">
            <div className="eyebrow iris">04 — Clinician studio</div>
            <h2 className="display split-h">Build and deploy your own specialist.</h2>
            <p className="fmuted">Design custom AI doctors — persona, knowledge base, avatar and clinical tools — without writing a prompt from scratch.</p>
            <div className="mini-grid">
              {[
                ["Copilot prompt builder", "Auto-constructs personas & behavioral guidelines"],
                ["Generative avatars", "Photo-realistic doctor portraits by specialty"],
                ["Custom RAG knowledge", "Upload guidelines/protocols; on-demand re-indexing"],
                ["14 clinical tools", "Calculators, lab interpreter, drug interactions, PubMed"],
              ].map(([t, d]) => (
                <div className="mini card" key={t}>
                  <b>{t}</b>
                  <span>{d}</span>
                </div>
              ))}
            </div>
            <a href="/market/studio/" className="btn btn-accent">Enter the studio <Arrow /></a>
          </Reveal>
          <Reveal className="fv" delay={80}>
            <div className="frame para" data-depth="-8">
              <div className="bar"><i /><i /><i /><span className="u">mediverse.ai/market/studio</span></div>
              <div className="shot"><img src={`${APP}/studio.png`} alt="Clinician studio" /></div>
            </div>
          </Reveal>
        </div>
      </section>

      {/* ===== PLATFORM ===== */}
      <section className="band" id="platform">
        <div className="wrap">
          <Reveal className="sec-head">
            <div className="eyebrow">The platform underneath</div>
            <h2>Distributed by design, private by default.</h2>
            <p>A Django system of record, a decoupled FastAPI RAG microservice, Celery workers for heavy inference, and strict multi-tenant isolation with PII redaction.</p>
          </Reveal>

          <div className="plat-grid">
            <Reveal className="arch card">
              {ARCH.map(([id, title, sub, port], i) => (
                <div className="anode" key={i}>
                  <span className="ai mono">{id}</span>
                  <div className="am">
                    <b>{title}</b>
                    <span>{sub}</span>
                  </div>
                  <span className="ap mono">{port}</span>
                </div>
              ))}
            </Reveal>
            <Reveal className="sec-card card" delay={60}>
              <h3 className="display">Patient data stays scoped.</h3>
              <ul className="seclist">
                <li><b>Automated PII redaction</b><span>Extraction keeps only analytes, units & flags — names, IDs and DOBs discarded.</span></li>
                <li><b>Multi-tenant isolation</b><span>Queries enforce company scoping; you see only your registered entity.</span></li>
                <li><b>Immutable audit trail</b><span>Every prediction, override and note is timestamped to the acting user.</span></li>
              </ul>
            </Reveal>
          </div>

          <Reveal className="models">
            <div className="mono models-label">Eight pretrained model checkpoints in production</div>
            <div className="models-grid">
              {MODELS.map(([arch, task, file], i) => (
                <div className="mcell card" key={i}>
                  <div className="march mono grad-text">{arch}</div>
                  <div className="mtask">{task}</div>
                  <div className="mfile mono">{file}</div>
                </div>
              ))}
            </div>
          </Reveal>
        </div>
      </section>

      {/* ===== ASSISTANTS ===== */}
      <section className="band" id="assistants">
        <div className="wrap">
          <Reveal className="sec-head center">
            <div className="eyebrow" style={{ justifyContent: "center" }}>Context-aware everywhere</div>
            <h2>Seven assistants, one per clinical surface.</h2>
            <p>Every page carries an AI assistant tuned to its domain, role and task — from ROP decision support to the global RAG microservice.</p>
          </Reveal>
          <div className="bot-grid">
            {ASSISTANTS.map(([n, name, path, desc], i) => (
              <Reveal className="bot card" key={i} delay={(i % 4) * 60}>
                <div className="bn mono">{n}</div>
                <h4>{name}</h4>
                <div className="bp mono">{path}</div>
                <p>{desc}</p>
              </Reveal>
            ))}
          </div>
        </div>
      </section>

      {/* ===== CTA ===== */}
      <section className="band cta-band">
        <div className="wrap">
          <Reveal className="cta card">
            <video className="cta-vid" autoPlay muted loop playsInline preload="metadata" poster={`${S}/particles-blue-dark-01.jpg`}>
              <source src={`${S}/holographic-hud-loop.mp4`} type="video/mp4" />
            </video>
            <div className="grid-lines" />
            <div className="eyebrow" style={{ justifyContent: "center" }}>Start where you stand</div>
            <h2 className="display">Bring clinical intelligence to your whole organization.</h2>
            <p>Screen an eye, audit a workforce, or spin up a specialist AI doctor — the platform meets your role at the door.</p>
            <div className="cta-row">
              <a href="/signup/choose-role/" className="btn btn-accent">Request access <Arrow /></a>
              <a href="/login/" className="btn btn-ghost">Sign in</a>
              <a href="/market/" className="btn btn-ghost">Explore AI doctors</a>
            </div>
          </Reveal>
        </div>
      </section>

      {/* ===== FOOTER ===== */}
      <footer className="ft">
        <div className="wrap ft-grid">
          <div className="ft-about">
            <a className="brand" href="/"><span className="brand-mark" /> Mediverse<span className="grad-text">AI</span></a>
            <p>Automated ophthalmic diagnostics, occupational health auditing, agentic clinical research and AI virtual consultations — one platform, four personas.</p>
          </div>
          <div className="ft-col">
            <h5 className="mono">Diagnostics</h5>
            <a href="/rop/">ROP Screening</a><a href="/kc/">Keratoconus</a><a href="/health/profile/">Health Profiles</a><a href="/health/doctor-dashboard/">Doctor Dashboard</a>
          </div>
          <div className="ft-col">
            <h5 className="mono">Platform</h5>
            <a href="/market/">Marketplace</a><a href="/market/studio/">Studio</a><a href="/history/">Audit History</a><a href="#platform">Architecture</a>
          </div>
          <div className="ft-col">
            <h5 className="mono">Get started</h5>
            <a href="/login/">Sign in</a><a href="/signup/choose-role/">Request access</a><a href="/signup/manager/">Register a company</a><a href="/signup/doctor/">Clinician signup</a>
          </div>
        </div>
        <div className="wrap ft-bottom mono">
          <span>© Mediverse AI · Aras AI — Clinical intelligence platform</span>
          <span>Django · FastAPI · Celery · Redis · WAHA</span>
        </div>
      </footer>
    </SmoothScroll>
  );
}
