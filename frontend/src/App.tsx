import React, { useEffect, useState } from 'react';
import Lenis from 'lenis';
import { motion, useScroll, useTransform } from 'framer-motion';
import {
  Activity,
  Brain,
  ShieldCheck,
  Stethoscope,
  Building2,
  Database,
  Cpu,
  ArrowRight,
  Sparkles,
  Search,
  MessageSquare,
  CheckCircle2,
  AlertTriangle,
  XCircle,
  Mic,
  Zap,
  ChevronRight,
  Send,
  UserCheck
} from 'lucide-react';

export default function App() {
  // 1. Lenis Smooth Scroll Initialization
  useEffect(() => {
    const lenis = new Lenis({
      duration: 1.2,
      easing: (t) => Math.min(1, 1.001 - Math.pow(2, -10 * t)),
      smoothWheel: true,
    });

    function raf(time: number) {
      lenis.raf(time);
      requestAnimationFrame(raf);
    }
    requestAnimationFrame(raf);

    return () => {
      lenis.destroy();
    };
  }, []);

  // Mouse spotlight position state
  const [mousePos, setMousePos] = useState({ x: 0, y: 0 });
  const handleMouseMove = (e: React.MouseEvent) => {
    setMousePos({ x: e.clientX, y: e.clientY });
  };

  // Scroll animations
  const { scrollY } = useScroll();
  const heroY = useTransform(scrollY, [0, 500], [0, 150]);
  const heroOpacity = useTransform(scrollY, [0, 400], [1, 0]);

  // Terminal Typing Animation State for Section 3 (Deep Research v3)
  const [typingIndex, setTypingIndex] = useState(0);
  const typingText = `[A0 Evidence Packet] Computing deterministic vitals: BP 138/88, BMI 28.4, eGFR 74.
[A1 Eye Vision Analyst] VLM conjunctiva crop read: Normal pallor (Anemia Neg).
[A2 Triage Agent] 2 cross-modal problems ranked: Hypertension (High), Dyslipidemia (Med).
[A3 Question Strategist] Formulated 6 patient-anchored queries across 7 archetypes.
[A4 Research Planner] Executing PubMed & Europe PMC API queries... Found 18 studies.
[A5 Clinical Author] Synthesizing dossier sections with sentence-level citations [1].
[A6 Adversarial Verifier] 3/3 skeptic AI lenses passed data consistency audit.
[A7 Completeness Critic] Modality audit complete. No unread lab panels detected.
[A8 Synthesis Agent] Formulated executive brief, recommendations, and fitness verdict.
[A9 Dossier Composer] Assembled editorial blocks & Agentic Research Trail.
STATUS: FIT WITH CONDITIONS (مشروط) [2]`;

  useEffect(() => {
    const interval = setInterval(() => {
      setTypingIndex((prev) => (prev < typingText.length ? prev + 1 : prev));
    }, 24);
    return () => clearInterval(interval);
  }, []);

  // Array of 10 Autonomous Agents for display
  const agentsList = [
    { id: 'A0', name: 'Evidence Packet', role: 'Deterministic Vitals/Labs Computation (packet.py)' },
    { id: 'A1', name: 'Eye Vision Analyst', role: 'VLM Conjunctiva Crop Inspection & Pallor' },
    { id: 'A2', name: 'Triage Agent', role: 'Cross-Modal Problem Severity Ranking' },
    { id: 'A3', name: 'Question Strategist', role: '7-Archetype Query Formulation (lit_query)' },
    { id: 'A4', name: 'Research Planner', role: 'Multi-Hop PubMed & Europe PMC Search' },
    { id: 'A5', name: 'Clinical Author', role: 'Single Register Persian Authoring w/ [n] Citations' },
    { id: 'A6', name: 'Adversarial Verifier', role: '3-Lens Skeptic Data Consistency Audit' },
    { id: 'A7', name: 'Completeness Critic', role: 'Gap Scan & 2nd Round Research Trigger' },
    { id: 'A8', name: 'Synthesis & Work Fitness', role: 'Executive Brief, Red Flags & Fitness Verdict' },
    { id: 'A9', name: 'Dossier Composer', role: 'Editorial Blocks & Agentic Research Trail' },
  ];

  return (
    <div
      onMouseMove={handleMouseMove}
      className="relative min-h-screen bg-[#0a0a0a] text-slate-100 selection:bg-cyan-500/20 selection:text-cyan-300 overflow-x-hidden font-sans"
    >
      {/* Dynamic Cursor Light Spotlight */}
      <div
        className="pointer-events-none fixed inset-0 z-30 transition-opacity duration-300"
        style={{
          background: `radial-gradient(600px circle at ${mousePos.x}px ${mousePos.y}px, rgba(0, 229, 255, 0.05), transparent 80%)`,
        }}
      />

      {/* ========================================================================= */}
      {/* NAVIGATION BAR */}
      {/* ========================================================================= */}
      <header className="fixed top-0 left-0 right-0 z-50 backdrop-blur-xl bg-[#0a0a0a]/70 border-b border-white/[0.08]">
        <div className="max-w-7xl mx-auto px-6 h-20 flex items-center justify-between">
          <a href="/" className="flex items-center gap-3 group">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-cyan-500 to-blue-600 p-[1px] shadow-lg shadow-cyan-500/20 group-hover:shadow-cyan-500/40 transition-all duration-300">
              <div className="w-full h-full bg-[#0a0a0a] rounded-[11px] flex items-center justify-center">
                <Brain className="w-5 h-5 text-cyan-400 group-hover:scale-110 transition-transform" />
              </div>
            </div>
            <div className="flex flex-col">
              <span className="font-extrabold text-lg tracking-tight bg-clip-text text-transparent bg-gradient-to-r from-white via-slate-100 to-slate-400">
                Mediverse <span className="text-cyan-400">AI</span>
              </span>
              <span className="text-[10px] tracking-widest text-slate-500 font-mono uppercase">Aras AI Engine</span>
            </div>
          </a>

          <nav className="hidden md:flex items-center gap-8 text-sm font-medium text-slate-400">
            <a href="#diagnostics" className="hover:text-cyan-400 transition-colors">Diagnostics</a>
            <a href="#deep-research" className="hover:text-cyan-400 transition-colors">Deep Research v3</a>
            <a href="#marketplace" className="hover:text-cyan-400 transition-colors">AI Marketplace</a>
            <a href="#governance" className="hover:text-cyan-400 transition-colors">Governance</a>
            <a href="#architecture" className="hover:text-cyan-400 transition-colors">Architecture</a>
          </nav>

          <div className="flex items-center gap-4">
            <a
              href="/login/"
              className="relative inline-flex items-center gap-2 px-5 py-2.5 rounded-xl font-semibold text-xs text-white bg-gradient-to-r from-cyan-500 to-blue-600 shadow-lg shadow-cyan-500/25 hover:shadow-cyan-500/40 hover:scale-[1.02] active:scale-[0.98] transition-all duration-200"
            >
              <span>Enter Portal</span>
              <ArrowRight className="w-4 h-4" />
            </a>
          </div>
        </div>
      </header>

      {/* ========================================================================= */}
      {/* SECTION 1: HERO SECTION */}
      {/* ========================================================================= */}
      <section className="relative pt-36 pb-24 md:pt-48 md:pb-36 overflow-hidden">
        {/* Background Grid & Aurora Cone */}
        <div className="absolute inset-0 bg-aurora-grid opacity-30 pointer-events-none" />
        <div className="absolute top-1/4 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[800px] h-[400px] bg-gradient-to-tr from-cyan-500/15 via-purple-600/15 to-blue-600/10 blur-[140px] rounded-full pointer-events-none" />

        <div className="max-w-7xl mx-auto px-6 relative z-10">
          <motion.div
            style={{ y: heroY, opacity: heroOpacity }}
            className="flex flex-col items-center text-center max-w-4xl mx-auto"
          >
            {/* Status Badge */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5 }}
              className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full border border-cyan-500/30 bg-cyan-500/10 backdrop-blur-md mb-8 text-cyan-300 text-xs font-semibold tracking-wide"
            >
              <span className="relative flex h-2 w-2">
                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-cyan-400 opacity-75"></span>
                <span className="relative inline-flex rounded-full h-2 w-2 bg-cyan-500"></span>
              </span>
              <span>Aras AI Engine v3.2 Active • 10-Agent Clinical RAG</span>
            </motion.div>

            {/* Massive Heading */}
            <motion.h1
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6, delay: 0.1 }}
              className="text-4xl md:text-7xl font-extrabold tracking-tight text-white leading-[1.08] mb-6"
            >
              Mediverse AI: The Next Era of <br className="hidden md:inline" />
              <span className="text-gradient-cyan">Clinical Intelligence.</span>
            </motion.h1>

            {/* Subheadline */}
            <motion.p
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6, delay: 0.2 }}
              className="text-lg md:text-xl text-slate-400 font-normal leading-relaxed mb-10 max-w-2xl"
            >
              Automated ophthalmic diagnostics, 10-agent clinical research, and AI-driven virtual consultations. Powered by Aras AI.
            </motion.p>

            {/* CTA Buttons */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6, delay: 0.3 }}
              className="flex flex-col sm:flex-row items-center gap-4 w-full sm:w-auto"
            >
              <a
                href="/login/"
                className="w-full sm:w-auto px-8 py-4 rounded-xl font-bold text-sm text-white bg-gradient-to-r from-cyan-500 via-blue-600 to-purple-600 shadow-xl shadow-cyan-500/25 hover:shadow-cyan-500/40 hover:scale-[1.02] active:scale-[0.98] transition-all duration-200 flex items-center justify-center gap-3"
              >
                <Sparkles className="w-4 h-4 text-cyan-200" />
                <span>Enter Enterprise Portal</span>
              </a>

              <a
                href="#architecture"
                className="w-full sm:w-auto px-8 py-4 rounded-xl font-semibold text-sm text-slate-300 glass-panel hover:bg-white/[0.08] border border-white/10 hover:border-white/20 transition-all duration-200 flex items-center justify-center gap-2"
              >
                <span>Explore Architecture</span>
                <ChevronRight className="w-4 h-4 text-slate-500" />
              </a>
            </motion.div>
          </motion.div>

          {/* 3D Tilted Dashboard Showcase Mockup */}
          <motion.div
            initial={{ opacity: 0, y: 60, rotateX: 20 }}
            animate={{ opacity: 1, y: 0, rotateX: 10 }}
            transition={{ duration: 0.8, delay: 0.4 }}
            style={{ perspective: 1200 }}
            className="mt-16 relative max-w-5xl mx-auto"
          >
            {/* Glowing Backdrop Rim */}
            <div className="absolute -inset-1 rounded-3xl bg-gradient-to-r from-cyan-500 to-purple-600 opacity-30 blur-2xl pointer-events-none" />

            <div className="relative glass-panel-glow rounded-2xl p-4 md:p-6 border border-white/15 overflow-hidden shadow-2xl">
              {/* Mock App Top Window Bar */}
              <div className="flex items-center justify-between pb-4 mb-4 border-b border-white/10">
                <div className="flex items-center gap-2">
                  <div className="w-3 h-3 rounded-full bg-rose-500/80" />
                  <div className="w-3 h-3 rounded-full bg-amber-500/80" />
                  <div className="w-3 h-3 rounded-full bg-emerald-500/80" />
                  <span className="ml-2 text-xs font-mono text-slate-500">mediverse.ai/dashboard/clinical-audit</span>
                </div>
                <div className="flex items-center gap-3">
                  <span className="px-2.5 py-1 rounded-full text-[10px] font-mono bg-cyan-500/10 text-cyan-400 border border-cyan-500/30">
                    Django Monolith + FastAPI Microservice
                  </span>
                </div>
              </div>

              {/* Mock Dashboard Grid */}
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                {/* Mock Card 1 */}
                <div className="bg-[#121216] p-4 rounded-xl border border-white/10 flex flex-col justify-between">
                  <div>
                    <div className="flex items-center justify-between mb-3">
                      <span className="text-xs font-semibold text-slate-400">ROP Screening Pipeline</span>
                      <span className="text-[10px] px-2 py-0.5 rounded bg-emerald-500/20 text-emerald-400 font-mono">Stage 2 • Zone I</span>
                    </div>
                    <div className="h-28 rounded-lg bg-slate-900 overflow-hidden relative border border-white/5 flex items-center justify-center">
                      <img
                        src="/static/assets/placeholder-retina.webp"
                        alt="Retinal Fundus Scan"
                        className="w-full h-full object-cover opacity-80"
                        onError={(e) => {
                          (e.target as HTMLElement).style.display = 'none';
                        }}
                      />
                      <div className="absolute inset-0 bg-gradient-to-t from-slate-950 via-transparent to-transparent" />
                      <div className="absolute bottom-2 left-2 right-2 flex items-center justify-between text-[11px] font-mono text-cyan-300 bg-black/60 px-2 py-1 rounded backdrop-blur">
                        <span>EfficientNet-B4</span>
                        <span>Prob: 99.4%</span>
                      </div>
                    </div>
                  </div>
                  <div className="mt-3 text-xs text-slate-400">
                    ICROP 3rd Ed. Advisory: <span className="text-white font-medium">Laser Photocoagulation Recommended</span>
                  </div>
                </div>

                {/* Mock Card 2 */}
                <div className="bg-[#121216] p-4 rounded-xl border border-white/10 flex flex-col justify-between">
                  <div>
                    <div className="flex items-center justify-between mb-3">
                      <span className="text-xs font-semibold text-slate-400">Keratoconus Topography</span>
                      <span className="text-[10px] px-2 py-0.5 rounded bg-purple-500/20 text-purple-300 font-mono">EyeNet Siamese</span>
                    </div>
                    <div className="h-28 rounded-lg bg-slate-900 overflow-hidden relative border border-white/5 flex items-center justify-center">
                      <img
                        src="/static/assets/placeholder-topography.webp"
                        alt="Corneal Topography Heatmap"
                        className="w-full h-full object-cover opacity-80"
                        onError={(e) => {
                          (e.target as HTMLElement).style.display = 'none';
                        }}
                      />
                      <div className="absolute inset-0 bg-gradient-to-tr from-purple-950/40 via-transparent to-cyan-950/40" />
                      <div className="absolute bottom-2 left-2 text-[11px] font-mono text-purple-300 bg-black/60 px-2 py-1 rounded backdrop-blur">
                        <span>Symmetry: SfRS</span>
                      </div>
                    </div>
                  </div>
                  <div className="mt-3 text-xs text-slate-400">
                    Pairwise OD/OS Evaluation: <span className="text-white font-medium">Favorable Refractive Status</span>
                  </div>
                </div>

                {/* Mock Card 3 */}
                <div className="bg-[#121216] p-4 rounded-xl border border-white/10 flex flex-col justify-between">
                  <div>
                    <div className="flex items-center justify-between mb-3">
                      <span className="text-xs font-semibold text-slate-400">Deep Research v3 Dossier</span>
                      <span className="text-[10px] px-2 py-0.5 rounded bg-cyan-500/20 text-cyan-300 font-mono">10-Agent RAG</span>
                    </div>
                    <div className="space-y-2 text-xs font-mono text-slate-300">
                      <div className="flex items-center gap-2 text-emerald-400">
                        <CheckCircle2 className="w-3.5 h-3.5" />
                        <span>A0 Packet: BP/BMI Computed</span>
                      </div>
                      <div className="flex items-center gap-2 text-cyan-400">
                        <Search className="w-3.5 h-3.5" />
                        <span>A4 Multi-Hop: PubMed PubMed/EPMC</span>
                      </div>
                      <div className="flex items-center gap-2 text-purple-400">
                        <ShieldCheck className="w-3.5 h-3.5" />
                        <span>A6 Skeptic: 3/3 Lenses Passed</span>
                      </div>
                    </div>
                  </div>
                  <div className="mt-3 p-2 rounded bg-amber-500/10 border border-amber-500/30 text-[11px] text-amber-300 font-semibold flex items-center justify-between">
                    <span>Work Fitness Verdict</span>
                    <span className="font-mono">Fit w/ Conditions</span>
                  </div>
                </div>
              </div>

              {/* Floating Glass Badges Surrounding Dashboard */}
              <div className="absolute -bottom-5 -left-5 hidden lg:flex items-center gap-2 px-4 py-2 rounded-xl glass-panel text-xs text-slate-200 shadow-xl border border-cyan-500/30">
                <Brain className="w-4 h-4 text-cyan-400" />
                <span>Deep Health Research v3</span>
              </div>
              <div className="absolute -top-5 -right-5 hidden lg:flex items-center gap-2 px-4 py-2 rounded-xl glass-panel text-xs text-slate-200 shadow-xl border border-purple-500/30">
                <Stethoscope className="w-4 h-4 text-purple-400" />
                <span>Bilingual AI Marketplace</span>
              </div>
            </div>
          </motion.div>
        </div>
      </section>

      {/* ========================================================================= */}
      {/* SECTION 2: OPHTHALMIC AI DIAGNOSTICS (BENTO GRID SHOWCASE) */}
      {/* ========================================================================= */}
      <section id="diagnostics" className="py-24 relative">
        <div className="max-w-7xl mx-auto px-6">
          <div className="flex flex-col items-center text-center mb-16">
            <span className="text-xs font-mono text-cyan-400 uppercase tracking-widest px-3 py-1 rounded-full bg-cyan-500/10 border border-cyan-500/20 mb-3">
              Computer Vision Suite
            </span>
            <h2 className="text-3xl md:text-5xl font-extrabold text-white tracking-tight">
              Ophthalmic AI Diagnostics.
            </h2>
            <p className="text-slate-400 text-base md:text-lg max-w-2xl mt-4">
              Multi-stage segmentation, plus-disease grading, and pairwise corneal topography powered by specialized neural backbones.
            </p>
          </div>

          {/* Bento Grid */}
          <div className="grid grid-cols-1 md:grid-cols-12 gap-6">
            {/* Bento Item 1: ROP Screening (8 Cols) */}
            <div className="md:col-span-8 glass-panel rounded-2xl p-6 md:p-8 relative group hover:border-cyan-500/40 transition-all duration-300">
              <div className="flex items-center justify-between mb-6">
                <div>
                  <span className="text-xs font-mono text-cyan-400 uppercase tracking-wider">Module: single_rop</span>
                  <h3 className="text-xl font-bold text-white mt-1">Retinopathy of Prematurity (ROP) 4-Stage Pipeline</h3>
                </div>
                <span className="px-3 py-1 rounded-full text-xs font-mono bg-cyan-500/10 text-cyan-300 border border-cyan-500/30">
                  ICROP 3rd Edition
                </span>
              </div>

              <div className="grid grid-cols-1 sm:grid-cols-2 gap-6 items-center">
                {/* Simulated Retina Scan Image Container */}
                <div className="relative aspect-square rounded-xl bg-slate-950 overflow-hidden border border-white/10 group-hover:border-cyan-500/30 transition-colors">
                  <img
                    src="/static/assets/placeholder-retina.webp"
                    alt="Retinal Fundus Image"
                    className="w-full h-full object-cover opacity-75"
                    onError={(e) => {
                      (e.target as HTMLElement).style.display = 'none';
                    }}
                  />
                  {/* Laser Scan Overlay Effect */}
                  <div className="absolute inset-x-0 h-1 bg-gradient-to-r from-transparent via-cyan-400 to-transparent shadow-[0_0_15px_#00e5ff] animate-scan" />

                  <div className="absolute top-3 left-3 px-2.5 py-1 rounded bg-black/70 backdrop-blur text-[10px] font-mono text-cyan-300 border border-cyan-500/30">
                    Vessel Segmentation: U-Net++
                  </div>
                  <div className="absolute bottom-3 right-3 px-2.5 py-1 rounded bg-black/70 backdrop-blur text-[10px] font-mono text-emerald-300 border border-emerald-500/30">
                    Plus Disease: EfficientNet-B4
                  </div>
                </div>

                {/* Diagnostic Output List */}
                <div className="space-y-4">
                  <div className="p-3.5 rounded-xl bg-[#121218] border border-white/10">
                    <span className="text-xs text-slate-400 block mb-1">Stage 1: Anatomical Segmentation</span>
                    <span className="text-sm font-semibold text-white">Optic Disk & Macula Identified</span>
                  </div>
                  <div className="p-3.5 rounded-xl bg-[#121218] border border-white/10">
                    <span className="text-xs text-slate-400 block mb-1">Stage 2: Plus Disease Classification</span>
                    <span className="text-sm font-semibold text-cyan-300">Pre-Plus Disease (Prob: 0.942)</span>
                  </div>
                  <div className="p-3.5 rounded-xl bg-[#121218] border border-white/10">
                    <span className="text-xs text-slate-400 block mb-1">Stage 3 & 4: Staging & Zonal Mapping</span>
                    <span className="text-sm font-semibold text-purple-300">Stage 2 • Zone II</span>
                  </div>
                  <div className="p-3.5 rounded-xl bg-gradient-to-r from-cyan-950/40 to-blue-950/40 border border-cyan-500/30">
                    <span className="text-xs text-cyan-400 font-semibold block mb-0.5">Clinical Guidance</span>
                    <span className="text-xs text-slate-300">Re-examine in 48 hours; observe vessel tortuosity.</span>
                  </div>
                </div>
              </div>
            </div>

            {/* Bento Item 2: KC Topography (4 Cols) */}
            <div className="md:col-span-4 glass-panel rounded-2xl p-6 md:p-8 flex flex-col justify-between group hover:border-purple-500/40 transition-all duration-300">
              <div>
                <div className="flex items-center justify-between mb-4">
                  <span className="text-xs font-mono text-purple-400 uppercase tracking-wider">Module: double_rop</span>
                  <span className="px-2.5 py-0.5 rounded text-[10px] font-mono bg-purple-500/10 text-purple-300 border border-purple-500/30">EyeNet CBAM</span>
                </div>
                <h3 className="text-xl font-bold text-white mb-2">Keratoconus (KC) Topography</h3>
                <p className="text-xs text-slate-400 leading-relaxed mb-6">
                  Pairwise OD/OS corneal topography feature extraction using Siamese transformers & multi-head self-attention.
                </p>

                {/* 3D Topography Heatmap Container */}
                <motion.div
                  whileHover={{ rotateY: 12, rotateX: -6 }}
                  className="relative aspect-video rounded-xl bg-slate-950 overflow-hidden border border-white/10 group-hover:border-purple-500/30 transition-transform duration-500 flex items-center justify-center"
                >
                  <img
                    src="/static/assets/placeholder-topography.webp"
                    alt="Corneal Heatmap"
                    className="w-full h-full object-cover opacity-80"
                    onError={(e) => {
                      (e.target as HTMLElement).style.display = 'none';
                    }}
                  />
                  <div className="absolute inset-0 bg-gradient-to-t from-slate-950 via-transparent to-transparent" />
                  <div className="absolute top-2 left-2 px-2 py-0.5 rounded bg-black/70 text-[10px] font-mono text-purple-300">
                    Pairwise Topography Map
                  </div>
                </motion.div>
              </div>

              <div className="mt-6 pt-4 border-t border-white/10 space-y-2 text-xs">
                <div className="flex justify-between text-slate-300">
                  <span>OD/OS Severity:</span>
                  <span className="font-semibold text-purple-300">Normal (5-Class)</span>
                </div>
                <div className="flex justify-between text-slate-300">
                  <span>Z-Class Symmetry:</span>
                  <span className="font-semibold text-emerald-400">SfRS (Symmetrical)</span>
                </div>
              </div>
            </div>

            {/* Bento Item 3: Non-Invasive Anemia (12 Cols) */}
            <div className="md:col-span-12 glass-panel rounded-2xl p-6 md:p-8 flex flex-col md:flex-row items-center justify-between gap-6 group hover:border-cyan-500/40 transition-all duration-300">
              <div className="flex items-center gap-5">
                <div className="w-14 h-14 rounded-xl bg-cyan-500/10 border border-cyan-500/30 flex items-center justify-center shrink-0">
                  <Activity className="w-7 h-7 text-cyan-400" />
                </div>
                <div>
                  <span className="text-xs font-mono text-cyan-400 uppercase tracking-wider">Module: test_analysis</span>
                  <h3 className="text-lg font-bold text-white mt-0.5">Non-Invasive Conjunctiva Anemia Screening</h3>
                  <p className="text-xs text-slate-400 mt-1 max-w-xl">
                    Smartphone palpebral conjunctiva photo segmentation paired with Vision-Language Models (VLM) for instant pallor & anemia risk evaluation.
                  </p>
                </div>
              </div>

              <div className="flex items-center gap-4 w-full md:w-auto">
                <div className="flex-1 md:flex-none px-4 py-2.5 rounded-xl bg-[#121218] border border-white/10 text-center">
                  <span className="text-[10px] text-slate-500 uppercase block font-mono">Phase-2 Crop</span>
                  <span className="text-xs font-semibold text-cyan-300">Palpebral Masked</span>
                </div>
                <div className="flex-1 md:flex-none px-4 py-2.5 rounded-xl bg-emerald-500/10 border border-emerald-500/30 text-center">
                  <span className="text-[10px] text-emerald-400/80 uppercase block font-mono">VLM Verdict</span>
                  <span className="text-xs font-bold text-emerald-400">Anemia Negative (98.4%)</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* ========================================================================= */}
      {/* SECTION 3: DEEP HEALTH RESEARCH V3 (THE 10-AGENT CLINICAL AUTHOR) */}
      {/* ========================================================================= */}
      <section id="deep-research" className="py-24 relative bg-slate-950/40 border-y border-white/[0.06]">
        <div className="max-w-7xl mx-auto px-6">
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-12 items-center mb-16">
            {/* Left Column: Explanation */}
            <div className="lg:col-span-5 space-y-6">
              <span className="text-xs font-mono text-purple-400 uppercase tracking-widest px-3 py-1 rounded-full bg-purple-500/10 border border-purple-500/20">
                Deep Research v3 Core
              </span>
              <h2 className="text-3xl md:text-5xl font-extrabold text-white tracking-tight leading-tight">
                The 10-Agent Autonomous Pipeline.
              </h2>
              <p className="text-slate-400 text-sm md:text-base leading-relaxed">
                Replaces gimmicky health scores with a rigorous, Persian clinical research dossier. Deterministic Python computation (`packet.py`) paired with multi-hop biomedical research over NCBI PubMed and Europe PMC APIs.
              </p>

              <div className="space-y-3 pt-2">
                <div className="flex items-start gap-3">
                  <div className="w-5 h-5 rounded-full bg-purple-500/20 text-purple-400 flex items-center justify-center text-xs font-bold shrink-0 mt-0.5">1</div>
                  <span className="text-xs text-slate-300">
                    <strong className="text-white">Sentence Evidence Grading:</strong> Every factual claim carries superscript chips <code className="text-cyan-400 font-mono">[1]</code> linked to graded evidence badges (*Guideline*, *Study*, *Patient Data*).
                  </span>
                </div>
                <div className="flex items-start gap-3">
                  <div className="w-5 h-5 rounded-full bg-purple-500/20 text-purple-400 flex items-center justify-center text-xs font-bold shrink-0 mt-0.5">2</div>
                  <span className="text-xs text-slate-300">
                    <strong className="text-white">Adversarial Skeptic Panel A6:</strong> 3 independent AI skeptic lenses audit triaged problems for patient data consistency before publication.
                  </span>
                </div>
                <div className="flex items-start gap-3">
                  <div className="w-5 h-5 rounded-full bg-purple-500/20 text-purple-400 flex items-center justify-center text-xs font-bold shrink-0 mt-0.5">3</div>
                  <span className="text-xs text-slate-300">
                    <strong className="text-white">Interactive 3D Co-Pilot:</strong> Pure CSS 3D Minecraft Steve assistant integrated into the employee profile page sidebar.
                  </span>
                </div>
              </div>
            </div>

            {/* Right Column: Simulated Typing IDE / Terminal */}
            <div className="lg:col-span-7 glass-panel-glow rounded-2xl p-6 border border-purple-500/20 relative shadow-2xl">
              <div className="flex items-center justify-between pb-4 mb-4 border-b border-white/10">
                <div className="flex items-center gap-2">
                  <div className="w-3 h-3 rounded-full bg-rose-500" />
                  <div className="w-3 h-3 rounded-full bg-amber-500" />
                  <div className="w-3 h-3 rounded-full bg-emerald-500" />
                  <span className="ml-2 text-xs font-mono text-purple-300">Deep_Research_v3_Agents_Loop.py</span>
                </div>
                <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-purple-500/20 text-purple-300">
                  PubMed / EPMC Live API
                </span>
              </div>

              {/* Typing Output Screen */}
              <div className="h-64 overflow-y-auto font-mono text-xs text-slate-300 leading-relaxed space-y-2 bg-[#09090d] p-4 rounded-xl border border-white/5">
                <p className="whitespace-pre-wrap text-cyan-300">
                  {typingText.slice(0, typingIndex)}
                  <span className="animate-pulse text-purple-400">▌</span>
                </p>
              </div>

              {/* Work Fitness Banners Simulation */}
              <div className="mt-6 grid grid-cols-1 sm:grid-cols-2 gap-4">
                <div className="p-4 rounded-xl bg-gradient-to-r from-amber-950/40 to-amber-900/20 border border-amber-500/40 flex items-center gap-3">
                  <AlertTriangle className="w-6 h-6 text-amber-400 shrink-0" />
                  <div>
                    <span className="text-[10px] font-mono text-amber-400 uppercase block">Work Fitness Status</span>
                    <span className="text-sm font-bold text-amber-200">Fit with Conditions (مشروط)</span>
                  </div>
                </div>

                <div className="p-4 rounded-xl bg-gradient-to-r from-rose-950/40 to-rose-900/20 border border-rose-500/40 flex items-center gap-3">
                  <XCircle className="w-6 h-6 text-rose-400 shrink-0" />
                  <div>
                    <span className="text-[10px] font-mono text-rose-400 uppercase block">Unfit Alert Trigger</span>
                    <span className="text-sm font-bold text-rose-200">Disqualified: Noise Exposure</span>
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Grid Displaying All 10 Autonomous Agents (A0 through A9) */}
          <div className="pt-8">
            <h3 className="text-center text-xl font-bold text-white mb-8">
              Complete Breakdown of All 10 Autonomous Agents
            </h3>
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-4">
              {agentsList.map((agent) => (
                <div
                  key={agent.id}
                  className="p-4 rounded-xl bg-[#121218] border border-white/10 hover:border-purple-500/40 transition-colors flex flex-col justify-between"
                >
                  <div>
                    <span className="px-2 py-0.5 rounded text-[10px] font-mono bg-purple-500/20 text-purple-300 font-bold border border-purple-500/30 inline-block mb-2">
                      {agent.id}
                    </span>
                    <h4 className="text-xs font-bold text-white mb-1">{agent.name}</h4>
                    <p className="text-[11px] text-slate-400 leading-snug">{agent.role}</p>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </section>

      {/* ========================================================================= */}
      {/* SECTION 4: VIRTUAL DOCTOR MARKETPLACE & STUDIO */}
      {/* ========================================================================= */}
      <section id="marketplace" className="py-24 relative overflow-hidden">
        <div className="absolute top-1/2 left-0 w-[500px] h-[500px] bg-cyan-500/10 blur-[150px] pointer-events-none" />

        <div className="max-w-7xl mx-auto px-6 relative z-10">
          <div className="flex flex-col items-center text-center mb-16">
            <span className="text-xs font-mono text-cyan-400 uppercase tracking-widest px-3 py-1 rounded-full bg-cyan-500/10 border border-cyan-500/20 mb-3">
              Doctor Marketplace & Studio
            </span>
            <h2 className="text-3xl md:text-5xl font-extrabold text-white tracking-tight">
              Bilingual AI Doctor Specialist Studio.
            </h2>
            <p className="text-slate-400 text-base md:text-lg max-w-2xl mt-4">
              Explore specialized AI medical doctors or build your own with custom system prompts, vector knowledge bases, and WhatsApp reminders.
            </p>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-center">
            {/* Floating Chat Interface (7 Cols) */}
            <div className="lg:col-span-7 glass-panel rounded-2xl p-6 border border-cyan-500/30 shadow-2xl relative">
              <div className="flex items-center justify-between pb-4 border-b border-white/10 mb-4">
                <div className="flex items-center gap-3">
                  <div className="w-10 h-10 rounded-full bg-cyan-500/20 border border-cyan-400 flex items-center justify-center text-cyan-300 font-bold text-sm">
                    DR
                  </div>
                  <div>
                    <h4 className="text-sm font-bold text-white">Dr. Arash (Cardiology & Renal)</h4>
                    <span className="text-[10px] font-mono text-emerald-400">Persona: Kind & Patient • Active</span>
                  </div>
                </div>
                <span className="px-2.5 py-1 rounded-full text-[10px] font-mono bg-cyan-500/10 text-cyan-300 border border-cyan-500/30">
                  Whisper Voice STT
                </span>
              </div>

              {/* Chat Thread Simulation */}
              <div className="space-y-4 text-xs font-sans mb-6">
                {/* User Message */}
                <div className="flex justify-end">
                  <div className="bg-cyan-600/30 border border-cyan-500/30 rounded-2xl rounded-tr-sm p-3.5 text-slate-100 max-w-md">
                    <div className="flex items-center gap-2 text-[10px] text-cyan-300 mb-1">
                      <Mic className="w-3 h-3" />
                      <span>Audio Note Transcribed via Whisper</span>
                    </div>
                    <p>"Hello Doctor, here are my CBC lab results and eGFR levels..."</p>
                  </div>
                </div>

                {/* AI Doctor Response */}
                <div className="flex justify-start">
                  <div className="bg-[#121218] border border-white/10 rounded-2xl rounded-tl-sm p-3.5 text-slate-200 max-w-md space-y-2">
                    <p>
                      "Analyzing your CBC panel against my specialized Cardiology & Nephrology Knowledge Base. Calculating eGFR: <strong className="text-cyan-300">74 mL/min/1.73m²</strong>."
                    </p>
                    <div className="p-2 rounded bg-cyan-500/10 text-[11px] text-cyan-300 border border-cyan-500/20">
                      Medical Calculator Tool: eGFR & Creatinine Clearance Executed
                    </div>
                  </div>
                </div>

                {/* WhatsApp Pop-up Banner */}
                <motion.div
                  initial={{ scale: 0.9, opacity: 0 }}
                  animate={{ scale: 1, opacity: 1 }}
                  transition={{ duration: 0.5, delay: 0.6 }}
                  className="p-3 rounded-xl bg-emerald-950/50 border border-emerald-500/40 text-emerald-300 text-xs flex items-center justify-between"
                >
                  <div className="flex items-center gap-2">
                    <MessageSquare className="w-4 h-4 text-emerald-400" />
                    <span>WhatsApp Follow-Up & Medication Reminder Scheduled (WAHA Gateway)</span>
                  </div>
                  <span className="text-[10px] font-mono bg-emerald-500/20 px-2 py-0.5 rounded">SENT</span>
                </motion.div>
              </div>

              {/* Chat Input Bar */}
              <div className="flex items-center gap-3 pt-3 border-t border-white/10">
                <input
                  type="text"
                  readOnly
                  value="Ask Dr. Arash about medication interactions or lab results..."
                  className="flex-1 bg-[#0d0d12] border border-white/10 rounded-xl px-4 py-2.5 text-xs text-slate-400 outline-none"
                />
                <button className="p-2.5 rounded-xl bg-cyan-500 text-black font-bold flex items-center justify-center">
                  <Send className="w-4 h-4" />
                </button>
              </div>
            </div>

            {/* Admin Prompt Builder & Features (5 Cols) */}
            <div className="lg:col-span-5 space-y-6">
              <div className="glass-panel p-6 rounded-2xl border border-white/10">
                <div className="flex items-center gap-3 mb-3">
                  <Cpu className="w-5 h-5 text-cyan-400" />
                  <h4 className="text-base font-bold text-white">Admin Doctor Studio (/market/studio/)</h4>
                </div>
                <p className="text-xs text-slate-400 leading-relaxed mb-4">
                  Construct custom virtual doctors, assign system prompts, generate realistic avatars using DALL-E, and manage private PDF/TXT vector indexes.
                </p>
                <div className="flex flex-wrap gap-2 text-[11px] font-mono">
                  <span className="px-2.5 py-1 rounded bg-white/5 border border-white/10 text-slate-300">BMI / BSA Calc</span>
                  <span className="px-2.5 py-1 rounded bg-white/5 border border-white/10 text-slate-300">FDA Drug Lookup</span>
                  <span className="px-2.5 py-1 rounded bg-white/5 border border-white/10 text-slate-300">MELD & CHA₂DS₂ Score</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* ========================================================================= */}
      {/* SECTION 5: CORPORATE GOVERNANCE & USAC CORE */}
      {/* ========================================================================= */}
      <section id="governance" className="py-24 relative bg-slate-950/60 border-t border-white/[0.06]">
        <div className="max-w-7xl mx-auto px-6">
          <div className="flex flex-col items-center text-center mb-16">
            <span className="text-xs font-mono text-purple-400 uppercase tracking-widest px-3 py-1 rounded-full bg-purple-500/10 border border-purple-500/20 mb-3">
              Identity & Multi-Tenancy
            </span>
            <h2 className="text-3xl md:text-5xl font-extrabold text-white tracking-tight">
              Corporate Governance & Access Control.
            </h2>
            <p className="text-slate-400 text-base md:text-lg max-w-2xl mt-4">
              Role-Based Access Control (RBAC) enforcing strict multi-tenant boundaries and audit tracking.
            </p>
          </div>

          {/* 3 Role Panels Connected by Glowing Data Flow Line */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-8 relative">
            {/* Connecting Glow Line behind cards */}
            <div className="hidden md:block absolute top-1/2 left-0 right-0 h-0.5 bg-gradient-to-r from-cyan-500 via-purple-500 to-emerald-500 -translate-y-1/2 z-0 opacity-40" />

            {/* Panel 1: Manager */}
            <div className="glass-panel p-6 rounded-2xl border border-cyan-500/30 relative z-10 hover:scale-[1.02] transition-transform">
              <div className="w-12 h-12 rounded-xl bg-cyan-500/10 border border-cyan-500/30 flex items-center justify-center mb-4">
                <Building2 className="w-6 h-6 text-cyan-400" />
              </div>
              <span className="text-[10px] font-mono text-cyan-400 uppercase block mb-1">Role: Corporate Manager</span>
              <h3 className="text-lg font-bold text-white mb-2">Company Management</h3>
              <p className="text-xs text-slate-400 leading-relaxed mb-4">
                Register corporate entities, issue National Code invitations, track employee health compliance, and inspect work fitness verdicts.
              </p>
              <div className="text-[11px] font-mono text-cyan-300 bg-cyan-950/40 p-2.5 rounded border border-cyan-500/20">
                URL: /managing/
              </div>
            </div>

            {/* Panel 2: Clinician */}
            <div className="glass-panel p-6 rounded-2xl border border-purple-500/30 relative z-10 hover:scale-[1.02] transition-transform">
              <div className="w-12 h-12 rounded-xl bg-purple-500/10 border border-purple-500/30 flex items-center justify-center mb-4">
                <Stethoscope className="w-6 h-6 text-purple-400" />
              </div>
              <span className="text-[10px] font-mono text-purple-400 uppercase block mb-1">Role: Examining Doctor</span>
              <h3 className="text-lg font-bold text-white mb-2">Clinician Audit Suite</h3>
              <p className="text-xs text-slate-400 leading-relaxed mb-4">
                Review fundus and topography scans, override machine labels with expert ground truth, record clinical notes, and manage referrals.
              </p>
              <div className="text-[11px] font-mono text-purple-300 bg-purple-950/40 p-2.5 rounded border border-purple-500/20">
                URL: /health/doctor-dashboard/
              </div>
            </div>

            {/* Panel 3: Employee */}
            <div className="glass-panel p-6 rounded-2xl border border-emerald-500/30 relative z-10 hover:scale-[1.02] transition-transform">
              <div className="w-12 h-12 rounded-xl bg-emerald-500/10 border border-emerald-500/30 flex items-center justify-center mb-4">
                <UserCheck className="w-6 h-6 text-emerald-400" />
              </div>
              <span className="text-[10px] font-mono text-emerald-400 uppercase block mb-1">Role: Employee / Patient</span>
              <h3 className="text-lg font-bold text-white mb-2">View-Locked Health Profile</h3>
              <p className="text-xs text-slate-400 leading-relaxed mb-4">
                Access read-only diagnostic reports, submit conjunctiva scans, consult health AI chatbots, and play retro Pacman during report compilation.
              </p>
              <div className="text-[11px] font-mono text-emerald-300 bg-emerald-950/40 p-2.5 rounded border border-emerald-500/20">
                URL: /health/profile/detail/
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* ========================================================================= */}
      {/* SECTION 6: SYSTEM ARCHITECTURE MAP (THE ENGINE ROOM) */}
      {/* ========================================================================= */}
      <section id="architecture" className="py-24 relative">
        <div className="max-w-7xl mx-auto px-6">
          <div className="flex flex-col items-center text-center mb-16">
            <span className="text-xs font-mono text-cyan-400 uppercase tracking-widest px-3 py-1 rounded-full bg-cyan-500/10 border border-cyan-500/20 mb-3">
              Distributed Monorepo
            </span>
            <h2 className="text-3xl md:text-5xl font-extrabold text-white tracking-tight">
              Engine Room Architecture.
            </h2>
            <p className="text-slate-400 text-base md:text-lg max-w-2xl mt-4">
              Dual-framework infrastructure combining monolithic Django governance with high-throughput FastAPI RAG microservices.
            </p>
          </div>

          {/* Node Diagram */}
          <div className="glass-panel p-8 md:p-12 rounded-2xl border border-white/10 relative overflow-hidden">
            <div className="grid grid-cols-1 md:grid-cols-3 gap-8 items-center text-center relative z-10">
              {/* Node 1 */}
              <div className="p-6 rounded-xl bg-[#101016] border border-cyan-500/30 flex flex-col items-center space-y-3">
                <Database className="w-10 h-10 text-cyan-400" />
                <h4 className="font-bold text-white text-base">Django Monolith (Port 8000)</h4>
                <p className="text-xs text-slate-400">PostgreSQL ORM • Auth (RBAC) • Audit Trails • Template Rendering</p>
              </div>

              {/* Node 2 */}
              <div className="p-6 rounded-xl bg-[#101016] border border-purple-500/30 flex flex-col items-center space-y-3">
                <Zap className="w-10 h-10 text-purple-400 animate-pulse" />
                <h4 className="font-bold text-white text-base">Redis Broker & Celery Workers</h4>
                <p className="text-xs text-slate-400">Async CV Inference • Deep Research Synthesis • Task Queue</p>
              </div>

              {/* Node 3 */}
              <div className="p-6 rounded-xl bg-[#101016] border border-blue-500/30 flex flex-col items-center space-y-3">
                <Cpu className="w-10 h-10 text-blue-400" />
                <h4 className="font-bold text-white text-base">FastAPI RAG Service (Port 8001)</h4>
                <p className="text-xs text-slate-400">BM25 + FAISS Hybrid RAG • Cross-Encoder Reranker • Web Crawler</p>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* ========================================================================= */}
      {/* SECTION 7: FOOTER & FINAL CTA */}
      {/* ========================================================================= */}
      <footer className="relative pt-24 pb-12 border-t border-white/10 bg-black">
        <div className="max-w-7xl mx-auto px-6">
          {/* Final CTA Box */}
          <div className="glass-panel-glow p-10 md:p-16 rounded-3xl border border-cyan-500/30 text-center relative overflow-hidden mb-20">
            <div className="absolute inset-0 bg-gradient-to-r from-cyan-500/10 via-purple-600/10 to-blue-600/10 pointer-events-none" />
            <h2 className="text-3xl md:text-5xl font-extrabold text-white tracking-tight mb-4 relative z-10">
              Ready to Modernize Your Clinical Workflow?
            </h2>
            <p className="text-slate-400 text-sm md:text-base max-w-xl mx-auto mb-8 relative z-10">
              Deploy Mediverse AI across your medical enterprise. Automated ROP/KC screening, deep research dossiers, and AI specialist doctors.
            </p>
            <a
              href="/login/"
              className="inline-flex items-center gap-3 px-8 py-4 rounded-xl font-bold text-sm text-white bg-gradient-to-r from-cyan-500 via-blue-600 to-purple-600 shadow-xl shadow-cyan-500/30 hover:scale-[1.03] transition-all relative z-10"
            >
              <span>Enter Enterprise Portal Now</span>
              <ArrowRight className="w-4 h-4" />
            </a>
          </div>

          {/* Footer Navigation Links */}
          <div className="flex flex-col md:flex-row items-center justify-between gap-6 pt-8 border-t border-white/10 text-xs text-slate-500">
            <div className="flex items-center gap-3">
              <Brain className="w-5 h-5 text-cyan-400" />
              <span className="font-bold text-slate-300 text-sm">Mediverse AI</span>
              <span>© 2026 Aras AI Platform. All rights reserved.</span>
            </div>

            <div className="flex flex-wrap items-center gap-6">
              <a href="/login/" className="hover:text-cyan-400 transition-colors">Portal Login</a>
              <a href="/history/" className="hover:text-cyan-400 transition-colors">Diagnostic History</a>
              <a href="/market/" className="hover:text-cyan-400 transition-colors">Doctor Marketplace</a>
              <a href="/health/profile/" className="hover:text-cyan-400 transition-colors">Health Profile</a>
              <a href="/dilemma/" className="hover:text-cyan-400 transition-colors">Ethical Dilemmas</a>
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
}
