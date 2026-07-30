# doctors_marketplace/services/agent_runtime.py
"""
Agentic orchestration: a bounded tool-calling loop over the agents a superuser
enabled for a given assistant.

Emits the same streaming event shapes as rag_chat (so the existing frontend works)
plus a live "stage" per agent step and optional "ui" events (e.g. an emergency
banner):

    {"type":"stage", "stage":<agent-key or 'thinking'>, "label":..., "icon":...}
    {"type":"ui",    "banner": {...}}
    {"type":"token", "text":...}
    {"type":"done",  "sources":[{id,type,title,url,domain}, ...]}
    {"type":"error", "message":...}

Design for correctness + latency:
  * Greetings/farewells/thanks skip the whole loop (one lean streamed reply).
  * Safety/`pre_pass` agents (e.g. red_flag_check) run FIRST, deterministically.
  * The model then calls tools (≤ MAX_ROUNDS rounds) to gather facts; ordering
    (identify drugs → check interactions, gather sources → compose) is guided by
    the system prompt and each agent's run_order.
  * A single final streamed pass composes the answer with numbered [n] citations
    built from every source-producing tool that ran.
"""
from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, Iterator, List

from .llm import LLMClient
from . import rag_chat
from .. import agents as agents_pkg
from ..agents.base import AgentContext

log = logging.getLogger("doctors_marketplace.services.agent_runtime")

MAX_ROUNDS = 4

_MATH_RULE = ("Write any math, numbers or formulas in plain text using Unicode "
              "(e.g. \"BMI = 26.1 kg/m²\", \"eGFR ≈ 69\"). Do NOT use LaTeX, dollar signs, "
              "or backslash commands like \\frac, \\text or \\[ \\].")


def _number_sources(source_dicts: List[Dict]) -> tuple[List[Dict], str]:
    """De-duplicate + number source contributions; build the [SOURCES] context."""
    ui_sources: List[Dict] = []
    seen: Dict[str, int] = {}
    lines: List[str] = []
    for s in source_dicts:
        key = (s.get("url") or "") + "|" + (s.get("title") or "")
        if key in seen:
            continue
        sid = len(ui_sources) + 1
        seen[key] = sid
        ui_sources.append({"id": sid, "type": s.get("type", "web"),
                           "title": s.get("title") or "Source",
                           "url": s.get("url"), "domain": s.get("domain")})
        kind = "web" if s.get("type") == "web" else "document"
        loc = f" — {s.get('url')}" if s.get("url") else ""
        lines.append(f"[{sid}] ({kind}: {s.get('title')}{loc})\n{(s.get('page_content') or '')[:1200]}")
    context = "\n---\n".join(lines)
    if len(context) > 14000:
        context = context[:14000] + "\n[...truncated...]"
    return ui_sources, context


def _stage(agent, fallback_label="Working"):
    return {"type": "stage", "stage": agent.key,
            "label": agent.stage_label or fallback_label, "icon": agent.icon}


def _tool_result_msg(tool_call_id, name, content):
    return {"role": "tool", "tool_call_id": tool_call_id, "name": name, "content": content}


def run_agentic_answer(*, doctor, session, user, query, system_text, history_msgs,
                       user_content, enabled_keys, now_iso="") -> Iterator[Dict[str, Any]]:
    """Run the enabled-agents pipeline for one turn, yielding live events."""
    enabled = agents_pkg.get_agents(enabled_keys)   # sorted by run_order
    if not enabled:
        # No agents configured → legacy KB+web pipeline (fully backward compatible).
        yield from rag_chat.stream_doctor_answer(
            doctor=doctor, query=query, system_text=system_text,
            history_msgs=history_msgs, user_content=user_content)
        return

    llm = LLMClient()
    t_start = time.perf_counter()
    intent = rag_chat.classify_intent(query)
    ctx = AgentContext(doctor=doctor, session=session, user=user,
                       history=history_msgs, query=query, now_iso=now_iso)
    pre_agents = [a for a in enabled if a.pre_pass]
    tool_agents = [a for a in enabled if not a.pre_pass]
    log.info("─" * 70)
    log.info("AGENT | doctor=%s intent=%s enabled=%s q=%r",
             getattr(doctor, "slug", doctor), intent,
             [a.key for a in enabled], (query or "")[:120])

    yield {"type": "stage", "stage": "thinking", "label": "Thinking", "icon": "thinking"}

    all_sources: List[Dict] = []
    safety_notes: List[str] = []

    # ---- 1) deterministic pre-pass (safety) ----
    for agent in pre_agents:
        yield _stage(agent)
        try:
            res = agent.run({}, ctx)
        except Exception as e:  # noqa: BLE001
            log.warning("AGENT | pre-pass %s failed: %s", agent.key, e)
            continue
        if res.ui:
            yield {"type": "ui", **res.ui}
        if res.content and res.ok and "No emergency" not in res.content:
            safety_notes.append(res.content)
        log.info("AGENT | pre %s -> %s", agent.key, res.display or res.content[:60])

    # ---- 2) tool-calling gather loop (skipped for small talk) ----
    convo: List[Dict[str, Any]] = []
    tool_summaries: List[str] = []
    if intent == "question" and tool_agents:
        tool_names = ", ".join(a.key for a in tool_agents)
        guidance = (
            f"You are a medical assistant. Current time: {now_iso or 'unknown'}.\n"
            f"You have these tools: {tool_names}. Use them to gather everything you need, then you "
            "will compose the final answer.\n"
            "STRICT RULES:\n"
            "• For ANY numeric clinical calculation (BMI, BSA, eGFR, creatinine clearance, "
            "CHA₂DS₂-VASc, MELD, anion gap, drug dose, IV rate) you MUST call medical_calculator — "
            "never do the arithmetic yourself.\n"
            "• For a specific lab value, call lab_interpreter. For a drug's facts, call drug_lookup. "
            "For drug safety together, call drug_interactions; against the patient, check_contraindications.\n"
            "• For facts you are unsure of or that need sources, call search_knowledge_base / search_web / "
            "search_pubmed before answering.\n"
            "• Ordering: run safety/triage early; identify a drug before checking its interactions or "
            "contraindications; gather sources before composing.\n"
            "• If a tool reports it needs more info (e.g. a missing number or phone), ask the user in your "
            "final answer instead of guessing.\n"
            "Never fabricate values or citations. Do NOT write the final answer during this phase — only call tools."
        )
        if safety_notes:
            guidance += "\n\nSAFETY: " + " ".join(safety_notes)
        convo = [{"role": "system", "content": guidance}]
        convo += history_msgs
        convo.append({"role": "user", "content": user_content})
        tools = [a.tool_schema() for a in tool_agents]
        by_key = {a.key: a for a in tool_agents}

        for round_i in range(MAX_ROUNDS):
            try:
                msg = llm.complete(convo, tools=tools, tool_choice="auto")
            except Exception as e:  # noqa: BLE001
                log.warning("AGENT | tool round failed: %s", e)
                break
            tool_calls = getattr(msg, "tool_calls", None)
            if not tool_calls:
                break
            convo.append({
                "role": "assistant", "content": msg.content or "",
                "tool_calls": [{"id": tc.id, "type": "function",
                                "function": {"name": tc.function.name,
                                             "arguments": tc.function.arguments}}
                               for tc in tool_calls],
            })
            for tc in tool_calls:
                agent = by_key.get(tc.function.name)
                if not agent:
                    convo.append(_tool_result_msg(tc.id, tc.function.name, "Unknown tool."))
                    continue
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except Exception:
                    args = {}
                yield _stage(agent)
                t0 = time.perf_counter()
                try:
                    res = agent.run(args, ctx)
                except Exception as e:  # noqa: BLE001
                    log.warning("AGENT | %s raised: %s", agent.key, e)
                    convo.append(_tool_result_msg(tc.id, agent.key, f"Tool error: {e}"))
                    continue
                log.info("AGENT | %s(%s) -> %s (%.2fs)", agent.key,
                         json.dumps(args, ensure_ascii=False)[:80],
                         res.display or "ok", time.perf_counter() - t0)
                if res.sources:
                    all_sources.extend(res.sources)
                if res.ui:
                    yield {"type": "ui", **res.ui}
                if res.display:
                    tool_summaries.append(res.display)
                convo.append(_tool_result_msg(tc.id, agent.key, res.content or "(no output)"))
        yield {"type": "stage", "stage": "thinking", "label": "Composing the answer", "icon": "thinking"}

    # ---- 3) compose the final answer (streamed, with [n] citations) ----
    ui_sources, context_str = _number_sources(all_sources)
    compose_system = system_text or ""
    if intent != "question":
        compose_system += ("\n\nThe user is making small talk (greeting/farewell/thanks). "
                           "Reply warmly and briefly in the user's language; no medical content, no citations.")
    else:
        if context_str:
            compose_system += "\n\n" + rag_chat._CITATION_RULES
        else:
            compose_system += ("\n\nAnswer clearly in the user's language (simple Markdown). "
                               "If you used no external sources, do not fabricate citations.")
    if safety_notes:
        compose_system += "\n\nSAFETY DIRECTIVE: " + " ".join(safety_notes)
    compose_system += "\n\n" + _MATH_RULE

    compose: List[Dict[str, Any]] = [{"role": "system", "content": compose_system}]
    if context_str:
        compose.append({"role": "system", "content": "[SOURCES]:\n" + context_str})
    compose += history_msgs
    compose.append({"role": "user", "content": user_content})
    # Give the composer the facts the tools gathered.
    if convo:
        tool_msgs = [m for m in convo if m.get("role") == "tool"]
        if tool_msgs:
            findings = "\n".join(f"- {m['name']}: {m['content'][:600]}" for m in tool_msgs)
            compose.append({"role": "system",
                            "content": "Facts gathered by your tools this turn:\n" + findings})

    log.info("AGENT | composing (%d source(s), %d tool finding(s))",
             len(ui_sources), len(tool_summaries))
    answer_parts: List[str] = []
    try:
        streamed = False
        for piece in llm.chat_stream(compose):
            streamed = True
            answer_parts.append(piece)
            yield {"type": "token", "text": piece}
        if not streamed:
            text = llm.chat(compose)
            answer_parts.append(text)
            yield {"type": "token", "text": text}
    except Exception:
        log.exception("AGENT | compose failed")
        try:
            text = llm.chat(compose)
            answer_parts.append(text)
            yield {"type": "token", "text": text}
        except Exception:
            yield {"type": "error",
                   "message": "متاسفم—الان به مدل پزشکی دسترسی ندارم. لطفاً کمی بعد دوباره تلاش کنید."}
            return

    answer = "".join(answer_parts).strip()
    log.info("AGENT | done in %.2fs (%d chars, %d source(s))",
             time.perf_counter() - t_start, len(answer), len(ui_sources))
    yield {"type": "done", "sources": ui_sources, "answer": answer}
