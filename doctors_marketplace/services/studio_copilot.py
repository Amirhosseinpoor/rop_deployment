# doctors_marketplace/services/studio_copilot.py
"""
Studio Copilot — an assistant that helps the website admin build a new
marketplace assistant (a "Doctor").

It has a real conversation with the admin, understands the whole
assistant-creation form (identity, specialty, persona, bilingual copy and the
all-important system prompt) and is fully aware of every agent/tool that can be
enabled. From the conversation it:

  * writes *high-quality, safety-aware* prompts (these assistants act as doctors
    and counselors, so a weak prompt is dangerous),
  * fills the form fields for the admin,
  * figures out which agents the assistant needs and ticks them,
  * explains any agent in depth, with examples, when asked.

The LLM drives everything through a single tool, ``update_assistant_form``,
whose ``message`` field is what we show in the chat. When the admin only wants
an explanation the model answers with plain text and no tool call.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List

from .llm import LLMClient
from .. import agents as agents_pkg
from ..models import Doctor


# Editable form fields the copilot may write, with a short human note used both
# in the system prompt and to build the tool schema.
_TEXT_FIELDS: Dict[str, str] = {
    "name": "Display name in English (e.g. 'Dr. Sarah Chen').",
    "name_fa": "Display name in Persian (فارسی).",
    "specialization_fa": "Specialty label in Persian (free text).",
    "headline": "One-line English headline for the marketplace card.",
    "headline_fa": "One-line Persian headline (فارسی).",
    "bio": "Short English bio (2–4 sentences) shown on the marketplace card.",
    "bio_fa": "Short Persian bio (فارسی).",
    "tags_fa": "Comma-separated tags (Persian or English), e.g. 'قلب,فشار خون,رژیم غذایی'.",
    "system_prompt": (
        "The assistant's full system prompt — the most important field. Must be "
        "detailed, structured, and safety-aware."
    ),
}


def _specialization_choices() -> List[Dict[str, str]]:
    return [{"value": v, "label": l} for v, l in Doctor.Specialization.choices]


def _persona_choices() -> List[Dict[str, str]]:
    return [{"value": v, "label": l} for v, l in Doctor.Persona.choices]


def _agent_catalog() -> List[Dict[str, Any]]:
    return agents_pkg.catalog()


def _agent_keys() -> List[str]:
    return [a["key"] for a in _agent_catalog()]


def build_tool_schema() -> Dict[str, Any]:
    """The single tool the copilot uses to write into the form."""
    field_props: Dict[str, Any] = {}
    for key, note in _TEXT_FIELDS.items():
        field_props[key] = {"type": "string", "description": note}
    field_props["specialization"] = {
        "type": "string",
        "enum": [c["value"] for c in _specialization_choices()],
        "description": "Specialty (must be one of the allowed values).",
    }
    field_props["persona"] = {
        "type": "string",
        "enum": [c["value"] for c in _persona_choices()],
        "description": "Conversational persona/tone (must be one of the allowed values).",
    }
    return {
        "type": "function",
        "function": {
            "name": "update_assistant_form",
            "description": (
                "Fill or update the assistant-creation form and/or the set of "
                "enabled agents. Call this whenever you have concrete values to "
                "write. You may set only the fields you want to change. ALWAYS "
                "include a short, friendly `message` telling the admin what you "
                "changed and why."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "message": {
                        "type": "string",
                        "description": "What to say to the admin (GitHub-flavoured markdown ok).",
                    },
                    "fields": {
                        "type": "object",
                        "description": "Form field values to set. Omit fields you don't want to change.",
                        "properties": field_props,
                    },
                    "agents": {
                        "type": "array",
                        "items": {"type": "string", "enum": _agent_keys()},
                        "description": (
                            "The COMPLETE list of agent keys that should be checked "
                            "after this update (it REPLACES the current selection). "
                            "Only include when you intend to change the agent set."
                        ),
                    },
                },
                "required": ["message"],
            },
        },
    }


def _catalog_block() -> str:
    lines = []
    current_cat = None
    for a in _agent_catalog():
        if a["category"] != current_cat:
            current_cat = a["category"]
            lines.append(f"\n### {current_cat}")
        lines.append(
            f"- **{a['name']}** (`{a['key']}`): {a['description']}\n"
            f"    - Input: {a['input']}\n"
            f"    - Output: {a['output']}\n"
            f"    - Example: {a['example']}"
        )
    return "\n".join(lines)


def build_system_prompt() -> str:
    specs = "\n".join(f"  - `{c['value']}` → {c['label']}" for c in _specialization_choices())
    personas = "\n".join(f"  - `{c['value']}` → {c['label']}" for c in _persona_choices())
    fields = "\n".join(f"  - `{k}` — {v}" for k, v in _TEXT_FIELDS.items())
    return f"""\
You are **Studio Copilot**, an expert assistant embedded in the MediverseAI studio.
You help the website admin create and configure a new marketplace assistant.

These marketplace assistants act as **doctors, medical specialists and counselors**
for real users. A weak, vague or unsafe prompt can cause real harm, so everything
you write must be professional, structured and safety-aware.

## Your job
1. Have a natural conversation with the admin to understand the assistant they want
   (who it's for, medical domain, tone, languages, must-do and must-not-do rules).
2. Ask focused clarifying questions when the intent is unclear — but don't
   interrogate; make smart assumptions and offer strong defaults the admin can tweak.
3. Write **high-quality copy** for the form fields (see below), including a rich,
   well-structured `system_prompt`.
4. Decide which **agents/tools** the assistant needs and tick them for the admin.
5. When the admin asks about an agent, explain it thoroughly with concrete examples.

## How you act on the form
Use the `update_assistant_form` tool to write into the form. Set only the fields you
mean to change. Put your chat reply in the tool's `message`. If the admin only wants
an explanation or is still thinking, just reply in plain text (no tool call).
When you fill the form, briefly say what you set and invite the admin to refine it.

## Form fields you can write
{fields}

  - `specialization` — MUST be one of:
{specs}
  - `persona` — MUST be one of:
{personas}

Write both English and Persian (فارسی) copy when you can — the marketplace is bilingual.
Persian text must be natural, fluent فارسی (not transliteration).

## Writing a great `system_prompt`
Produce a structured prompt (use short markdown sections/headings). A strong medical
or counseling system prompt typically covers:
  - **Role & identity**: who the assistant is and its specialty.
  - **Scope**: what topics it handles and what it must refuse / redirect.
  - **Tone**: matching the chosen persona; empathetic, clear, plain language.
  - **Clinical safety**: never diagnose definitively; encourage professional care;
    detect emergencies/red flags and tell the user to seek urgent help; no dosing or
    treatment changes without a clinician; acknowledge uncertainty.
  - **Interaction style**: ask about symptoms/history, one step at a time, summarize.
  - **Boundaries**: no prescriptions, no guaranteed outcomes, respect privacy.
  - **Language**: mirror the user's language (Persian/English).
Tailor these to the specific specialty (e.g. a Family Counselor emphasizes
non-judgmental listening and crisis/self-harm escalation, not drug dosing).
Never output a lazy one-line prompt.

## Available agents/tools (enable only what the assistant truly needs)
The chat runtime exposes ONLY the agents you enable, and calls them automatically
in a safe order. Safety agents always run first. Enable agents that match the
assistant's job; don't over-enable. Typical picks:
  - Almost every medical assistant benefits from `search_knowledge_base` (its own
    uploaded documents) and the safety agents `red_flag_check` + `symptom_triage`.
  - Medication-focused specialists → the Medications agents.
  - Assistants that cite current evidence → `search_web`, `fetch_url`, `search_pubmed`.
  - Assistants that track patients over time → the Follow-up (WhatsApp) agents.
  - A pure counselor usually needs the safety agents and knowledge base, not drugs/labs.
When you tick agents, the `agents` array must be the COMPLETE desired selection.

{_catalog_block()}

## First-draft rule (important)
As soon as the admin names a specialty or an audience, DON'T stall with a list of
questions — immediately call `update_assistant_form` to produce a complete first
draft: fill EVERY field you reasonably can (name, both headlines, both bios, tags,
specialization, persona, and a full `system_prompt`) and tick the agents that fit.
Invent sensible, professional defaults. THEN, in your `message`, tell the admin what
you drafted and ask at most 1–2 short refining questions so they can adjust. Only
ask questions *before* drafting if the request is truly too vague to act on at all.

## Always justify the agents you tick
Whenever you enable (or change) agents, your `message` MUST include a short
"Why these agents" section — one concise line per agent explaining why THIS
assistant needs it (tie it to the specialty/audience). Also briefly note any
notable agent you deliberately left OFF and why. Keep each reason to one line.

## Style
Be concise, warm and expert. Use markdown. When you explain agents, give a short
"when to enable it" note and a realistic example. Default to acting (filling fields
and ticking agents) rather than only describing what you would do.
"""


def _describe_state(form_state: Dict[str, Any]) -> str:
    """A compact snapshot of the current form so the model doesn't overwrite good copy."""
    parts = []
    for k in list(_TEXT_FIELDS) + ["specialization", "persona"]:
        v = (form_state or {}).get(k)
        if v:
            v = str(v)
            if len(v) > 400:
                v = v[:400] + "…"
            parts.append(f"- {k}: {v}")
    agents = (form_state or {}).get("agents") or []
    parts.append(f"- enabled agents: {', '.join(agents) if agents else '(none yet)'}")
    return "Current form state:\n" + "\n".join(parts)


def _sanitize_actions(actions: Dict[str, Any]) -> Dict[str, Any]:
    """Keep only known fields / valid agent keys before sending to the client."""
    out: Dict[str, Any] = {}
    fields = actions.get("fields") or {}
    if isinstance(fields, dict):
        allowed = set(_TEXT_FIELDS) | {"specialization", "persona"}
        spec_vals = {c["value"] for c in _specialization_choices()}
        pers_vals = {c["value"] for c in _persona_choices()}
        clean = {}
        for k, v in fields.items():
            if k not in allowed or v is None:
                continue
            if k == "specialization" and v not in spec_vals:
                continue
            if k == "persona" and v not in pers_vals:
                continue
            clean[k] = v
        if clean:
            out["fields"] = clean
    if "agents" in actions and isinstance(actions["agents"], list):
        out["agents"] = agents_pkg.valid_keys(actions["agents"])
    return out


def _agent_names(keys: List[str]) -> List[str]:
    by_key = {a["key"]: a for a in _agent_catalog()}
    return [f"{by_key[k]['name']} (`{k}`): {by_key[k]['description']}" for k in keys if k in by_key]


def _narrate(client: LLMClient, history: List[Dict[str, str]],
             actions: Dict[str, Any]) -> str:
    """Second pass: when the model filled the form but gave no (or a weak)
    message, generate a proper explanation — including why each agent was picked."""
    fields = list((actions.get("fields") or {}).keys())
    agents = actions.get("agents")
    lines = ["You just filled these form fields: " + (", ".join(fields) or "(none)") + "."]
    if agents is not None:
        lines.append("You enabled these agents:\n- " + "\n- ".join(_agent_names(agents) or ["(none)"]))
    lines.append(
        "Write a concise, friendly message to the admin (markdown) that: (1) says what "
        "you drafted, (2) includes a short **Why these agents** section with one line per "
        "enabled agent tying it to this assistant, and (3) asks at most 1–2 refining "
        "questions. Do NOT call any tool now — just write the message."
    )
    messages = [{"role": "system", "content": build_system_prompt()}]
    for m in history[-12:]:
        if m.get("role") in ("user", "assistant") and (m.get("content") or "").strip():
            messages.append({"role": m["role"], "content": m["content"].strip()})
    messages.append({"role": "system", "content": "\n\n".join(lines)})
    try:
        return client.chat(messages, temperature=0.5).strip()
    except Exception:  # noqa: BLE001
        return ""


def run_copilot(history: List[Dict[str, str]], form_state: Dict[str, Any]) -> Dict[str, Any]:
    """
    history: [{role: 'user'|'assistant', content: str}, ...] (the visible chat).
    form_state: current values of the form fields (so the model is grounded).

    Returns {"reply": str, "actions": {"fields": {...}, "agents": [...]} | None}.
    """
    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": build_system_prompt()},
        {"role": "system", "content": _describe_state(form_state)},
    ]
    for m in history[-20:]:
        role = m.get("role")
        content = (m.get("content") or "").strip()
        if role in ("user", "assistant") and content:
            messages.append({"role": role, "content": content})

    client = LLMClient()
    msg = client.complete(messages, tools=[build_tool_schema()], tool_choice="auto",
                          temperature=0.5)

    reply = (getattr(msg, "content", None) or "").strip()
    actions = None
    tool_calls = getattr(msg, "tool_calls", None) or []
    for tc in tool_calls:
        if getattr(tc.function, "name", "") != "update_assistant_form":
            continue
        try:
            args = json.loads(tc.function.arguments or "{}")
        except (ValueError, TypeError):
            continue
        if args.get("message"):
            reply = args["message"].strip()
        cleaned = _sanitize_actions(args)
        if cleaned:
            actions = cleaned

    # If the model wrote into the form but skipped (or gave a thin) message, or it
    # (re)selected agents, generate a proper explanation with a per-agent rationale.
    weak = len(reply) < 60
    set_agents = actions and actions.get("agents") is not None
    if actions and (weak or set_agents):
        narration = _narrate(client, history, actions)
        if narration:
            reply = narration

    if not reply:
        reply = ("I've updated the form." if actions
                 else "Tell me about the assistant you'd like to build and I'll set it up.")
    return {"reply": reply, "actions": actions}
