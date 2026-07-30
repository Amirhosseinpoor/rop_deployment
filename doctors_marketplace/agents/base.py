# doctors_marketplace/agents/base.py
"""
Base interface for modular chat *agents* (a.k.a. tools).

Each agent is a self-contained capability the superuser can switch on per
assistant in the studio. The chat runtime exposes ONLY the agents enabled for a
given doctor to the LLM as callable tools, so behaviour is fully modular and the
model picks the right tool for each turn.

An agent declares rich metadata (used both to build the OpenAI tool schema and to
render the studio explanation card) and implements ``run(args, ctx)``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class AgentContext:
    """Everything an agent may need, passed by the runtime."""
    doctor: Any
    session: Any = None
    user: Any = None
    history: List[Dict[str, Any]] = field(default_factory=list)
    query: str = ""
    now_iso: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentResult:
    """What an agent returns.

    content : str  — text fed back to the LLM as the tool result.
    sources : list — numbered-citation contributions; each dict may hold
                     {type:'local'|'web', title, url, domain, page_content}.
    ui      : dict — optional UI side-effect streamed to the client
                     (e.g. {"banner": {...}}), rendered by the frontend.
    display : str  — short human summary of what the agent did.
    ok      : bool — False signals a soft failure (still fed to the model).
    """
    content: str = ""
    sources: List[Dict[str, Any]] = field(default_factory=list)
    ui: Optional[Dict[str, Any]] = None
    display: str = ""
    ok: bool = True


class BaseAgent:
    # ---- identity / studio metadata ----
    key: str = ""                 # stable id, also the tool/function name
    name: str = ""                # human label
    icon: str = "tool"            # frontend stage-icon key
    category: str = "General"     # studio grouping
    description: str = ""         # studio: what it does (also the tool description)
    input_desc: str = ""          # studio: Input
    output_desc: str = ""         # studio: Output
    example: str = ""             # studio: Example
    stage_label: str = ""         # live status text shown while running

    # ---- runtime behaviour ----
    produces_sources: bool = False
    # Lower runs earlier. Agents with pre_pass=True run automatically BEFORE the
    # model (deterministic ordering for safety-critical steps like red-flag).
    run_order: int = 100
    pre_pass: bool = False
    # OpenAI JSON-schema for the tool's arguments.
    parameters: Dict[str, Any] = {"type": "object", "properties": {}}

    def tool_schema(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.key,
                "description": self.description,
                "parameters": self.parameters or {"type": "object", "properties": {}},
            },
        }

    def catalog_entry(self) -> Dict[str, Any]:
        """Metadata for the studio agent-selection card."""
        return {
            "key": self.key, "name": self.name, "icon": self.icon,
            "category": self.category, "description": self.description,
            "input": self.input_desc, "output": self.output_desc,
            "example": self.example,
        }

    def run(self, args: Dict[str, Any], ctx: AgentContext) -> AgentResult:  # pragma: no cover
        raise NotImplementedError
