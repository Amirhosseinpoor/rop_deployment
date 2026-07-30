# doctors_marketplace/agents/__init__.py
"""Modular chat agents. Importing this package registers every agent."""
from .base import BaseAgent, AgentContext, AgentResult  # noqa: F401
from .registry import (  # noqa: F401
    all_agents, get_agent, get_agents, valid_keys, catalog,
)

# Import agent modules so their @register decorators run.
from . import retrieval, drugs, clinical, whatsapp  # noqa: F401,E402
