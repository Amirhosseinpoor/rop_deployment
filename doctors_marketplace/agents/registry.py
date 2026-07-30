# doctors_marketplace/agents/registry.py
"""Agent registry. Agent modules call @register; the runtime and studio read it."""
from __future__ import annotations

from typing import Dict, List

from .base import BaseAgent

_REGISTRY: Dict[str, BaseAgent] = {}
# Deterministic display order for the studio (by category then registration).
_ORDER: List[str] = []


def register(cls):
    """Class decorator: instantiate and register an agent by its key."""
    inst = cls()
    if not inst.key:
        raise ValueError(f"{cls.__name__} has no key")
    _REGISTRY[inst.key] = inst
    if inst.key not in _ORDER:
        _ORDER.append(inst.key)
    return cls


def all_agents() -> Dict[str, BaseAgent]:
    return dict(_REGISTRY)


def get_agent(key: str):
    return _REGISTRY.get(key)


def get_agents(keys):
    """Enabled agents in a stable, sequencing-aware order (run_order, then key)."""
    chosen = [_REGISTRY[k] for k in (keys or []) if k in _REGISTRY]
    return sorted(chosen, key=lambda a: (a.run_order, a.key))


def valid_keys(keys):
    return [k for k in (keys or []) if k in _REGISTRY]


def catalog() -> List[dict]:
    """Studio metadata for every registered agent, grouped by category."""
    return [_REGISTRY[k].catalog_entry() for k in _ORDER]
