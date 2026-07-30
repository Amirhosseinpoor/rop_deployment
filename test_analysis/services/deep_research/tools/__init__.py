"""
Deep Health Research v2 — tools layer.

Thin, reusable primitives that the research agents use to touch the outside
world. Everything the LLM agents do flows through these functions so the agents
themselves stay pure "reason over the given evidence" units.

  T5  json_llm     — guaranteed-JSON structured LLM call (raw client, JSON mode)
  T4  vision_read  — multimodal read of an image (eye crops, report scans)
  T1  web_search   — Serper → Google organic results
  T2  read_and_rank— scrape + embed-rank web pages into ranked Documents
  T3  kb_retrieve  — internal RAG knowledge-base similarity search
"""
from .llm_json import json_llm, SYNTH_MODEL, REASONING_MODEL, VISION_MODEL  # noqa: F401
from .vision import vision_read  # noqa: F401
from .web import web_search, read_and_rank, kb_retrieve, web_enabled, domain_of  # noqa: F401
from .literature import lit_search  # noqa: F401
