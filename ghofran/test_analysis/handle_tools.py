"""Tool-call dispatchers for the report pipeline's LLMs.

Ported verbatim (logic-wise) from the original ``handle_tools.py``: given an LLM
message containing a tool call, run the matching Python function and wrap the
result in the ``role: tool`` envelope the Chat Completions API expects.
"""
from __future__ import annotations

import json

from .disease_models import predict_hypertension_risk
from .scrapers import scrape_doctors


def handle_tool_call(message):
    """Execute a ``predict_hypertension_risk`` tool call and return its envelope."""
    tool_call = message.tool_calls[0]
    if tool_call.function.name == "predict_hypertension_risk":
        args = json.loads(tool_call.function.arguments)
        content = predict_hypertension_risk(**args)
        return {"role": "tool", "content": content, "tool_call_id": tool_call.id}


def handle_doctors_call(message):
    """Execute a ``scrape_doctors`` tool call and return its envelope."""
    tool_call = message.tool_calls[0]
    if tool_call.function.name == "scrape_doctors":
        args = json.loads(tool_call.function.arguments)
        content = scrape_doctors(
            city=args.get("city"),
            speciality=args.get("speciality"),
            region=args.get("region"),
            insurance=args.get("insurance"),
        )
        return {"role": "tool", "content": content, "tool_call_id": tool_call.id}
