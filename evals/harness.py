"""Run one eval case through the real agent, on a chosen path.

Nothing here changes agent behaviour. The harness only controls the
environment around ``invoke()``:

- **Clock.** The agent reads ``date.today()``; the event catalog is fixed.
  The ``date`` name in the modules that read it is swapped for a subclass
  whose ``today()`` returns the pinned date, so labels like "next week"
  stay valid.
- **Path.** ``EVENT_AGENT_LLM=off`` forces the deterministic path; unset,
  the agent takes the LLM path when credentials are present.
- **Faults.** A case can ask for a tool to raise (``search_down``,
  ``rank_down``) to reach fallbacks no realistic request can trigger.
- **Model calls.** On the LLM path, ``tool_calling.make_client`` is wrapped
  so every ``messages.create`` round-trip is recorded: token usage, stop
  reason, the tool calls the model made and whether each one errored. This
  survives a failed tool loop, when the agent itself keeps no transcript.
"""

from __future__ import annotations

import os
import time
from contextlib import ExitStack, contextmanager
from datetime import date
from unittest import mock

from agent import query_parser, recommendation_agent, tool_calling, tools

PINNED_TODAY = date(2026, 9, 17)
PATHS = ("deterministic", "llm")


@contextmanager
def pinned_clock(today: date):
    class PinnedDate(date):
        @classmethod
        def today(cls):
            return cls(today.year, today.month, today.day)

    with ExitStack() as stack:
        for module in (query_parser, tool_calling):
            stack.enter_context(mock.patch.object(module, "date", PinnedDate))
        yield


def _raise(name: str):
    def fail(*args, **kwargs):
        raise RuntimeError(f"injected fault: {name} unavailable")

    return fail


FAULTS = {
    "search_down": ("search_events",),
    "rank_down": ("rank_events",),
}


@contextmanager
def injected_fault(fault: str | None):
    with ExitStack() as stack:
        for name in FAULTS.get(fault, ()) if fault else ():
            stack.enter_context(mock.patch.object(tools, name, _raise(name)))
        yield


@contextmanager
def forced_path(path: str):
    value = "off" if path == "deterministic" else "auto"
    with mock.patch.dict(os.environ, {"EVENT_AGENT_LLM": value}):
        yield


class _RecordingMessages:
    def __init__(self, inner, calls: list[dict]):
        self._inner = inner
        self._calls = calls

    def create(self, **kwargs):
        # The tool results for the previous turn are the last request message.
        last = kwargs.get("messages", [{}])[-1].get("content")
        results = [
            {"tool_use_id": b["tool_use_id"], "is_error": bool(b.get("is_error"))}
            for b in (last if isinstance(last, list) else [])
            if isinstance(b, dict) and b.get("type") == "tool_result"
        ]
        call = {"tool_results_sent": results}
        self._calls.append(call)
        started_at = time.perf_counter()
        try:
            response = self._inner.create(**kwargs)
        except Exception as exc:
            call.update(latency_ms=(time.perf_counter() - started_at) * 1000, error=str(exc))
            raise
        usage = response.usage
        call.update(
            latency_ms=(time.perf_counter() - started_at) * 1000,
            stop_reason=response.stop_reason,
            usage={
                "input_tokens": usage.input_tokens or 0,
                "output_tokens": usage.output_tokens or 0,
                "cache_write_tokens": getattr(usage, "cache_creation_input_tokens", None) or 0,
                "cache_read_tokens": getattr(usage, "cache_read_input_tokens", None) or 0,
            },
            text=[b.text for b in response.content if b.type == "text"],
            tool_uses=[
                {"id": b.id, "name": b.name, "input": b.input}
                for b in response.content if b.type == "tool_use"
            ],
        )
        return response


class RecordingClient:
    """Wraps an Anthropic client and records every messages.create call."""

    def __init__(self, inner, calls: list[dict]):
        self.messages = _RecordingMessages(inner.messages, calls)


def summarize_calls(calls: list[dict]) -> dict:
    """Collapse the recorded model calls into tool-use facts and token totals."""
    errored = {r["tool_use_id"] for c in calls for r in c["tool_results_sent"] if r["is_error"]}
    answered = {r["tool_use_id"] for c in calls for r in c["tool_results_sent"]}
    tool_uses = [t for c in calls for t in c.get("tool_uses", [])]
    searches = [t for t in tool_uses if t["name"] == "search_events"]
    ok_searches = [t for t in searches if t["id"] in answered and t["id"] not in errored]
    usage = {k: sum(c.get("usage", {}).get(k, 0) for c in calls)
             for k in ("input_tokens", "output_tokens", "cache_write_tokens", "cache_read_tokens")}
    return {
        "iterations": len(calls),
        "tool_calls": [t["name"] for t in tool_uses],
        "tool_errors": len(errored),
        "unknown_tools": sorted({t["name"] for t in tool_uses} - {"fetch_user_preferences", "search_events"}),
        "fetched_preferences": any(t["name"] == "fetch_user_preferences" for t in tool_uses),
        "searched": bool(ok_searches),
        # The model's own reading of the request: its last successful search,
        # or failing that its last attempted one.
        "search_args": (ok_searches or searches or [{}])[-1].get("input"),
        "text": [t for c in calls for t in c.get("text", [])],
        "stop_reasons": [c.get("stop_reason") for c in calls],
        "api_errors": [c["error"] for c in calls if "error" in c],
        "model_latency_ms": sum(c.get("latency_ms", 0) for c in calls),
        "usage": usage,
    }


@contextmanager
def recorded_model_calls():
    """Yield a list that fills with the model calls made inside the block."""
    calls: list[dict] = []
    real_make_client = tool_calling.make_client

    def make_client():
        return RecordingClient(real_make_client(), calls)

    with mock.patch.object(tool_calling, "make_client", make_client):
        yield calls


def run_case(case: dict, path: str, today: date = PINNED_TODAY) -> tuple[dict, dict | None]:
    """Run one case on one path.

    Returns the agent's result dict and, on the LLM path, a summary of the
    model calls (None on the deterministic path).
    """
    with pinned_clock(today), forced_path(path), injected_fault(case.get("fault")), recorded_model_calls() as calls:
        result = recommendation_agent.invoke(case["user_id"], case["request"])
    result.pop("logs", None)
    return result, (summarize_calls(calls) if path == "llm" else None)
