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
"""

from __future__ import annotations

import os
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


def run_case(case: dict, path: str, today: date = PINNED_TODAY) -> dict:
    """Run one case on one path and return the agent's result dict."""
    with pinned_clock(today), forced_path(path), injected_fault(case.get("fault")):
        result = recommendation_agent.invoke(case["user_id"], case["request"])
    result.pop("logs", None)
    return result
