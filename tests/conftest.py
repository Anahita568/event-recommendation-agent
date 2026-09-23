from datetime import date

import pytest

from agent import query_parser, tool_calling, tools

# The event catalog is fixed, so every test sees the same "today". Without
# this, relative dates ("next week") and the past-event filter in
# search_events would change with the calendar.
TODAY = date(2026, 9, 17)


@pytest.fixture(autouse=True)
def pinned_today(monkeypatch):
    class PinnedDate(date):
        @classmethod
        def today(cls):
            return cls(TODAY.year, TODAY.month, TODAY.day)

    for module in (query_parser, tool_calling, tools):
        monkeypatch.setattr(module, "date", PinnedDate)


@pytest.fixture(autouse=True)
def deterministic_by_default(monkeypatch):
    """Never touch Bedrock from the test suite unless a test opts in.

    Tests that exercise the LLM path re-patch ``bedrock_available`` and
    ``run_tool_loop`` themselves; a later monkeypatch wins.
    """
    monkeypatch.setattr(tool_calling, "bedrock_available", lambda: (False, "disabled in tests"))
