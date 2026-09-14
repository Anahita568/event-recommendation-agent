import pytest

from agent import tool_calling


@pytest.fixture(autouse=True)
def deterministic_by_default(monkeypatch):
    """Never touch Bedrock from the test suite unless a test opts in.

    Tests that exercise the LLM path re-patch ``bedrock_available`` and
    ``run_tool_loop`` themselves; a later monkeypatch wins.
    """
    monkeypatch.setattr(tool_calling, "bedrock_available", lambda: (False, "disabled in tests"))
