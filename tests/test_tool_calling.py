"""Tests for the LLM tool-calling loop, driven by a scripted fake Bedrock client.

No network: the fake client replays a fixed sequence of model turns and
records every request it received, so tests can assert on what was sent
back to the model (error results, single-message batching, etc.).
"""

import json
from datetime import date
from types import SimpleNamespace

import pytest

from agent import tool_calling, tools
from agent.tool_calling import ToolLoopError, build_tools, run_tool_loop
from agent.tool_calling import bedrock_available as real_bedrock_available  # bound before conftest patches it
from metrics import AgentLogger

TODAY = date(2026, 9, 13)


def text(t):
    return SimpleNamespace(type="text", text=t)


def tool_use(name, input, id="tu_1"):
    return SimpleNamespace(type="tool_use", id=id, name=name, input=input)


def turn(*blocks, stop_reason="tool_use"):
    return SimpleNamespace(content=list(blocks), stop_reason=stop_reason)


class FakeClient:
    """Replays scripted turns and records each request's kwargs."""

    def __init__(self, turns):
        self._turns = list(turns)
        self.requests = []
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        # Snapshot: the loop keeps appending to the same messages list.
        self.requests.append({**kwargs, "messages": list(kwargs["messages"])})
        if not self._turns:
            raise AssertionError("fake client ran out of scripted turns")
        return self._turns.pop(0)


def tool_results(request):
    """The tool_result blocks in the last user message of a request."""
    last = request["messages"][-1]
    assert last["role"] == "user"
    return [b for b in last["content"] if isinstance(b, dict) and b.get("type") == "tool_result"]


# --------------------------------------------------------------------------- #
# Tool definitions
# --------------------------------------------------------------------------- #


def test_tool_definitions_expose_schema_and_bind_user():
    registry = build_tools("u01")
    defs = {spec.name: spec.to_anthropic_tool() for spec in registry.values()}

    assert set(defs) == {"fetch_user_preferences", "search_events"}
    # Identity is not a model-controlled argument.
    assert defs["fetch_user_preferences"]["input_schema"]["properties"] == {}
    assert defs["fetch_user_preferences"]["input_schema"]["additionalProperties"] is False
    genres = defs["search_events"]["input_schema"]["properties"]["genres"]
    assert genres["items"]["enum"] == tool_calling.VALID_GENRES
    assert "genres" in defs["search_events"]["input_schema"]["required"]


def test_bound_fetch_tool_uses_request_user(monkeypatch):
    seen = []
    monkeypatch.setattr(tools, "fetch_user_preferences", lambda uid: seen.append(uid) or {"genres": [], "budget": 0})
    build_tools("u07")["fetch_user_preferences"].fn()
    assert seen == ["u07"]


# --------------------------------------------------------------------------- #
# The loop
# --------------------------------------------------------------------------- #


def test_happy_path_model_fetches_then_searches(monkeypatch):
    monkeypatch.setattr(
        tools, "fetch_user_preferences",
        lambda uid: {"genres": ["Music"], "past_events_count": 1, "budget": 60, "attendance_frequency": 1},
    )
    events = [{"id": "e1", "genre": "Music", "popularity_score": 50, "price": 40}]
    captured = {}

    def fake_search(genres, price_max=None, date_from=None, date_to=None, limit=20):
        captured.update(genres=genres, price_max=price_max, date_from=date_from, date_to=date_to)
        return events

    monkeypatch.setattr(tools, "search_events", fake_search)

    client = FakeClient([
        turn(tool_use("fetch_user_preferences", {}, id="a")),
        turn(tool_use("search_events", {"genres": ["Music"], "price_max": 60, "date_from": "2026-09-21", "date_to": "2026-09-27"}, id="b")),
        turn(text("Searched for Music events next week under $60."), stop_reason="end_turn"),
    ])
    logger = AgentLogger()

    outcome = run_tool_loop("u01", "a concert next week", logger, client=client, today=TODAY)

    assert outcome.user_preferences["genres"] == ["Music"]
    assert outcome.parsed_intent == {"genres": ["Music"], "date_from": "2026-09-21", "date_to": "2026-09-27"}
    assert outcome.search_results == events
    assert captured == {"genres": ["Music"], "price_max": 60, "date_from": "2026-09-21", "date_to": "2026-09-27"}
    assert outcome.summary == "Searched for Music events next week under $60."
    assert outcome.iterations == 3 and outcome.tool_calls == 2 and outcome.tool_errors == 0

    # Every model round-trip and every tool call is metered.
    names = [c.tool_name for c in logger.tool_calls]
    assert names == ["bedrock.messages", "fetch_user_preferences[llm]", "bedrock.messages", "search_events[llm]", "bedrock.messages"]
    assert all(c.success for c in logger.tool_calls)

    # The request carried our tool definitions and system prompt.
    first = client.requests[0]
    assert {t["name"] for t in first["tools"]} == {"fetch_user_preferences", "search_events"}
    assert "2026-09-13" in first["system"]
    # Tool results were fed back with the matching ids.
    assert tool_results(client.requests[1])[0]["tool_use_id"] == "a"
    assert json.loads(tool_results(client.requests[2])[0]["content"])["count"] == 1


def test_invalid_arguments_are_returned_as_error_and_model_can_retry(monkeypatch):
    calls = []
    monkeypatch.setattr(
        tools, "search_events",
        lambda genres, price_max=None, date_from=None, date_to=None, limit=20: calls.append(genres) or [{"id": "e1"}],
    )
    client = FakeClient([
        turn(tool_use("search_events", {"genres": ["Jazz"], "date_from": "next week"}, id="bad")),
        turn(tool_use("search_events", {"genres": ["Music"]}, id="good")),
        turn(text("done"), stop_reason="end_turn"),
    ])
    logger = AgentLogger()

    outcome = run_tool_loop("u01", "jazz next week", logger, client=client, today=TODAY)

    # The bad call never reached the tool; the model got an error result and corrected itself.
    assert calls == [["Music"]]
    err = tool_results(client.requests[1])[0]
    assert err["is_error"] is True and err["tool_use_id"] == "bad"
    assert "genres" in err["content"] and "date_from" in err["content"]
    assert outcome.tool_errors == 1 and outcome.search_results == [{"id": "e1"}]
    assert [c.success for c in logger.tool_calls if c.tool_name == "search_events[llm]"] == [False, True]


def test_tool_exception_is_reported_not_raised(monkeypatch):
    def missing(uid):
        raise ValueError(f"User not found: {uid}")

    monkeypatch.setattr(tools, "fetch_user_preferences", missing)
    monkeypatch.setattr(
        tools, "search_events", lambda genres, price_max=None, date_from=None, date_to=None, limit=20: [{"id": "e9"}]
    )
    client = FakeClient([
        turn(tool_use("fetch_user_preferences", {}, id="p")),
        turn(tool_use("search_events", {"genres": ["Comedy"]}, id="s")),
        turn(text("done"), stop_reason="end_turn"),
    ])

    outcome = run_tool_loop("ghost", "comedy tonight", AgentLogger(), client=client, today=TODAY)

    err = tool_results(client.requests[1])[0]
    assert err["is_error"] is True and "User not found: ghost" in err["content"]
    assert outcome.user_preferences is None
    assert outcome.parsed_intent["genres"] == ["Comedy"]


def test_parallel_tool_calls_return_all_results_in_one_message(monkeypatch):
    monkeypatch.setattr(tools, "fetch_user_preferences", lambda uid: {"genres": ["Art"], "budget": 0})
    monkeypatch.setattr(
        tools, "search_events", lambda genres, price_max=None, date_from=None, date_to=None, limit=20: []
    )
    client = FakeClient([
        turn(tool_use("fetch_user_preferences", {}, id="x"), tool_use("search_events", {"genres": ["Art"]}, id="y")),
        turn(text("done"), stop_reason="end_turn"),
    ])

    outcome = run_tool_loop("u01", "art", AgentLogger(), client=client, today=TODAY)

    results = tool_results(client.requests[1])
    assert [r["tool_use_id"] for r in results] == ["x", "y"]
    assert outcome.search_results == []  # empty is a valid, successful search


def test_unknown_tool_name_is_an_error_result(monkeypatch):
    monkeypatch.setattr(
        tools, "search_events", lambda genres, price_max=None, date_from=None, date_to=None, limit=20: [{"id": "e1"}]
    )
    client = FakeClient([
        turn(tool_use("delete_all_events", {}, id="nope")),
        turn(tool_use("search_events", {"genres": ["Film"]}, id="ok")),
        turn(text("done"), stop_reason="end_turn"),
    ])

    run_tool_loop("u01", "a movie", AgentLogger(), client=client, today=TODAY)

    err = tool_results(client.requests[1])[0]
    assert err["is_error"] is True and "Unknown tool" in err["content"]


def test_finishing_without_a_search_raises():
    client = FakeClient([turn(text("I have no idea."), stop_reason="end_turn")])
    with pytest.raises(ToolLoopError, match="without a successful search_events"):
        run_tool_loop("u01", "hello", AgentLogger(), client=client, today=TODAY)


def test_iteration_cap_is_enforced(monkeypatch):
    monkeypatch.setattr(tools, "fetch_user_preferences", lambda uid: {"genres": ["Art"], "budget": 0})
    client = FakeClient([turn(tool_use("fetch_user_preferences", {}, id=str(i))) for i in range(10)])
    with pytest.raises(ToolLoopError, match="exhausted 3 iterations"):
        run_tool_loop("u01", "loop forever", AgentLogger(), client=client, max_iterations=3, today=TODAY)
    assert len(client.requests) == 3


def test_unexpected_stop_reason_raises():
    client = FakeClient([turn(text("..."), stop_reason="max_tokens")])
    with pytest.raises(ToolLoopError, match="unexpected stop_reason 'max_tokens'"):
        run_tool_loop("u01", "hi", AgentLogger(), client=client, today=TODAY)


def test_api_failure_is_metered_and_raised():
    class Boom:
        def __init__(self):
            self.messages = SimpleNamespace(create=self._create)

        def _create(self, **kwargs):
            raise RuntimeError("throttled")

    logger = AgentLogger()
    with pytest.raises(ToolLoopError, match="Bedrock call failed on iteration 1"):
        run_tool_loop("u01", "hi", logger, client=Boom(), today=TODAY)
    assert [(c.tool_name, c.success) for c in logger.tool_calls] == [("bedrock.messages", False)]


# --------------------------------------------------------------------------- #
# Availability gate
# --------------------------------------------------------------------------- #

_CRED_VARS = [
    "AWS_ACCESS_KEY_ID",
    "AWS_PROFILE",
    "AWS_CONTAINER_CREDENTIALS_RELATIVE_URI",
    "AWS_CONTAINER_CREDENTIALS_FULL_URI",
    "AWS_EXECUTION_ENV",
]


def _clear_creds(monkeypatch):
    for var in _CRED_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.delenv("EVENT_AGENT_LLM", raising=False)
    monkeypatch.setattr(tool_calling.os.path, "exists", lambda path: False)


def test_bedrock_unavailable_without_credentials(monkeypatch):
    _clear_creds(monkeypatch)
    assert real_bedrock_available() == (False, "no AWS credentials configured")


def test_bedrock_available_with_access_key(monkeypatch):
    _clear_creds(monkeypatch)
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "fake-key")
    assert real_bedrock_available() == (True, None)


def test_bedrock_can_be_forced_off(monkeypatch):
    _clear_creds(monkeypatch)
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "fake-key")
    monkeypatch.setenv("EVENT_AGENT_LLM", "off")
    assert real_bedrock_available() == (False, "EVENT_AGENT_LLM=off")
