from agent import tool_calling, tools
from agent.recommendation_agent import build_graph, invoke
from agent.tool_calling import ToolLoopError, ToolLoopOutcome

FIXED_EVENTS = [
    {"id": "e1", "genre": "Music", "popularity_score": 10, "price": 20},
    {"id": "e2", "genre": "Music", "popularity_score": 90, "price": 30},
]


def _enable_llm(monkeypatch, loop):
    monkeypatch.setattr(tool_calling, "bedrock_available", lambda: (True, None))
    monkeypatch.setattr(tool_calling, "run_tool_loop", loop)


# --------------------------------------------------------------------------- #
# Graph shape
# --------------------------------------------------------------------------- #


def test_fallbacks_are_reached_only_through_conditional_edges():
    compiled = build_graph().compile()
    edges = compiled.get_graph().edges

    conditional = {(e.source, e.target) for e in edges if e.conditional}
    assert conditional == {
        ("start", "llm_tool_loop"),
        ("start", "parse_query"),
        ("llm_tool_loop", "parse_query"),
        ("llm_tool_loop", "alternative_source_fallback"),
        ("llm_tool_loop", "rank_events"),
        ("fetch_preferences", "graceful_degradation_fallback"),
        ("fetch_preferences", "search_events"),
        ("search_events", "alternative_source_fallback"),
        ("search_events", "rank_events"),
        ("rank_events", "partial_results_fallback"),
        ("rank_events", "format_output"),
    }

    # Every fallback is its own node, entered only via a conditional edge.
    for fallback in ("graceful_degradation_fallback", "alternative_source_fallback", "partial_results_fallback"):
        incoming = [e for e in edges if e.target == fallback]
        assert incoming and all(e.conditional for e in incoming), fallback


# --------------------------------------------------------------------------- #
# Deterministic path
# --------------------------------------------------------------------------- #


def test_agent_happy_path(monkeypatch):
    monkeypatch.setattr(
        tools, "fetch_user_preferences",
        lambda user_id: {"genres": ["Music"], "past_events_count": 2, "budget": 50, "attendance_frequency": 2},
    )
    monkeypatch.setattr(
        tools, "search_events", lambda genres, price_max=None, date_from=None, date_to=None, limit=20: FIXED_EVENTS
    )
    monkeypatch.setattr(
        tools, "rank_events",
        lambda events, history=None, limit=5: sorted(events, key=lambda e: e["popularity_score"], reverse=True),
    )

    result = invoke("u01", "find me a concert")

    assert result["execution_path"] == "deterministic"
    assert result["fallback_count"] == 0
    assert result["fallback_strategies_used"] == []
    assert [e["id"] for e in result["recommendations"]] == ["e2", "e1"]


def test_agent_missing_user_routes_to_graceful_degradation(monkeypatch):
    def raise_not_found(user_id):
        raise ValueError(f"User not found: {user_id}")

    monkeypatch.setattr(tools, "fetch_user_preferences", raise_not_found)

    result = invoke("no-such-user", "find me something fun")

    assert result["fallback_strategies_used"] == ["graceful_degradation_fallback"]
    assert len(result["recommendations"]) > 0


def test_agent_no_matching_events_routes_to_alternative_source(monkeypatch):
    monkeypatch.setattr(
        tools, "fetch_user_preferences",
        lambda user_id: {"genres": ["Nonexistent"], "past_events_count": 0, "budget": 20, "attendance_frequency": 0},
    )
    monkeypatch.setattr(
        tools, "search_events", lambda genres, price_max=None, date_from=None, date_to=None, limit=20: []
    )

    result = invoke("u01", "find me something fun")

    assert result["fallback_strategies_used"] == ["alternative_source_fallback"]
    assert len(result["recommendations"]) > 0


def test_agent_search_exception_routes_to_alternative_source(monkeypatch):
    def boom(genres, price_max=None, date_from=None, date_to=None, limit=20):
        raise TimeoutError("catalog unavailable")

    monkeypatch.setattr(tools, "search_events", boom)

    result = invoke("u01", "find me a concert")

    assert result["fallback_strategies_used"] == ["alternative_source_fallback"]
    assert len(result["recommendations"]) > 0


def test_agent_ranking_failure_routes_to_partial_results(monkeypatch):
    def boom(events, history=None, limit=5):
        raise TimeoutError("ranker timed out")

    monkeypatch.setattr(tools, "rank_events", boom)

    result = invoke("u01", "find me a concert")

    assert result["fallback_strategies_used"] == ["partial_results_fallback"]
    assert 0 < len(result["recommendations"]) <= 10


def test_agent_relaxes_dates_before_genre_when_week_has_no_match():
    # The catalog starts weeks after the fixed "next week", so the exact
    # search is empty. The fallback must keep the requested genre.
    result = invoke("u01", "find me a concert next week")

    assert result["fallback_strategies_used"] == ["alternative_source_fallback"]
    assert result["fallback_details"] == {"alternative_source_fallback": "drop_dates"}
    assert result["recommendations"]
    assert {e["genre"] for e in result["recommendations"]} == {"Music"}


def test_agent_ranking_uses_attendance_history():
    # u01 likes Food & Drink, Sports and Film, has attended three Sports
    # events and one Food & Drink event, and skipped a Film.
    result = invoke("u01", "something fun")
    recs = result["recommendations"]

    assert result["fallback_strategies_used"] == []
    assert recs == sorted(recs, key=lambda e: e["rank_score"], reverse=True)
    by_genre = {}
    for e in recs:
        by_genre.setdefault(e["genre"], e)
    sports, food = by_genre["Sports"], by_genre["Food & Drink"]
    assert sports["rank_score"] == sports["popularity_score"] + 3 * tools.ATTENDED_WEIGHT
    assert "attended 3 Sports" in sports["rank_reason"]
    # Personalisation outranks raw popularity: the Sports pick is less
    # popular than the Food & Drink pick but lands above it.
    assert sports["popularity_score"] < food["popularity_score"]
    assert recs.index(sports) < recs.index(food)


def test_agent_returns_metrics():
    result = invoke("u01", "find me a concert")

    assert isinstance(result["total_latency_ms"], float)
    assert result["total_latency_ms"] >= 0
    assert "Tool calls:" in result["logs"]
    assert "Total latency:" in result["logs"]


# --------------------------------------------------------------------------- #
# LLM path
# --------------------------------------------------------------------------- #


def test_llm_path_skips_deterministic_nodes(monkeypatch):
    def fake_loop(user_id, query, agent_logger, **kwargs):
        agent_logger.log_tool_call("search_events[llm]", 1.0, True)
        return ToolLoopOutcome(
            parsed_intent={"genres": ["Music"], "date_from": None, "date_to": None},
            search_results=FIXED_EVENTS,
            user_preferences={"genres": ["Music"], "budget": 50},
            summary="Searched Music.",
        )

    _enable_llm(monkeypatch, fake_loop)
    monkeypatch.setattr(tools, "fetch_user_preferences", lambda uid: (_ for _ in ()).throw(AssertionError("not called")))

    result = invoke("u01", "find me a concert")

    assert result["execution_path"] == "llm_tools"
    assert result["llm_summary"] == "Searched Music."
    assert result["fallback_strategies_used"] == []
    assert [e["id"] for e in result["recommendations"]] == ["e2", "e1"]
    assert "parse_query[rule_based]" not in result["logs"]


def test_llm_path_empty_search_routes_to_alternative_source(monkeypatch):
    def fake_loop(user_id, query, agent_logger, **kwargs):
        return ToolLoopOutcome(parsed_intent={"genres": ["Dance"], "date_from": None, "date_to": None}, search_results=[])

    _enable_llm(monkeypatch, fake_loop)

    result = invoke("u01", "ballet tonight")

    assert result["execution_path"] == "llm_tools"
    assert result["fallback_strategies_used"] == ["alternative_source_fallback"]
    assert len(result["recommendations"]) > 0


def test_llm_failure_routes_to_deterministic_path(monkeypatch):
    def failing_loop(user_id, query, agent_logger, **kwargs):
        raise ToolLoopError("Bedrock call failed on iteration 1: throttled")

    _enable_llm(monkeypatch, failing_loop)

    result = invoke("u01", "find me a concert")

    assert result["execution_path"] == "deterministic"
    assert "throttled" in result["llm_error"]
    assert result["parsed_intent"]["genres"] == ["Music"]
    assert "parse_query[rule_based]" in result["logs"]
    assert len(result["recommendations"]) > 0


def test_llm_unavailable_reason_is_reported(monkeypatch):
    monkeypatch.setattr(tool_calling, "bedrock_available", lambda: (False, "no AWS credentials configured"))

    result = invoke("u01", "find me a concert")

    assert result["execution_path"] == "deterministic"
    assert result["llm_error"] == "no AWS credentials configured"
