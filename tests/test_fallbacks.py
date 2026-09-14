import json
from pathlib import Path

from agent import tools
from agent.fallbacks import (
    alternative_source_fallback,
    graceful_degradation_fallback,
    partial_results_fallback,
)
from agent.state import AgentState, QueryIntent, UserPreferences

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def _all_genres():
    with open(DATA_DIR / "events.json") as f:
        events = json.load(f)
    return sorted({e["genre"] for e in events})


def test_fallback_a_default_profile():
    state = AgentState(user_id="no-such-user", query="anything fun")
    result = graceful_degradation_fallback(state)

    assert result.user_preferences.genres == _all_genres()
    assert result.fallback_count == 1
    assert "graceful_degradation_fallback" in result.fallback_strategies_used


def test_fallback_b_trending_when_nothing_to_relax():
    state = AgentState(user_id="u01", query="anything fun")
    result = alternative_source_fallback(state)

    assert len(result.search_results) == 20
    scores = [e["popularity_score"] for e in result.search_results]
    assert scores == sorted(scores, reverse=True)
    assert result.fallback_count == 1
    assert "alternative_source_fallback" in result.fallback_strategies_used
    assert result.fallback_details == {"alternative_source_fallback": "trending"}


def test_fallback_b_drops_dates_first_and_keeps_genre_and_budget():
    state = AgentState(
        user_id="u01", query="a concert next week",
        parsed_intent=QueryIntent(genres=["Music"], date_from="2026-09-21", date_to="2026-09-27"),
        user_preferences=UserPreferences(genres=["Sports"], budget=60),
    )
    result = alternative_source_fallback(state)

    assert result.fallback_details["alternative_source_fallback"] == "drop_dates"
    assert result.search_results
    assert all(e["genre"] == "Music" and e["price"] <= 60 for e in result.search_results)


def test_fallback_b_drops_budget_second(monkeypatch):
    calls = []

    def fake_search(genres, price_max=None, date_from=None, date_to=None, limit=20):
        calls.append((tuple(genres), price_max))
        return [] if price_max is not None else [{"id": "e1", "genre": genres[0], "popularity_score": 1, "price": 999}]

    monkeypatch.setattr(tools, "search_events", fake_search)
    state = AgentState(
        user_id="u01", query="a concert next week",
        parsed_intent=QueryIntent(genres=["Music"], date_from="2026-09-21", date_to="2026-09-27"),
        user_preferences=UserPreferences(genres=[], budget=10),
    )
    result = alternative_source_fallback(state)

    assert calls == [(("Music",), 10), (("Music",), None)]
    assert result.fallback_details["alternative_source_fallback"] == "drop_dates_and_budget"
    assert result.search_results == [{"id": "e1", "genre": "Music", "popularity_score": 1, "price": 999}]


def test_fallback_b_uses_stored_genres_when_query_named_none():
    state = AgentState(
        user_id="u01", query="anything next week",
        parsed_intent=QueryIntent(date_from="2026-09-21", date_to="2026-09-27"),
        user_preferences=UserPreferences(genres=["Comedy"], budget=0),
    )
    result = alternative_source_fallback(state)

    assert result.fallback_details["alternative_source_fallback"] == "drop_dates"
    assert {e["genre"] for e in result.search_results} == {"Comedy"}


def test_fallback_b_search_exception_falls_through_to_trending(monkeypatch):
    def boom(genres, price_max=None, date_from=None, date_to=None, limit=20):
        raise TimeoutError("catalog unavailable")

    monkeypatch.setattr(tools, "search_events", boom)
    state = AgentState(
        user_id="u01", query="a concert next week",
        parsed_intent=QueryIntent(genres=["Music"], date_from="2026-09-21", date_to="2026-09-27"),
    )
    result = alternative_source_fallback(state)

    assert result.fallback_details["alternative_source_fallback"] == "trending"
    assert len(result.search_results) == 20


def test_fallback_c_partial_results():
    dummy_events = [{"id": f"e{i}", "popularity_score": i} for i in range(15)]
    state = AgentState(user_id="u01", query="anything fun", search_results=dummy_events)
    result = partial_results_fallback(state)

    assert result.ranked_recommendations == dummy_events[:10]
    assert result.fallback_count == 1
    assert "partial_results_fallback" in result.fallback_strategies_used
