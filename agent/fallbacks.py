"""Fallback strategies for the event recommendation agent.

Each fallback is a graph node reached through a conditional edge when the
preceding tool call failed or produced an unusable result. Fallbacks always
increment state.fallback_count and record their name in
state.fallback_strategies_used; when a fallback has several stages, the
stage that produced results is recorded in state.fallback_details.
"""

import logging

from agent import tools
from agent.state import AgentState
from agent.tools import _load_json

logger = logging.getLogger(__name__)

TRENDING_LIMIT = 20


def graceful_degradation_fallback(state: AgentState) -> AgentState:
    """Fallback A: user preference lookup failed.

    Uses a synthetic default profile with all genres equally weighted and
    no history, so downstream search still has genres to work with.
    """
    logger.warning("Fallback triggered: User preference lookup failed; using default profile")

    events = _load_json("events.json")
    all_genres = sorted({e["genre"] for e in events})

    state.user_preferences.genres = all_genres
    state.user_preferences.attended = []
    state.user_preferences.skipped = []
    state.user_preferences.browsed = []
    state.user_preferences.budget = 0

    state.fallback_count += 1
    state.fallback_strategies_used.append("graceful_degradation_fallback")
    return state


def alternative_source_fallback(state: AgentState) -> AgentState:
    """Fallback B: event search returned nothing (or failed).

    Relaxes the request one constraint at a time instead of discarding it:

    1. ``drop_dates``: keep the genres and budget, drop the date range.
    2. ``drop_dates_and_budget``: keep the genres only.
    3. ``trending``: the top 20 events by popularity regardless of genre.

    A user who asked for "a concert next week" therefore still gets
    concerts, just on other dates, and only lands on the trending list
    when their genres have nothing at all. The stage that produced results
    is recorded in ``state.fallback_details["alternative_source_fallback"]``.
    """
    intent = state.parsed_intent
    genres = intent.genres or state.user_preferences.genres
    budget = state.user_preferences.budget if state.user_preferences.budget > 0 else None
    had_dates = bool(intent.date_from or intent.date_to)

    stages: list[tuple[str, dict]] = []
    if genres and had_dates:
        stages.append(("drop_dates", {"genres": genres, "price_max": budget}))
    if genres and budget is not None:
        stages.append(("drop_dates_and_budget", {"genres": genres}))

    logger.warning(
        "Fallback triggered: event search returned nothing for genres=%s; relaxing via %s",
        genres, [name for name, _ in stages] + ["trending"],
    )

    results: list[dict] = []
    stage_used = "trending"
    for name, kwargs in stages:
        try:
            results = tools.search_events(**kwargs)
        except Exception as exc:  # noqa: BLE001 - a failing catalog just means "try the next stage"
            logger.warning("alternative_source_fallback: stage %s failed: %s", name, exc)
            results = []
        if results:
            stage_used = name
            break

    if not results:
        events = _load_json("events.json")
        results = sorted(events, key=lambda e: e["popularity_score"], reverse=True)[:TRENDING_LIMIT]

    logger.warning("alternative_source_fallback: stage %s produced %d events", stage_used, len(results))

    state.search_results = results
    state.fallback_details["alternative_source_fallback"] = stage_used
    state.fallback_count += 1
    state.fallback_strategies_used.append("alternative_source_fallback")
    return state


def partial_results_fallback(state: AgentState) -> AgentState:
    """Fallback C: ranking step failed or timed out.

    Returns the top 10 unranked events from search_results so the user
    still gets recommendations, flagged as partial/unranked.
    """
    logger.warning("Fallback triggered: Ranking timed out; returning partial results")

    state.ranked_recommendations = state.search_results[:10]

    state.fallback_count += 1
    state.fallback_strategies_used.append("partial_results_fallback")
    return state
