"""Tool definitions for the event recommendation agent.

Each tool loads from the synthetic JSON data in data/ and is designed to be
called directly by agent nodes (see agent/recommendation_agent.py).
"""

import json
import logging
from collections import Counter
from datetime import date
from pathlib import Path

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def _load_json(filename: str):
    path = DATA_DIR / filename
    with open(path) as f:
        return json.load(f)


def upcoming(events: list[dict]) -> list[dict]:
    """Drop events dated before today; they can no longer be attended."""
    today = date.today().isoformat()
    return [e for e in events if e["date"] >= today]


def fetch_user_preferences(user_id: str) -> dict:
    """Look up a user's preferences by id.

    Args:
        user_id: The id of the user to look up (e.g. "u01").

    Returns:
        A dict with genres, past_events_count, budget, attendance_frequency,
        and history (attended / skipped / browsed event ids).

    Raises:
        ValueError: If no user with the given id exists.
    """
    users = _load_json("users.json")
    history = _load_json("user_history.json")

    user = next((u for u in users if u["id"] == user_id), None)
    if user is None:
        logger.error("fetch_user_preferences: user_id %s not found", user_id)
        raise ValueError(f"User not found: {user_id}")

    user_history = history.get(user_id, {})
    attended = user_history.get("attended", [])

    return {
        "genres": user["genres"],
        "past_events_count": len(user["past_events"]),
        "budget": user["avg_ticket_price"],
        "attendance_frequency": len(attended),
        "history": {
            "attended": attended,
            "skipped": user_history.get("skipped", []),
            "browsed": user_history.get("browsed", []),
        },
    }


def search_events(
    genres: list[str],
    price_max: float | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    limit: int = 20,
) -> list[dict]:
    """Search for upcoming events matching the given genres, price cap, and date range.

    Events dated before today are never returned, whatever date_from says,
    so a search with no date range (or one relaxed by a fallback) can't
    recommend something that already happened.

    Args:
        genres: List of genres to match against event genre.
        price_max: If given, exclude events priced above this amount.
        date_from: If given (ISO "YYYY-MM-DD"), exclude events before this date.
        date_to: If given (ISO "YYYY-MM-DD"), exclude events after this date.
        limit: Maximum number of events to return. When more events match,
            the most popular ones are kept, so a cap never hides the best
            candidates from the ranking step.

    Returns:
        A list of matching event dicts, or an empty list if none match.
    """
    events = upcoming(_load_json("events.json"))

    matches = [e for e in events if e["genre"] in genres]
    if price_max is not None:
        matches = [e for e in matches if e["price"] <= price_max]
    if date_from is not None:
        matches = [e for e in matches if e["date"] >= date_from]
    if date_to is not None:
        matches = [e for e in matches if e["date"] <= date_to]

    if not matches:
        logger.warning(
            "search_events: no events found for genres=%s price_max=%s date_from=%s date_to=%s",
            genres, price_max, date_from, date_to,
        )
        return []

    # Cap by popularity, not catalog order: the ranker downstream only ever
    # sees what is returned here.
    matches.sort(key=lambda e: e["popularity_score"], reverse=True)
    return matches[:limit]


def rank_by_popularity(events: list[dict], limit: int = 5) -> list[dict]:
    """Rank events by popularity score, descending.

    Args:
        events: List of event dicts, each containing a popularity_score.
        limit: Maximum number of events to return.

    Returns:
        The top `limit` events sorted by popularity_score descending,
        or an empty list if `events` is empty.
    """
    if not events:
        logger.warning("rank_by_popularity: called with empty events list")
        return []

    ranked = sorted(events, key=lambda e: e["popularity_score"], reverse=True)
    return ranked[:limit]


# Per-event weights applied to popularity_score (0-100) for each event in
# the user's history that shares the candidate's genre. Attending is the
# strongest signal, browsing a weak one, and skipping counts against.
ATTENDED_WEIGHT = 8
BROWSED_WEIGHT = 3
SKIPPED_WEIGHT = 5


def rank_events(events: list[dict], history: dict | None = None, limit: int = 5) -> list[dict]:
    """Rank events by popularity, personalised by the user's history.

    Each candidate starts from its popularity_score and gains
    ATTENDED_WEIGHT for every event the user attended in the same genre,
    BROWSED_WEIGHT for every one they browsed, and loses SKIPPED_WEIGHT for
    every one they skipped. Events the user already attended are excluded
    unless nothing else is left. Without a history this is a plain
    popularity sort.

    Args:
        events: Candidate event dicts, each with id, genre, popularity_score.
        history: Optional dict with "attended", "skipped", and "browsed"
            lists of event ids.
        limit: Maximum number of events to return.

    Returns:
        The top `limit` events, each copied with a `rank_score` field and a
        `rank_reason` string, or an empty list if `events` is empty.
    """
    if not events:
        logger.warning("rank_events: called with empty events list")
        return []

    history = history or {}
    if not any(history.get(k) for k in ("attended", "skipped", "browsed")):
        return [
            {**e, "rank_score": e["popularity_score"], "rank_reason": f"popularity {e['popularity_score']}"}
            for e in rank_by_popularity(events, limit)
        ]

    catalog = {e["id"]: e for e in _load_json("events.json")}

    def genre_counts(ids: list[str]) -> Counter:
        return Counter(catalog[i]["genre"] for i in ids if i in catalog)

    attended = genre_counts(history.get("attended", []))
    browsed = genre_counts(history.get("browsed", []))
    skipped = genre_counts(history.get("skipped", []))
    attended_ids = set(history.get("attended", []))

    def scored(e: dict) -> dict:
        genre = e["genre"]
        bonus = ATTENDED_WEIGHT * attended[genre] + BROWSED_WEIGHT * browsed[genre] - SKIPPED_WEIGHT * skipped[genre]
        parts = [f"popularity {e['popularity_score']}"]
        if attended[genre]:
            parts.append(f"attended {attended[genre]} {genre}")
        if browsed[genre]:
            parts.append(f"browsed {browsed[genre]} {genre}")
        if skipped[genre]:
            parts.append(f"skipped {skipped[genre]} {genre}")
        return {**e, "rank_score": e["popularity_score"] + bonus, "rank_reason": ", ".join(parts)}

    candidates = [e for e in events if e["id"] not in attended_ids] or events
    ranked = sorted((scored(e) for e in candidates), key=lambda e: e["rank_score"], reverse=True)
    return ranked[:limit]
