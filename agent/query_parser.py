"""Deterministic natural-language query parsing: extract genre and date-range intent.

This is the no-cloud path. When Amazon Bedrock is available the graph
instead lets Claude choose and call the tools directly (see
agent/tool_calling.py); this module is what runs when it isn't, or when
that path fails.
"""

import logging
import re
from datetime import date, timedelta

logger = logging.getLogger(__name__)

VALID_GENRES = [
    "Music", "Comedy", "Theater", "Sports", "Tech",
    "Food & Drink", "Art", "Film", "Dance", "Family",
]

GENRE_KEYWORDS = {
    "Music": ["concert", "music", "gig", "band", "live music"],
    "Comedy": ["comedy", "stand-up", "standup", "comedian"],
    "Theater": ["theater", "theatre", "play", "musical"],
    "Sports": ["sports", "game", "match", "tournament"],
    "Tech": ["tech", "hackathon", "conference", "meetup"],
    "Food & Drink": ["food", "drink", "tasting", "brewery", "wine", "beer"],
    "Art": ["art", "gallery", "exhibit", "exhibition"],
    "Film": ["film", "movie", "screening", "cinema"],
    "Dance": ["dance", "dancing", "ballet"],
    "Family": ["family", "kids", "children", "carnival"],
}


def _keyword_pattern(keywords: list[str]) -> re.Pattern:
    """Match any keyword as a whole word (optionally pluralised with "s").

    Whole-word matching keeps "art" from firing on "party" or "start",
    "play" on "playing", and "match" on "matches".
    """
    alternatives = "|".join(re.escape(kw) for kw in keywords)
    return re.compile(rf"\b(?:{alternatives})s?\b")


GENRE_PATTERNS = {genre: _keyword_pattern(keywords) for genre, keywords in GENRE_KEYWORDS.items()}


def _empty_intent() -> dict:
    return {"genres": [], "date_from": None, "date_to": None}


def _week_bounds(d: date) -> tuple[date, date]:
    monday = d - timedelta(days=d.weekday())
    return monday, monday + timedelta(days=6)


def _month_bounds(d: date) -> tuple[date, date]:
    next_month = d.replace(day=28) + timedelta(days=4)
    last_day = next_month - timedelta(days=next_month.day)
    return d.replace(day=1), last_day


def parse_query_rule_based(query: str, today: date | None = None) -> dict:
    """Extract genres and a date range from free text using keyword matching.

    Args:
        query: The user's free-text request.
        today: Reference date for relative phrases like "tonight" or
            "next week". Defaults to the current date.

    Returns:
        A dict with genres (list, possibly empty), date_from, and date_to
        (ISO date strings, or None if the query has no time reference).
    """
    today = today or date.today()
    text = query.lower()
    intent = _empty_intent()

    for genre, pattern in GENRE_PATTERNS.items():
        if pattern.search(text):
            intent["genres"].append(genre)

    if "tonight" in text or "today" in text:
        intent["date_from"] = intent["date_to"] = today.isoformat()
    elif "tomorrow" in text:
        tomorrow = today + timedelta(days=1)
        intent["date_from"] = intent["date_to"] = tomorrow.isoformat()
    elif "next week" in text:
        monday, sunday = _week_bounds(today + timedelta(days=7))
        intent["date_from"], intent["date_to"] = monday.isoformat(), sunday.isoformat()
    elif "weekend" in text:
        # The coming Saturday-Sunday; if today is already the weekend, the
        # range starts today rather than reaching back to a past Saturday.
        monday, sunday = _week_bounds(today)
        saturday = monday + timedelta(days=5)
        intent["date_from"], intent["date_to"] = max(today, saturday).isoformat(), sunday.isoformat()
    elif "this week" in text:
        _, sunday = _week_bounds(today)
        intent["date_from"], intent["date_to"] = today.isoformat(), sunday.isoformat()
    elif "next month" in text:
        first, last = _month_bounds(today.replace(day=28) + timedelta(days=4))
        intent["date_from"], intent["date_to"] = first.isoformat(), last.isoformat()
    elif "this month" in text:
        _, last = _month_bounds(today)
        intent["date_from"], intent["date_to"] = today.isoformat(), last.isoformat()

    return intent
