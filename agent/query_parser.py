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

# A stated price cap: "under $15", "less than 20 dollars", "$30 or less".
# An amount needs a "$" or a currency word, so "under 18" (an age) isn't read
# as a price.
_AMOUNT = r"(?:\$\s*(\d+(?:\.\d+)?)|(\d+(?:\.\d+)?)\s*(?:dollars|bucks))"
PRICE_CAP_PATTERNS = [
    re.compile(rf"\b(?:under|below|less than|cheaper than|up to|no more than|at most|max(?:imum)?)\s+{_AMOUNT}"),
    re.compile(rf"{_AMOUNT}\s+(?:or less|or under|max)\b"),
]
# "free" as a price, but not as availability ("I'm free tonight", "free time").
FREE_PATTERN = re.compile(r"\bfree\b(?!\s+(?:time|tonight|today|tomorrow|this|next|on|to|for|after)\b)")


def _empty_intent() -> dict:
    return {"genres": [], "date_from": None, "date_to": None, "price_max": None}


def _stated_price_max(text: str) -> float | None:
    """The price cap the request states, or None if it states none."""
    if FREE_PATTERN.search(text):
        return 0.0
    for pattern in PRICE_CAP_PATTERNS:
        if match := pattern.search(text):
            return float(next(g for g in match.groups() if g is not None))
    return None


def _week_bounds(d: date) -> tuple[date, date]:
    monday = d - timedelta(days=d.weekday())
    return monday, monday + timedelta(days=6)


def _month_bounds(d: date) -> tuple[date, date]:
    next_month = d.replace(day=28) + timedelta(days=4)
    last_day = next_month - timedelta(days=next_month.day)
    return d.replace(day=1), last_day


def parse_query_rule_based(query: str, today: date | None = None) -> dict:
    """Extract genres, a date range and a stated price cap from free text using keyword matching.

    Args:
        query: The user's free-text request.
        today: Reference date for relative phrases like "tonight" or
            "next week". Defaults to the current date.

    Returns:
        A dict with genres (list, possibly empty), date_from and date_to
        (ISO date strings, or None if the query has no time reference), and
        price_max (a stated cap such as "under $15" or "free", else None).
    """
    today = today or date.today()
    text = query.lower()
    intent = _empty_intent()

    for genre, pattern in GENRE_PATTERNS.items():
        if pattern.search(text):
            intent["genres"].append(genre)

    intent["price_max"] = _stated_price_max(text)

    if "tonight" in text or "today" in text:
        intent["date_from"] = intent["date_to"] = today.isoformat()
    elif "tomorrow" in text:
        tomorrow = today + timedelta(days=1)
        intent["date_from"] = intent["date_to"] = tomorrow.isoformat()
    elif "next weekend" in text:
        # Checked before "next week", which it contains: the Saturday-Sunday
        # of next week, not the whole week.
        monday, sunday = _week_bounds(today + timedelta(days=7))
        intent["date_from"], intent["date_to"] = (monday + timedelta(days=5)).isoformat(), sunday.isoformat()
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
