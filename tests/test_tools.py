import json
from pathlib import Path

import pytest

from agent import tools
from agent.tools import fetch_user_preferences, rank_by_popularity, rank_events, search_events

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


@pytest.fixture(scope="module")
def users():
    with open(DATA_DIR / "users.json") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def events():
    with open(DATA_DIR / "events.json") as f:
        return json.load(f)


def test_fetch_user_preferences_exists(users):
    user = users[0]
    prefs = fetch_user_preferences(user["id"])

    assert prefs["genres"] == user["genres"]
    assert prefs["past_events_count"] == len(user["past_events"])
    assert prefs["budget"] == user["avg_ticket_price"]
    assert isinstance(prefs["attendance_frequency"], int)
    assert set(prefs["history"]) == {"attended", "skipped", "browsed"}
    assert prefs["attendance_frequency"] == len(prefs["history"]["attended"])


def test_fetch_user_preferences_missing():
    with pytest.raises(ValueError):
        fetch_user_preferences("no-such-user")


def test_search_events_by_genre(events):
    genre = events[0]["genre"]
    results = search_events([genre])

    assert len(results) > 0
    assert all(e["genre"] == genre for e in results)


def test_search_events_price_filter(events):
    genre = events[0]["genre"]
    price_max = 25
    results = search_events([genre], price_max=price_max)

    assert all(e["price"] <= price_max for e in results)


def test_search_events_date_filter(events):
    genre = events[0]["genre"]
    dates = sorted(e["date"] for e in events if e["genre"] == genre)
    date_from, date_to = dates[0], dates[0]

    results = search_events([genre], date_from=date_from, date_to=date_to)

    assert len(results) > 0
    assert all(date_from <= e["date"] <= date_to for e in results)


def test_search_events_empty():
    results = search_events(["Not A Real Genre"])
    assert results == []


def test_rank_by_popularity(events):
    ranked = rank_by_popularity(events[:10])

    scores = [e["popularity_score"] for e in ranked]
    assert scores == sorted(scores, reverse=True)


def test_rank_by_popularity_limit(events):
    ranked = rank_by_popularity(events, limit=3)
    assert len(ranked) == 3


def test_rank_by_popularity_empty():
    assert rank_by_popularity([]) == []


def test_search_events_cap_keeps_the_most_popular_matches(events):
    all_genres = sorted({e["genre"] for e in events})
    top_score = max(e["popularity_score"] for e in events)

    results = search_events(all_genres, limit=20)

    assert len(results) == 20
    assert results[0]["popularity_score"] == top_score
    assert rank_by_popularity(results, limit=1)[0]["popularity_score"] == top_score


# --------------------------------------------------------------------------- #
# Personalised ranking
# --------------------------------------------------------------------------- #

CANDIDATES = [
    {"id": "m1", "genre": "Music", "popularity_score": 70, "price": 20},
    {"id": "c1", "genre": "Comedy", "popularity_score": 80, "price": 20},
    {"id": "f1", "genre": "Film", "popularity_score": 90, "price": 20},
]

CATALOG = CANDIDATES + [
    {"id": "m_past", "genre": "Music", "popularity_score": 50, "price": 20},
    {"id": "f_skip", "genre": "Film", "popularity_score": 50, "price": 20},
    {"id": "c_browse", "genre": "Comedy", "popularity_score": 50, "price": 20},
]


@pytest.fixture
def fixed_catalog(monkeypatch):
    monkeypatch.setattr(tools, "_load_json", lambda filename: CATALOG)


def test_rank_events_without_history_is_popularity_order():
    ranked = rank_events(CANDIDATES)
    assert [e["id"] for e in ranked] == ["f1", "c1", "m1"]
    assert all(e["rank_score"] == e["popularity_score"] for e in ranked)
    assert ranked[0]["rank_reason"] == "popularity 90"


def test_rank_events_boosts_attended_genres_and_penalises_skipped(fixed_catalog):
    history = {"attended": ["m_past", "m_past"], "skipped": ["f_skip", "f_skip"], "browsed": ["c_browse"]}
    ranked = rank_events(CANDIDATES, history=history)

    by_id = {e["id"]: e for e in ranked}
    assert by_id["m1"]["rank_score"] == 70 + 2 * tools.ATTENDED_WEIGHT
    assert by_id["f1"]["rank_score"] == 90 - 2 * tools.SKIPPED_WEIGHT
    assert by_id["c1"]["rank_score"] == 80 + tools.BROWSED_WEIGHT
    assert [e["id"] for e in ranked] == ["m1", "c1", "f1"]
    assert "attended 2 Music" in by_id["m1"]["rank_reason"]
    assert "skipped 2 Film" in by_id["f1"]["rank_reason"]


def test_rank_events_excludes_already_attended_events(fixed_catalog):
    ranked = rank_events(CANDIDATES, history={"attended": ["f1"]})
    assert [e["id"] for e in ranked] == ["c1", "m1"]


def test_rank_events_keeps_attended_events_when_nothing_else_is_left(fixed_catalog):
    ranked = rank_events(CANDIDATES[:1], history={"attended": ["m1"]})
    assert [e["id"] for e in ranked] == ["m1"]


def test_rank_events_does_not_mutate_input(fixed_catalog):
    before = [dict(e) for e in CANDIDATES]
    rank_events(CANDIDATES, history={"attended": ["m_past"]})
    assert CANDIDATES == before


def test_rank_events_respects_limit_and_empty():
    assert rank_events([]) == []
    assert len(rank_events(CANDIDATES, limit=2)) == 2
