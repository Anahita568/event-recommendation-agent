from datetime import date

from agent import query_parser


def test_rule_based_extracts_genre_and_tonight():
    intent = query_parser.parse_query_rule_based("find me a concert tonight", today=date(2026, 9, 13))
    assert intent["genres"] == ["Music"]
    assert intent["date_from"] == intent["date_to"] == "2026-09-13"


def test_rule_based_extracts_next_week():
    intent = query_parser.parse_query_rule_based("anything fun next week", today=date(2026, 9, 13))
    assert intent["date_from"] == "2026-09-14"
    assert intent["date_to"] == "2026-09-20"


def test_rule_based_no_keywords_returns_empty_intent():
    intent = query_parser.parse_query_rule_based("surprise me", today=date(2026, 9, 13))
    assert intent["genres"] == []
    assert intent["date_from"] is None
    assert intent["date_to"] is None


def test_rule_based_multiple_genre_keywords():
    intent = query_parser.parse_query_rule_based("a comedy show or a film screening", today=date(2026, 9, 13))
    assert set(intent["genres"]) == {"Comedy", "Film"}


def test_rule_based_keywords_match_whole_words_only():
    for query in ("find me a party tonight", "something to start the weekend"):
        assert query_parser.parse_query_rule_based(query, today=date(2026, 9, 14))["genres"] == []
    assert query_parser.parse_query_rule_based("kids playing outside", today=date(2026, 9, 14))["genres"] == ["Family"]
    assert query_parser.parse_query_rule_based("what matches my taste", today=date(2026, 9, 14))["genres"] == []


def test_rule_based_keywords_accept_simple_plurals():
    intent = query_parser.parse_query_rule_based("any concerts or movies?", today=date(2026, 9, 14))
    assert set(intent["genres"]) == {"Music", "Film"}


def test_rule_based_weekend_on_a_weekday_is_the_coming_saturday_and_sunday():
    intent = query_parser.parse_query_rule_based("this weekend", today=date(2026, 9, 16))  # Wednesday
    assert (intent["date_from"], intent["date_to"]) == ("2026-09-19", "2026-09-20")


def test_rule_based_weekend_on_a_sunday_never_reaches_into_the_past():
    intent = query_parser.parse_query_rule_based("this weekend", today=date(2026, 9, 13))  # Sunday
    assert (intent["date_from"], intent["date_to"]) == ("2026-09-13", "2026-09-13")


def test_rule_based_next_weekend_is_saturday_and_sunday_of_next_week():
    intent = query_parser.parse_query_rule_based("comedy next weekend", today=date(2026, 9, 17))  # Thursday
    assert (intent["date_from"], intent["date_to"]) == ("2026-09-26", "2026-09-27")
    # "next week" still means the whole of next week.
    intent = query_parser.parse_query_rule_based("comedy next week", today=date(2026, 9, 17))
    assert (intent["date_from"], intent["date_to"]) == ("2026-09-21", "2026-09-27")
