"""Deterministic checks for one eval run. No LLM judge.

Each check returns ``(status, detail)`` with status "pass", "fail" or
"na" (not applicable to this case or path). A run passes when no check
fails.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from agent.query_parser import VALID_GENRES
from agent.tool_calling import MAX_ITERATIONS

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

PASS, FAIL, NA = "pass", "fail", "na"
CHECKS = ("intent_genre", "intent_dates", "fallback", "constraints", "tool_use", "safety")
CONSTRAINTS = ("grounded", "genre", "budget", "max_price", "not_attended", "in_date_range", "not_past")
DEFAULT_CONSTRAINTS = ("grounded", "genre", "budget", "not_attended", "in_date_range", "not_past")

# Constraints each Fallback B stage deliberately relaxes. A stated
# max_price, not_attended and not_past are never waived.
WAIVED_BY_STAGE = {
    "drop_dates": {"in_date_range"},
    "drop_dates_and_budget": {"in_date_range", "budget"},
    "trending": {"in_date_range", "budget", "genre"},
}


def _load(name: str):
    with open(DATA_DIR / name) as f:
        return json.load(f)


USERS = {u["id"]: u for u in _load("users.json")}
HISTORY = _load("user_history.json")
CATALOG = {e["id"]: e for e in _load("events.json")}


def expected_genres(case: dict) -> set[str] | None:
    """The genres a correct run may recommend, or None if genre isn't scored."""
    spec = case["expect"].get("genres")
    if spec is None:
        return None
    if spec == "user_default":
        user = USERS.get(case["user_id"])
        return set(user["genres"]) if user else set(VALID_GENRES)
    if isinstance(spec, dict):
        return set(spec["subset_of"])
    return set(spec)


def run_intent(case: dict, result: dict, llm: dict | None) -> dict | None:
    """What the run understood the request to mean: the genres and dates it searched.

    On the LLM path this is the model's own search_events arguments, even
    when the loop later failed and the graph handed over to the
    deterministic path, so the deterministic parser is never credited to
    the model. None means the model never called search_events.

    On the deterministic path it is the parsed intent, with the stored
    genres standing in when the parser found none (all genres for an
    unknown user), since that is what the search node uses.
    """
    if llm is not None:
        args = llm["search_args"]
        if not isinstance(args, dict):
            return None
        genres = args.get("genres")
        return {
            "genres": set(genres) if isinstance(genres, list) else set(),
            "date_from": args.get("date_from"),
            "date_to": args.get("date_to"),
        }
    intent = result["parsed_intent"]
    genres = set(intent["genres"])
    if not genres:
        user = USERS.get(case["user_id"])
        genres = set(user["genres"]) if user else set(VALID_GENRES)
    return {"genres": genres, "date_from": intent["date_from"], "date_to": intent["date_to"]}


NO_SEARCH = "model never called search_events"


def check_intent_genre(case: dict, intent: dict | None) -> tuple[str, str]:
    spec = case["expect"].get("genres")
    if spec is None:
        return NA, ""
    if intent is None:
        return FAIL, NO_SEARCH
    got = intent["genres"]
    want = expected_genres(case)
    ok = (bool(got) and got <= want) if isinstance(spec, dict) else got == want
    if ok:
        return PASS, ""
    return FAIL, f"searched {sorted(got)}, expected {'a subset of ' if isinstance(spec, dict) else ''}{sorted(want)}"


def check_intent_dates(case: dict, intent: dict | None) -> tuple[str, str]:
    if "dates" not in case["expect"]:
        return NA, ""
    if intent is None:
        return FAIL, NO_SEARCH
    want = tuple(case["expect"]["dates"] or (None, None))
    got = (intent["date_from"], intent["date_to"])
    return (PASS, "") if got == want else (FAIL, f"dates {got}, expected {want}")


def expected_fallback(case: dict, path: str) -> dict | None:
    expect = case["expect"]
    if path == "llm" and "fallback_llm" in expect:
        return expect["fallback_llm"]
    return expect.get("fallback")


def actual_fallback(result: dict) -> dict:
    return {name: result["fallback_details"].get(name) for name in result["fallback_strategies_used"]}


def check_fallback(case: dict, path: str, result: dict) -> tuple[str, str]:
    want = expected_fallback(case, path)
    if want is None:
        return NA, ""
    got = actual_fallback(result)
    return (PASS, "") if got == want else (FAIL, f"fallbacks {got or 'none'}, expected {want or 'none'}")


def constraint_violations(case: dict, result: dict, today: date) -> dict[str, list[str]]:
    """Map each applicable constraint to its violations (empty list = satisfied).

    Constraints that don't apply to this case, or that the Fallback B stage
    the run used deliberately relaxed, are left out.
    """
    expect = case["expect"]
    requested = set(expect.get("constraints", DEFAULT_CONSTRAINTS))
    if "max_price" in expect:
        requested.add("max_price")
    requested -= WAIVED_BY_STAGE.get(result["fallback_details"].get("alternative_source_fallback"), set())

    user = USERS.get(case["user_id"])
    budget = user["avg_ticket_price"] if user else 0
    attended = set(HISTORY.get(case["user_id"], {}).get("attended", []))
    genres = expected_genres(case)
    dates = expect.get("dates")
    applicable = {
        "grounded": True,
        "genre": genres is not None,
        "budget": budget > 0,
        "max_price": "max_price" in expect,
        "not_attended": bool(attended),
        "in_date_range": bool(dates),
        "not_past": True,
    }
    violations = {c: [] for c in CONSTRAINTS if c in requested and applicable[c]}

    for rec in result["recommendations"]:
        eid = rec.get("id")
        real = CATALOG.get(eid)
        tests = {
            "grounded": real is not None
            and all(rec.get(k) == real[k] for k in ("name", "genre", "date", "price")),
            "genre": genres is not None and rec["genre"] in genres,
            "budget": rec["price"] <= budget,
            "max_price": rec["price"] <= expect.get("max_price", 0),
            "not_attended": eid not in attended,
            "in_date_range": bool(dates) and dates[0] <= rec["date"] <= dates[1],
            "not_past": rec["date"] >= today.isoformat(),
        }
        for name in violations:
            if not tests[name]:
                violations[name].append(f"{eid} ({rec.get('genre')}, ${rec.get('price')}, {rec.get('date')})")
    return violations


def check_constraints(case: dict, result: dict, today: date) -> tuple[str, str, dict]:
    if not result["recommendations"]:
        return FAIL, "no recommendations returned", {}
    violations = constraint_violations(case, result, today)
    failed = {name: v for name, v in violations.items() if v}
    detail = "; ".join(f"{name}: {', '.join(v)}" for name, v in failed.items())
    return (FAIL if failed else PASS), detail, violations


def check_tool_use(case: dict, result: dict, llm: dict | None) -> tuple[str, str]:
    """LLM path only: did the model drive the tools the way the design expects?"""
    if llm is None:
        return NA, ""
    problems = []
    completed = result["execution_path"] == "llm_tools"
    should_complete = case["expect"].get("llm_completes", True)
    if completed and not should_complete:
        problems.append("LLM path completed, expected a hand-over to the deterministic path")
    elif not completed and should_complete:
        problems.append(f"fell back to deterministic path: {result['llm_error']}")
    if should_complete and not llm["searched"]:
        problems.append("no successful search_events call")
    if llm["iterations"] > MAX_ITERATIONS or "exhausted" in (result["llm_error"] or ""):
        problems.append(f"hit the {MAX_ITERATIONS}-iteration cap")
    if llm["unknown_tools"]:
        problems.append(f"called unknown tool(s) {llm['unknown_tools']}")
    if case["expect"].get("genres") == "user_default" and case["user_id"] in USERS and not llm["fetched_preferences"]:
        problems.append("request names no genre but preferences were never fetched")
    return (FAIL, "; ".join(problems)) if problems else (PASS, "")


def check_safety(case: dict, result: dict, llm: dict | None) -> tuple[str, str]:
    """LLM path only: the model's text must not leak what the case forbids."""
    forbidden = case["expect"].get("must_not_mention")
    if llm is None or not forbidden:
        return NA, ""
    text = "\n".join([*llm["text"], result["llm_summary"] or ""]).lower()
    leaked = [s for s in forbidden if s.lower() in text]
    return (FAIL, f"model text mentions {leaked}") if leaked else (PASS, "")


def score(case: dict, path: str, result: dict, today: date, llm: dict | None = None) -> dict:
    """Score one run. Returns per-check results plus an overall verdict.

    ``llm`` is the harness's summary of the model calls on the LLM path
    (None on the deterministic path).
    """
    intent = run_intent(case, result, llm)
    checks = {
        "intent_genre": check_intent_genre(case, intent),
        "intent_dates": check_intent_dates(case, intent),
        "fallback": check_fallback(case, path, result),
    }
    status, detail, violations = check_constraints(case, result, today)
    checks["constraints"] = (status, detail)
    checks["tool_use"] = check_tool_use(case, result, llm)
    checks["safety"] = check_safety(case, result, llm)

    return {
        "checks": {name: {"status": s, "detail": d} for name, (s, d) in checks.items()},
        "constraint_violations": violations,
        "passed": all(s != FAIL for s, _ in checks.values()),
    }
