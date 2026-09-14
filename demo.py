#!/usr/bin/env python3
"""Command-line entry point for the event recommendation agent.

Usage:
    python demo.py <user_id> "<query>"     one-shot: run a single query
    python demo.py                         interactive mode
    python demo.py -v <user_id> "<query>"  one-shot with full per-node logs
    python demo.py --no-llm <user_id> "<query>"  force the deterministic path
    python demo.py --transcript <user_id> "<query>"  also print the full Bedrock exchange
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

from agent.recommendation_agent import invoke

DATA_DIR = Path(__file__).resolve().parent / "data"


def _load_users() -> list[dict]:
    with open(DATA_DIR / "users.json") as f:
        return json.load(f)


def _print_result(result: dict, transcript: bool = False) -> None:
    print(f"\nQuery: {result['query']!r}")
    note = " (Claude on Bedrock chose the tool calls)" if result["execution_path"] == "llm_tools" else ""
    print(f"Execution path: {result['execution_path']}{note}")
    if result["llm_error"]:
        print(f"LLM path skipped/failed: {result['llm_error']}")
    if result["llm_summary"]:
        print(f"Model summary: {result['llm_summary']}")
    print(f"Parsed intent: {result['parsed_intent']}")
    if result["fallback_strategies_used"]:
        print(f"Fallbacks used: {result['fallback_strategies_used']}")
    for name, detail in result["fallback_details"].items():
        print(f"  {name}: {detail}")
    print(f"Latency: {result['total_latency_ms']:.2f}ms")

    if not result["recommendations"]:
        print("\nNo recommendations found.")
        return

    print("\nRecommendations:")
    for event in result["recommendations"]:
        reason = f"  [{event['rank_reason']}]" if event.get("rank_reason") else ""
        print(
            f"  - {event['name']} ({event['genre']}) at {event['venue']} "
            f"on {event['date']} — ${event['price']}{reason}"
        )

    if transcript:
        print("\nBedrock transcript:")
        if result["llm_transcript"]:
            print(json.dumps(result["llm_transcript"], indent=2))
        else:
            print("  (none: the LLM path did not run)")


def run_once(user_id: str, query: str, transcript: bool = False) -> None:
    result = invoke(user_id, query)
    _print_result(result, transcript=transcript)


def run_interactive() -> None:
    users = _load_users()
    valid_ids = {user["id"] for user in users}

    print("Event Recommendation Agent — interactive mode")
    print(f"Available users ({len(users)}):")
    for user in users:
        print(f"  {user['id']}: {user['name']} — likes {', '.join(user['genres'])}")
    print("\nTry a query like 'find me a concert tonight' or 'something fun next week'.")
    print("Type 'quit' or press Ctrl-D at either prompt to exit.\n")

    while True:
        try:
            user_id = input("User id: ").strip()
        except EOFError:
            print()
            break
        if user_id.lower() in ("quit", "exit"):
            break
        if user_id not in valid_ids:
            print(f"Unknown user id {user_id!r}. Try one of: {', '.join(sorted(valid_ids))}\n")
            continue

        try:
            query = input("Query: ").strip()
        except EOFError:
            print()
            break
        if query.lower() in ("quit", "exit"):
            break

        run_once(user_id, query)
        print()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Event Recommendation Agent CLI")
    parser.add_argument("user_id", nargs="?", help="User id, e.g. u01 (omit for interactive mode)")
    parser.add_argument("query", nargs="?", help="Natural-language query, e.g. 'find me a concert tonight'")
    parser.add_argument("-v", "--verbose", action="store_true", help="Show full per-node logging output")
    parser.add_argument(
        "--no-llm", action="store_true",
        help="Skip the Bedrock tool-calling path and use rule-based parsing (same as EVENT_AGENT_LLM=off)",
    )
    parser.add_argument(
        "--transcript", action="store_true",
        help="Print the full message exchange with Claude on Bedrock (one-shot mode only)",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    if not args.verbose:
        logging.disable(logging.CRITICAL)
    if args.no_llm:
        os.environ["EVENT_AGENT_LLM"] = "off"

    if args.user_id and args.query:
        run_once(args.user_id, args.query, transcript=args.transcript)
    elif not args.user_id and not args.query:
        run_interactive()
    else:
        parser.error("provide both a user id and a query, or neither for interactive mode")


if __name__ == "__main__":
    sys.exit(main())
