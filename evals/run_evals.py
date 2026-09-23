#!/usr/bin/env python3
"""Run the eval suite and report how well each agent path does.

Usage:
    python -m evals.run_evals                         deterministic path (plus LLM when available)
    python -m evals.run_evals --path deterministic    offline only
    python -m evals.run_evals --case s01 --case p03   a subset of cases
    python -m evals.run_evals --category phrasing

Results print as tables and are written to evals/results/<timestamp>_<paths>.json.
The exit code is 0 even when cases fail: evals measure, they don't gate.
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import subprocess
import sys
from collections import defaultdict
from datetime import date, datetime
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evals import harness, scoring  # noqa: E402

EVALS_DIR = Path(__file__).resolve().parent
CASES_PATH = EVALS_DIR / "cases.jsonl"
RESULTS_DIR = EVALS_DIR / "results"

CATEGORIES = ("straightforward", "phrasing", "ambiguous", "fallback", "adversarial", "out_of_scope")
EXPECT_KEYS = {"genres", "dates", "fallback", "fallback_llm", "constraints", "max_price", "must_not_mention", "llm_completes"}


# --------------------------------------------------------------------------- #
# Cases
# --------------------------------------------------------------------------- #


def load_cases(path: Path = CASES_PATH) -> list[dict]:
    """Load and validate the dataset, failing loudly on a malformed case."""
    cases, seen = [], set()
    for lineno, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        case = json.loads(line)
        where = f"{path.name}:{lineno} ({case.get('id')})"
        for key in ("id", "category", "user_id", "request", "expect"):
            if key not in case:
                raise ValueError(f"{where}: missing {key!r}")
        if case["id"] in seen:
            raise ValueError(f"{where}: duplicate id")
        if case["category"] not in CATEGORIES:
            raise ValueError(f"{where}: unknown category {case['category']!r}")
        if case.get("fault") and case["fault"] not in harness.FAULTS:
            raise ValueError(f"{where}: unknown fault {case['fault']!r}")
        if unknown := set(case["expect"]) - EXPECT_KEYS:
            raise ValueError(f"{where}: unknown expect keys {sorted(unknown)}")
        if unknown := set(case["expect"].get("constraints", ())) - set(scoring.CONSTRAINTS):
            raise ValueError(f"{where}: unknown constraints {sorted(unknown)}")
        seen.add(case["id"])
        cases.append(case)
    return cases


# --------------------------------------------------------------------------- #
# Running
# --------------------------------------------------------------------------- #


def run_path(cases: list[dict], path: str, today: date) -> list[dict]:
    records = []
    progress = sys.stderr.isatty()
    for i, case in enumerate(cases, 1):
        if progress:
            print(f"\r  {path}: {i}/{len(cases)} {case['id']:<6}", end="", file=sys.stderr, flush=True)
        result = harness.run_case(case, path, today)
        records.append({
            "case_id": case["id"],
            "category": case["category"],
            "path": path,
            "run": 1,
            "latency_ms": result["total_latency_ms"],
            **scoring.score(case, path, result, today),
            "result": result,
        })
    if progress:
        print(file=sys.stderr)
    return records


# --------------------------------------------------------------------------- #
# Summaries
# --------------------------------------------------------------------------- #


def _rate(statuses: list[str]) -> dict | None:
    scored = [s for s in statuses if s != scoring.NA]
    if not scored:
        return None
    passed = sum(s == scoring.PASS for s in scored)
    return {"passed": passed, "total": len(scored), "rate": passed / len(scored)}


def _percentile(values: list[float], pct: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round(pct / 100 * (len(ordered) - 1)))]


def summarize(records: list[dict], paths: list[str]) -> dict:
    summary = {}
    for path in paths:
        rows = [r for r in records if r["path"] == path]
        latencies = [r["latency_ms"] for r in rows]
        metrics = {
            name: _rate([r["checks"][name]["status"] for r in rows if name in r["checks"]])
            for name in scoring.CHECKS
        }
        metrics["overall"] = _rate([scoring.PASS if r["passed"] else scoring.FAIL for r in rows])

        by_category = {
            cat: _rate([scoring.PASS if r["passed"] else scoring.FAIL for r in rows if r["category"] == cat])
            for cat in CATEGORIES
        }

        constraint_rates = {}
        for name in scoring.CONSTRAINTS:
            applicable = [r for r in rows if name in r["constraint_violations"]]
            constraint_rates[name] = _rate(
                [scoring.FAIL if r["constraint_violations"][name] else scoring.PASS for r in applicable]
            )

        summary[path] = {
            "metrics": metrics,
            "by_category": by_category,
            "constraints": constraint_rates,
            "latency_ms": {
                "p50": statistics.median(latencies),
                "p95": _percentile(latencies, 95),
                "max": max(latencies),
            } if latencies else None,
        }
    return summary


# --------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------- #


def _fmt_rate(rate: dict | None) -> str:
    if rate is None:
        return "n/a"
    return f"{rate['rate']:>4.0%} ({rate['passed']}/{rate['total']})"


def _table(title: str, header: list[str], rows: list[list[str]]) -> str:
    widths = [max(len(str(row[i])) for row in [header, *rows]) for i in range(len(header))]

    def line(row):
        return "  ".join(str(c).ljust(w) for c, w in zip(row, widths))

    return "\n".join([title, line(header), line(["-" * w for w in widths]), *map(line, rows)])


def print_report(summary: dict, records: list[dict], paths: list[str]) -> None:
    header = ["", *paths]
    metric_names = [*scoring.CHECKS, "overall"]
    rows = [[m, *(_fmt_rate(summary[p]["metrics"][m]) for p in paths)] for m in metric_names]
    rows.append(["latency p50 / p95", *(
        f"{summary[p]['latency_ms']['p50']:.0f} / {summary[p]['latency_ms']['p95']:.0f} ms" for p in paths
    )])
    print()
    print(_table("Metrics (pass rate over scored runs)", header, rows))

    print()
    rows = [[c, *(_fmt_rate(summary[p]["by_category"][c]) for p in paths)] for c in CATEGORIES]
    print(_table("Cases passing every check, by category", header, rows))

    print()
    rows = [[c, *(_fmt_rate(summary[p]["constraints"][c]) for p in paths)] for c in scoring.CONSTRAINTS]
    print(_table("Constraint satisfaction (runs where every recommendation complies)", header, rows))

    failures = [r for r in records if not r["passed"]]
    print(f"\nFailing runs ({len(failures)}):")
    for r in failures:
        print(f"  [{r['path']}] {r['case_id']} ({r['category']})")
        for name, check in r["checks"].items():
            if check["status"] == scoring.FAIL:
                print(f"      {name}: {check['detail']}")


def _git_sha() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True,
            cwd=EVALS_DIR,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def write_results(config: dict, summary: dict, records: list[dict], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    out = out_dir / f"{stamp}_{'+'.join(config['paths'])}.json"
    out.write_text(json.dumps({"config": config, "summary": summary, "runs": records}, indent=2, default=str))
    return out


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the event agent eval suite")
    parser.add_argument("--path", choices=["deterministic"], default="deterministic",
                        help="Which agent path to evaluate")
    parser.add_argument("--case", action="append", default=[], help="Only run this case id (repeatable)")
    parser.add_argument("--category", action="append", default=[], choices=CATEGORIES,
                        help="Only run this category (repeatable)")
    parser.add_argument("--today", type=date.fromisoformat, default=harness.PINNED_TODAY,
                        help=f"Pinned 'today' for the agent (default {harness.PINNED_TODAY}); labels assume the default")
    parser.add_argument("--out-dir", type=Path, default=RESULTS_DIR, help="Where to write the JSON results")
    parser.add_argument("-v", "--verbose", action="store_true", help="Show the agent's own logging")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if not args.verbose:
        logging.disable(logging.CRITICAL)

    cases = load_cases()
    if args.case:
        unknown = set(args.case) - {c["id"] for c in cases}
        if unknown:
            print(f"Unknown case id(s): {', '.join(sorted(unknown))}", file=sys.stderr)
            return 2
        cases = [c for c in cases if c["id"] in args.case]
    if args.category:
        cases = [c for c in cases if c["category"] in args.category]
    if not cases:
        print("No cases selected.", file=sys.stderr)
        return 2

    paths = [args.path]
    print(f"Running {len(cases)} case(s) on {', '.join(paths)} with today={args.today}", file=sys.stderr)
    records = []
    for path in paths:
        records += run_path(cases, path, args.today)

    summary = summarize(records, paths)
    print_report(summary, records, paths)

    config = {
        "paths": paths,
        "today": args.today.isoformat(),
        "cases": [c["id"] for c in cases],
        "git_sha": _git_sha(),
        "started_at": datetime.now().isoformat(timespec="seconds"),
    }
    out = write_results(config, summary, records, args.out_dir)
    print(f"\nResults written to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
