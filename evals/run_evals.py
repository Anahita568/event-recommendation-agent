#!/usr/bin/env python3
"""Run the eval suite and report how well each agent path does.

Usage:
    python -m evals.run_evals                         deterministic, plus LLM when Bedrock is available
    python -m evals.run_evals --path deterministic    offline only, no AWS calls
    python -m evals.run_evals --path llm --runs 5     LLM path only, each case 5 times
    python -m evals.run_evals --case s01 --case p03   a subset of cases
    python -m evals.run_evals --category phrasing

Results print as tables and are written to evals/results/<timestamp>_<paths>.json.
The exit code is 0 even when cases fail: evals measure, they don't gate.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import statistics
import subprocess
import sys
from datetime import date, datetime
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent import tool_calling  # noqa: E402
from evals import harness, pricing, scoring  # noqa: E402

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


def run_path(cases: list[dict], path: str, runs: int, today: date, rates) -> list[dict]:
    records = []
    progress = sys.stderr.isatty()
    total = len(cases) * runs
    for run in range(1, runs + 1):
        for i, case in enumerate(cases, 1):
            if progress:
                done = (run - 1) * len(cases) + i
                print(f"\r  {path}: {done}/{total} {case['id']:<6}", end="", file=sys.stderr, flush=True)
            result, llm = harness.run_case(case, path, today)
            if llm is not None:
                llm["cost_usd"] = pricing.cost_usd(llm["usage"], rates)
            records.append({
                "case_id": case["id"],
                "category": case["category"],
                "path": path,
                "run": run,
                "latency_ms": result["total_latency_ms"],
                **scoring.score(case, path, result, today, llm),
                "llm": llm,
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


def _verdict(passed: bool) -> str:
    return scoring.PASS if passed else scoring.FAIL


def _percentile(values: list[float], pct: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round(pct / 100 * (len(ordered) - 1)))]


def per_case(records: list[dict]) -> dict:
    """Pass counts per (path, case) across repeated runs, with a flaky flag."""
    out: dict[str, dict] = {}
    for r in records:
        entry = out.setdefault(f"{r['path']}:{r['case_id']}", {
            "path": r["path"], "case_id": r["case_id"], "category": r["category"], "passed": 0, "runs": 0,
        })
        entry["runs"] += 1
        entry["passed"] += r["passed"]
    for entry in out.values():
        entry["pass_rate"] = entry["passed"] / entry["runs"]
        entry["flaky"] = 0 < entry["passed"] < entry["runs"]
    return out


def summarize(records: list[dict], paths: list[str]) -> dict:
    cases = per_case(records)
    summary = {}
    for path in paths:
        rows = [r for r in records if r["path"] == path]
        path_cases = [c for c in cases.values() if c["path"] == path]
        latencies = [r["latency_ms"] for r in rows]

        metrics = {name: _rate([r["checks"][name]["status"] for r in rows]) for name in scoring.CHECKS}
        metrics["overall"] = _rate([_verdict(r["passed"]) for r in rows])
        metrics["every_run"] = _rate([_verdict(c["passed"] == c["runs"]) for c in path_cases])

        by_category = {
            cat: _rate([_verdict(r["passed"]) for r in rows if r["category"] == cat]) for cat in CATEGORIES
        }

        constraint_rates = {}
        for name in scoring.CONSTRAINTS:
            applicable = [r for r in rows if name in r["constraint_violations"]]
            constraint_rates[name] = _rate(
                [_verdict(not r["constraint_violations"][name]) for r in applicable]
            )

        llm_rows = [r["llm"] for r in rows if r["llm"] is not None]
        llm = None
        if llm_rows:
            costs = [x["cost_usd"] for x in llm_rows]
            llm = {
                "runs": len(llm_rows),
                "completed_rate": sum(r["result"]["execution_path"] == "llm_tools" for r in rows) / len(rows),
                "mean_iterations": statistics.mean(x["iterations"] for x in llm_rows),
                "mean_tool_calls": statistics.mean(len(x["tool_calls"]) for x in llm_rows),
                "tokens": {k: sum(x["usage"][k] for x in llm_rows) for k in llm_rows[0]["usage"]},
                "mean_input_tokens": statistics.mean(x["usage"]["input_tokens"] for x in llm_rows),
                "mean_output_tokens": statistics.mean(x["usage"]["output_tokens"] for x in llm_rows),
                "total_cost_usd": None if None in costs else sum(costs),
            }

        summary[path] = {
            "metrics": metrics,
            "by_category": by_category,
            "constraints": constraint_rates,
            "latency_ms": {
                "p50": statistics.median(latencies),
                "p95": _percentile(latencies, 95),
                "max": max(latencies),
            },
            "llm": llm,
            "flaky_cases": sorted(c["case_id"] for c in path_cases if c["flaky"]),
        }
    return summary, cases


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


def _llm_cell(summary: dict, path: str, fmt) -> str:
    llm = summary[path]["llm"]
    return fmt(llm) if llm else "n/a"


def print_report(summary: dict, cases: dict, records: list[dict], paths: list[str], runs: int) -> None:
    header = ["", *paths]
    rows = [[m, *(_fmt_rate(summary[p]["metrics"][m]) for p in paths)] for m in (*scoring.CHECKS, "overall")]
    if runs > 1:
        rows.append([f"cases passing all {runs} runs", *(_fmt_rate(summary[p]["metrics"]["every_run"]) for p in paths)])
    rows.append(["latency p50 / p95", *(
        f"{summary[p]['latency_ms']['p50']:.0f} / {summary[p]['latency_ms']['p95']:.0f} ms" for p in paths
    )])
    if "llm" in paths:
        rows += [
            ["LLM loop completed", *(_llm_cell(summary, p, lambda x: f"{x['completed_rate']:.0%}") for p in paths)],
            ["model calls / tool calls (mean)", *(_llm_cell(
                summary, p, lambda x: f"{x['mean_iterations']:.1f} / {x['mean_tool_calls']:.1f}") for p in paths)],
            ["tokens in / out (mean per run)", *(_llm_cell(
                summary, p, lambda x: f"{x['mean_input_tokens']:.0f} / {x['mean_output_tokens']:.0f}") for p in paths)],
            ["est. cost (total)", *(_llm_cell(
                summary, p, lambda x: "unknown model" if x["total_cost_usd"] is None else f"${x['total_cost_usd']:.4f}"
            ) for p in paths)],
        ]
    print()
    print(_table("Metrics (pass rate over scored runs)", header, rows))

    print()
    rows = [[c, *(_fmt_rate(summary[p]["by_category"][c]) for p in paths)] for c in CATEGORIES]
    print(_table("Runs passing every check, by category", header, rows))

    print()
    rows = [[c, *(_fmt_rate(summary[p]["constraints"][c]) for p in paths)] for c in scoring.CONSTRAINTS]
    print(_table("Constraint satisfaction (runs where every recommendation complies)", header, rows))

    for path in paths:
        if summary[path]["flaky_cases"]:
            flaky = ", ".join(
                f"{cid} ({cases[f'{path}:{cid}']['passed']}/{cases[f'{path}:{cid}']['runs']})"
                for cid in summary[path]["flaky_cases"]
            )
            print(f"\nFlaky on {path} (passed some runs, not all): {flaky}")

    failures = [r for r in records if not r["passed"]]
    print(f"\nFailing runs ({len(failures)}):")
    for r in failures:
        run = f" run {r['run']}" if runs > 1 and r["path"] == "llm" else ""
        print(f"  [{r['path']}{run}] {r['case_id']} ({r['category']})")
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


def write_results(config: dict, summary: dict, cases: dict, records: list[dict], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    name = "+".join(config["paths"])
    if config.get("model"):
        name += "_" + config["model"].replace(":", "-").replace("/", "-")
    out = out_dir / f"{stamp}_{name}.json"
    payload = {"config": config, "summary": summary, "cases": list(cases.values()), "runs": records}
    out.write_text(json.dumps(payload, indent=2, default=str))
    return out


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the event agent eval suite")
    parser.add_argument(
        "--path", choices=["auto", "deterministic", "llm"], default="auto",
        help="auto: deterministic, plus LLM when Bedrock credentials are present (default)",
    )
    parser.add_argument("--runs", type=int, default=1,
                        help="Repeat each LLM-path case N times (the deterministic path always runs once)")
    parser.add_argument("--case", action="append", default=[], help="Only run this case id (repeatable)")
    parser.add_argument("--category", action="append", default=[], choices=CATEGORIES,
                        help="Only run this category (repeatable)")
    parser.add_argument("--today", type=date.fromisoformat, default=harness.PINNED_TODAY,
                        help=f"Pinned 'today' for the agent (default {harness.PINNED_TODAY}); labels assume the default")
    parser.add_argument("--price-in", type=float, help="Override input price, USD per million tokens")
    parser.add_argument("--price-out", type=float, help="Override output price, USD per million tokens")
    parser.add_argument("--out-dir", type=Path, default=RESULTS_DIR, help="Where to write the JSON results")
    parser.add_argument("-v", "--verbose", action="store_true", help="Show the agent's own logging")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if args.runs < 1:
        parser.error("--runs must be at least 1")
    if (args.price_in is None) != (args.price_out is None):
        parser.error("pass --price-in and --price-out together")
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

    llm_ok, llm_reason = tool_calling.bedrock_available()
    if args.path == "llm" and not llm_ok:
        print(f"LLM path unavailable: {llm_reason}", file=sys.stderr)
        return 2
    paths = {"deterministic": ["deterministic"], "llm": ["llm"]}.get(args.path) or (
        ["deterministic", "llm"] if llm_ok else ["deterministic"]
    )
    if args.path == "auto" and not llm_ok:
        print(f"Skipping the LLM path: {llm_reason}", file=sys.stderr)

    model = tool_calling.BEDROCK_MODEL_ID if "llm" in paths else None
    rates = (args.price_in, args.price_out) if args.price_in is not None else (model and pricing.rates_for(model))

    print(f"Running {len(cases)} case(s) on {', '.join(paths)} with today={args.today}"
          + (f", model={model}, runs={args.runs}" if model else ""), file=sys.stderr)
    records = []
    for path in paths:
        records += run_path(cases, path, args.runs if path == "llm" else 1, args.today, rates)

    summary, per_case_results = summarize(records, paths)
    print_report(summary, per_case_results, records, paths, args.runs)
    if model:
        source = "your --price-in/--price-out" if args.price_in is not None else "Anthropic list prices"
        print(f"\nCost is an estimate at {source}; Bedrock bills separately.")

    config = {
        "paths": paths,
        "model": model,
        "aws_region": os.environ.get("AWS_REGION") if model else None,
        "runs": args.runs,
        "today": args.today.isoformat(),
        "price_per_mtok": {"input": rates[0], "output": rates[1]} if rates else None,
        "cases": [c["id"] for c in cases],
        "git_sha": _git_sha(),
        "started_at": datetime.now().isoformat(timespec="seconds"),
    }
    out = write_results(config, summary, per_case_results, records, args.out_dir)
    print(f"\nResults written to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
