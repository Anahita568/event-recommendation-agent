# Event Recommendation Agent

> **What this is:** I started this alongside my employer's AI training sandbox while
> studying for the AWS AI certification, then kept extending it on my own to learn
> agentic AI fundamentals hands-on: tool calling, guardrails, fallback design, and
> LangGraph. It's a learning project, not a production system, and the scope reflects
> that.

A LangGraph agent that turns a request like *"find me a concert next week"*
into ranked event recommendations, using the user's genre preferences,
budget, and attendance history.

- **Claude on Amazon Bedrock drives the tools** when credentials are present: it decides whether to look up the user, what to search for, and when it's done. Arguments are schema-validated, tool errors go back to the model, and the loop is bounded.
- **A deterministic path runs without AWS**: keyword parsing and a fixed tool sequence. No cloud call, no key needed.
- **Fallbacks are LangGraph conditional edges**, one node per fallback, and a test asserts the exact edge set.
- **An empty search relaxes one constraint at a time** (dates, then budget, then genre) instead of throwing the request away.
- **Ranking is personalised**: genres the user attended are boosted, genres they skipped are penalised, and every recommendation carries a one-line reason.
- Every tool call and model round-trip is timed and logged. All data is synthetic (`data/*.json`).

## [Try it in your browser](https://claude.ai/code/artifact/c59cd02b-9945-405e-9bd7-c8e2a4c8158e)

![Box Office Agent screenshot](docs/events-agent.png)

Pick a patron, type a request, and expand the trace to see each edge the
graph took. It's a JavaScript port of the deterministic path against the
same dataset, not the Python backend itself.

Built with Claude Code as an AI coding assistant.

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

pytest                                             # 59 tests, no AWS needed
python -m evals.run_evals --path deterministic     # 37 eval cases, offline (see Evals)
python3 demo.py u01 "find me a concert next week"  # one query (-v for per-node logs)
python3 demo.py                                    # interactive mode
```

```
$ python3 demo.py u01 "find me a concert next week"

Parsed intent: {'genres': ['Music'], 'date_from': '2026-09-21', 'date_to': '2026-09-27'}
Fallbacks used: ['alternative_source_fallback']
  alternative_source_fallback: drop_dates

Recommendations:
  - Indie Music Live (Music) at Skyline Rooftop on 2027-01-16 — $15  [popularity 74, browsed 1 Music, skipped 1 Music]
  - Local Music Tour Stop (Music) at Downtown Pavilion on 2026-12-12 — $0  [popularity 72, browsed 1 Music, skipped 1 Music]
  ...
```

Nothing matched that exact week, so the agent kept the genre and dropped the
dates rather than returning trending events of any kind.

### Using Bedrock

The repo ships **no AWS credentials**, on purpose. To run the LLM path,
bring your own: enable Bedrock and a Claude model in your account, then
configure a profile (`aws configure --profile <name>`) or export access keys.

```bash
export AWS_PROFILE=<name>
export AWS_REGION=us-east-1
export BEDROCK_MODEL_ID=anthropic.claude-opus-5   # any Claude model enabled in your account
python3 demo.py --transcript u01 "comedy tonight" # prints the full exchange with the model
```

`make_client()` (`agent/tool_calling.py`) picks the right SDK client from
`BEDROCK_MODEL_ID`'s format: short ids (`anthropic.claude-opus-5`) go through
the newer Messages-API `bedrock-mantle` endpoint; dated, region-prefixed ids
(`us.anthropic.claude-sonnet-4-5-20250929-v1:0`) go through the legacy
`InvokeModel` endpoint. Both need a one-time per-model agreement accepted in
your AWS account before the first call succeeds:

```bash
TOKEN=$(aws bedrock list-foundation-model-agreement-offers \
  --model-id <model-id-without-region-prefix> --query "offers[0].offerToken" --output text)
aws bedrock create-foundation-model-agreement \
  --model-id <model-id-without-region-prefix> --offer-token "$TOKEN"
```

Set `EVENT_AGENT_LLM=off` to force the deterministic path even with
credentials. Never commit credentials; tests and the deterministic path
need none.

**A note on the LLM path.** This now runs against live Bedrock. `anthropic.claude-opus-5`
turned out to be gated behind AWS's `bedrock-mantle` early-access program (not
self-service — the account needs allowlisting by `bedrock-ant-eap@amazon.com`
even after the model agreement and IAM permissions are correctly set up), so
the verified transcripts below use the legacy endpoint with Claude Sonnet 4.5
instead. Swap `BEDROCK_MODEL_ID` back to `anthropic.claude-opus-5` once EAP
access clears. A real transcript, keyword search with zero hits triggering the
date-relaxation fallback:

```
$ python3 demo.py --transcript u01 "comedy tonight"

Execution path: llm_tools (Claude on Bedrock chose the tool calls)
Model summary: I searched for Comedy events tonight (2026-09-17) but found no matches.
Fallbacks used: ['alternative_source_fallback']
  alternative_source_fallback: drop_dates
```

And the golden path, a genre + date request resolved in a single search with
real hits, no fallback needed:

```
$ python3 demo.py --transcript u02 "any sports events next month"

Execution path: llm_tools (Claude on Bedrock chose the tool calls)
Model summary: I searched for Sports events in October 2026.

Recommendations:
  - Grand Sports League Night (Sports) at Maple Street Theater on 2026-10-08 — $25  [popularity 68]
  - Urban Sports Tournament (Sports) at The Underground on 2026-10-22 — $75  [popularity 68]
  ...
```

`tests/test_tool_calling.py` still covers the loop's edge cases (bad
arguments, tool exceptions, parallel calls, unknown tools, iteration cap)
against a scripted fake client, since those failure modes are impractical to
trigger against a live model on demand.

## How it works

```
                            ┌─────────┐
                            │  start  │  Bedrock available? → execution_path
                            └────┬────┘
                   ┌─────────────┴──────────────┐
                   ▼                            ▼
         ┌──────────────────┐  failed   ┌──────────────────┐
         │  llm_tool_loop   │──────────►│   parse_query    │  rule-based genre + date
         │  Claude picks &  │           └────────┬─────────┘
         │  calls the tools │                    ▼
         └──┬───────────┬───┘           ┌──────────────────┐
         ok │     empty │               │ fetch_preferences│
            │           │               └───┬──────────┬───┘
            │           │            failed │          │ ok
            │           │                   ▼          │
            │           │  ┌────────────────────────┐  │
            │           │  │ graceful_degradation_  │  │  Fallback A: default profile
            │           │  │ fallback               │  │
            │           │  └───────────┬────────────┘  │
            │           │              ▼               ▼
            │           │            ┌────────────────────┐
            │           │            │   search_events    │
            │           │            └───┬────────────┬───┘
            │           │  empty/failed  │            │ ok
            │           ▼                ▼            │
            │  ┌────────────────────────────┐         │
            │  │ alternative_source_fallback│         │  Fallback B: drop dates → drop budget → trending
            │  └─────────────┬──────────────┘         │
            │                ▼                        ▼
            │          ┌────────────────────────────────┐
            └─────────►│  rank_events                   │  popularity + attendance history
                       └───────┬─────────────────────┬──┘
                        failed │                     │ ok
                               ▼                     │
                  ┌─────────────────────────┐        │
                  │ partial_results_fallback│        │  Fallback C: top 10 unranked
                  └────────────┬────────────┘        │
                               ▼                     ▼
                             ┌────────────────────────┐
                             │     format_output      │ ──► END
                             └────────────────────────┘
```

Each labelled arrow is a LangGraph conditional edge. A failing node records
its error in the state, and the edge routes to the matching fallback node.

**LLM path.** Claude gets two tools, `fetch_user_preferences` (no arguments,
so it can only see the current user) and `search_events`, and decides what to
call. Three guardrails:

1. **Inputs are validated** against a schema before any tool runs.
2. **Errors go back to the model** as error results, so it can adjust instead of crashing.
3. **The loop is capped.** If it stalls or finishes without a search, the graph reruns the request on the deterministic path.

Claude only plans and searches. Ranking and fallbacks are plain code shared by both paths.

**Ranking.** Popularity score, adjusted by the user's history in the same
genre: +8 per attended event, +3 per browsed, −5 per skipped. Already-attended
events are dropped. Each result carries its score and a reason, such as
`popularity 64, attended 3 Sports`.

## Tests

```bash
pytest -v
```

| File | Covers |
|---|---|
| `test_agent.py` | exact conditional-edge set, every fallback route, LLM path success / empty / failover / skipped |
| `test_tool_calling.py` | the LLM loop against a scripted fake client: bad arguments, tool exceptions, parallel calls, unknown tools, iteration cap, API failure |
| `test_fallbacks.py` | each fallback, including every relaxation stage of Fallback B |
| `test_tools.py` | search filters and the personalised ranker |
| `test_query_parser.py` | keyword genre and date extraction |

## Evals

Tests check that the code does what it should. Evals measure how well the
agent does on realistic requests, including ones it isn't built to handle.
They live in `evals/` and are not part of `pytest`, so the test suite stays
free and offline.

```bash
python -m evals.run_evals --path deterministic   # offline, no AWS calls, ~1s
python -m evals.run_evals --path llm --runs 5    # Bedrock only, each case 5 times
python -m evals.run_evals                         # deterministic, plus LLM if credentials are present
python -m evals.run_evals --category adversarial --case s01   # filter
```

The default `auto` mode calls Bedrock whenever AWS credentials are found,
which costs money. Use `--path deterministic` to stay offline.

**Dataset.** `evals/cases.jsonl` has 37 labeled cases over the synthetic
users and events:

| Category | n | What it probes |
|---|---|---|
| straightforward | 9 | plain genre + date requests |
| phrasing | 9 | synonyms ("improv", "flick", "kiddos"), typos, weekday names, "in October", stated budgets ("under $15", "free") |
| ambiguous | 5 | "a show", "game night", "something fun"; any reasonable reading passes |
| fallback | 6 | unknown user (Fallback A), relaxation (Fallback B), and injected tool failures for the trending stage and Fallback C |
| adversarial | 6 | asking for another user's history, a false identity claim, price and instruction injection, a nonexistent tool, fake system tags |
| out_of_scope | 2 | weather, flights: only safety and grounding are scored |

Labels describe the correct behaviour on either path, not whatever the
current code does. "Today" is pinned to 2026-09-17 while the evals run, so
relative dates keep meaning the same thing as the calendar moves. The
harness pins the clock, picks the path and injects faults by patching
around `invoke()`; agent code is untouched.

**Metrics.** Every check is deterministic; there is no LLM judge.

| Check | Passes when |
|---|---|
| `intent_genre`, `intent_dates` | the genres and dates actually searched match the label. On the LLM path this is the model's own `search_events` arguments, so a run that fell back to the deterministic parser isn't credited to the model |
| `fallback` | the fallbacks that fired, and the relaxation stage, match the label |
| `constraints` | every recommendation is a real catalog event, in the expected genre, within the user's budget and any stated price, not already attended, in the requested dates, and not in the past. A relaxation stage waives only the constraint it relaxed |
| `tool_use` (LLM) | the loop finished (or handed over when it should), made a successful search, stayed under the 6-iteration cap, called no unknown tools, and fetched preferences when no genre was named |
| `safety` (LLM) | the model's text contains none of the case's forbidden strings (other users' names and event ids, system prompt fragments) |

The report also covers latency (p50/p95), and on the LLM path model calls,
tool calls, tokens and an estimated cost. Cost uses Anthropic list prices
from `evals/pricing.py`; Bedrock bills separately, so pass `--price-in` and
`--price-out` (USD per million tokens) for your real rates. With `--runs N`
each LLM case repeats and the report adds per-case pass rates and flags
flaky cases. Each run writes `evals/results/<timestamp>_<path>[_<model>].json`
(gitignored) with the config, summaries and every run's checks and output.

**Deterministic baseline**, 26 of 37 cases passing:

| Category | Pass |
|---|---|
| straightforward | 8/9 |
| phrasing | 0/9 |
| ambiguous | 4/5 |
| fallback | 6/6 |
| adversarial | 6/6 |
| out_of_scope | 2/2 |

The failures are real gaps, left in place:

- The keyword parser misses synonyms, typos, weekday names and month names.
- Stated budgets ("under $15", "free") are ignored.
- Dropping the dates can surface events that already happened.

**Known gaps in coverage.**

- The `drop_dates_and_budget` stage can't be reached with this data, because every genre has events under $15.
- The trending stage and Fallback C are only reached through injected faults.
