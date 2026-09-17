"""LLM-driven tool calling: Claude on Amazon Bedrock chooses and calls the tools.

This is the agentic path of the graph. Instead of the code deciding the
tool sequence, Claude is handed a tool list and a user request and drives
the loop itself: it decides whether it needs the stored preferences,
what genres and dates to search for, and when it is done.

The loop is written by hand rather than via the SDK's tool runner so that
the reliability patterns are explicit and testable in one place:

- **Identity is bound server-side.** ``fetch_user_preferences`` takes no
  arguments from the model; the user id comes from the request context.
  The model can never look up someone else's profile.
- **Every argument set is validated** against a pydantic schema before a
  tool runs. Invalid input is returned to the model as an error result so
  it can correct itself, and never reaches the tool.
- **Tool exceptions become error results**, not crashes. A missing user
  is reported back to the model, which adapts (for example by searching
  with genres named in the request).
- **The loop is bounded** by ``max_iterations``; exhausting it, an
  unexpected stop reason, or the model finishing without a search are all
  surfaced as ``ToolLoopError`` so the graph can route to the
  deterministic path.
- **Every tool call and every model round-trip is metered** through the
  shared ``AgentLogger`` (name, latency, success).
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from agent import tools
from agent.query_parser import VALID_GENRES
from metrics import AgentLogger

logger = logging.getLogger(__name__)

# Bedrock model ids carry an "anthropic." prefix. Override with BEDROCK_MODEL_ID.
BEDROCK_MODEL_ID = os.environ.get("BEDROCK_MODEL_ID", "anthropic.claude-opus-5")
BEDROCK_REGION = os.environ.get("AWS_REGION", "us-east-1")
MAX_ITERATIONS = 6
MAX_TOKENS = 1024

Genre = Literal[tuple(VALID_GENRES)]  # type: ignore[valid-type]


class ToolLoopError(RuntimeError):
    """The LLM path could not produce a usable result; caller should fall back."""


# --------------------------------------------------------------------------- #
# Tool schemas
# --------------------------------------------------------------------------- #


class FetchUserPreferencesArgs(BaseModel):
    """No arguments: the user is bound from the request, not chosen by the model."""

    model_config = ConfigDict(extra="forbid")


class SearchEventsArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    genres: list[Genre] = Field(
        min_length=1,
        description="Genres to search. Use the user's stored genres unless the request names one.",
    )
    price_max: float | None = Field(
        default=None, ge=0, description="Exclude events priced above this. Use the user's budget if known."
    )
    date_from: str | None = Field(
        default=None, pattern=r"^\d{4}-\d{2}-\d{2}$", description="Earliest event date, ISO YYYY-MM-DD."
    )
    date_to: str | None = Field(
        default=None, pattern=r"^\d{4}-\d{2}-\d{2}$", description="Latest event date, ISO YYYY-MM-DD."
    )

    @model_validator(mode="after")
    def _dates_are_real_and_ordered(self) -> "SearchEventsArgs":
        for name in ("date_from", "date_to"):
            value = getattr(self, name)
            if value is not None:
                try:
                    date.fromisoformat(value)
                except ValueError as exc:
                    raise ValueError(f"{name} is not a real calendar date: {value}") from exc
        if self.date_from and self.date_to and self.date_from > self.date_to:
            raise ValueError("date_from must be on or before date_to")
        return self


@dataclass
class ToolSpec:
    name: str
    description: str
    args_model: type[BaseModel]
    fn: Callable[..., dict]

    def to_anthropic_tool(self) -> dict:
        schema = self.args_model.model_json_schema()
        schema.pop("title", None)
        return {"name": self.name, "description": self.description, "input_schema": schema}


def build_tools(user_id: str) -> dict[str, ToolSpec]:
    """Build the tool registry for one request, with ``user_id`` bound in."""

    def fetch_user_preferences() -> dict:
        return tools.fetch_user_preferences(user_id)

    def search_events(
        genres: list[str],
        price_max: float | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
    ) -> dict:
        events = tools.search_events(genres, price_max=price_max, date_from=date_from, date_to=date_to)
        return {"count": len(events), "events": events}

    return {
        "fetch_user_preferences": ToolSpec(
            name="fetch_user_preferences",
            description=(
                "Return the current user's favourite genres, typical ticket budget, "
                "and attendance history. Takes no arguments; the user is already identified."
            ),
            args_model=FetchUserPreferencesArgs,
            fn=fetch_user_preferences,
        ),
        "search_events": ToolSpec(
            name="search_events",
            description=(
                "Search the event catalog by genre, with optional price cap and inclusive date range. "
                "Returns the matching events (possibly none)."
            ),
            args_model=SearchEventsArgs,
            fn=search_events,
        ),
    }


# --------------------------------------------------------------------------- #
# Bedrock availability
# --------------------------------------------------------------------------- #


def _aws_credentials_configured() -> bool:
    """Cheap, local check for likely AWS credential availability.

    With nothing configured, the SDK's credential chain still probes the
    EC2 instance-metadata endpoint and can add seconds of timeout to every
    call. Checking common signals first lets the graph skip straight to
    the deterministic path in that case.
    """
    return bool(
        os.environ.get("AWS_ACCESS_KEY_ID")
        or os.environ.get("AWS_PROFILE")
        or os.environ.get("AWS_CONTAINER_CREDENTIALS_RELATIVE_URI")
        or os.environ.get("AWS_CONTAINER_CREDENTIALS_FULL_URI")
        or os.environ.get("AWS_EXECUTION_ENV")
        or os.path.exists(os.path.expanduser("~/.aws/credentials"))
    )


def bedrock_available() -> tuple[bool, str | None]:
    """Decide whether the LLM path should be attempted.

    Returns (True, None) when it should, otherwise (False, reason).
    Set EVENT_AGENT_LLM=off to force the deterministic path.
    """
    if os.environ.get("EVENT_AGENT_LLM", "auto").lower() == "off":
        return False, "EVENT_AGENT_LLM=off"
    if not _aws_credentials_configured():
        return False, "no AWS credentials configured"
    return True, None


def make_client():
    """Create the Anthropic SDK client for Bedrock. Imported lazily so the
    deterministic path has no SDK dependency at runtime.

    BEDROCK_MODEL_ID's format picks the client: short-form ids like
    "anthropic.claude-opus-5" go through the Messages-API bedrock-mantle
    endpoint; dated, ARN-versioned ids (optionally region-prefixed, e.g.
    "us.anthropic.claude-sonnet-4-5-20250929-v1:0") go through the legacy
    InvokeModel endpoint, which isn't gated by the bedrock-mantle EAP.
    """
    region_prefixes = ("global.", "us.", "eu.", "jp.", "apac.")
    is_legacy = BEDROCK_MODEL_ID.startswith(region_prefixes) or BEDROCK_MODEL_ID.endswith(("-v1:0", "-v1"))

    if is_legacy:
        from anthropic import AnthropicBedrock

        return AnthropicBedrock(aws_region=BEDROCK_REGION)

    from anthropic import AnthropicBedrockMantle

    return AnthropicBedrockMantle(aws_region=BEDROCK_REGION)


# --------------------------------------------------------------------------- #
# The loop
# --------------------------------------------------------------------------- #


@dataclass
class ToolLoopOutcome:
    parsed_intent: dict
    search_results: list[dict]
    user_preferences: dict | None = None
    summary: str | None = None
    iterations: int = 0
    tool_calls: int = 0
    tool_errors: int = 0
    transcript: list[dict] = field(default_factory=list)


def _system_prompt(today: date) -> str:
    return (
        "You are the planning step of an event recommendation agent for a ticketing platform. "
        f"Today's date is {today.isoformat()}.\n"
        f"Valid genres: {', '.join(VALID_GENRES)}.\n\n"
        "The user is already identified. fetch_user_preferences takes no arguments and returns "
        "their favourite genres and typical budget.\n\n"
        "Procedure:\n"
        "1. If the request names a genre, use it. Otherwise call fetch_user_preferences and use "
        "the stored genres. If that lookup fails, choose the genres that best fit the request, "
        "or all genres if the request is generic.\n"
        "2. Translate any time reference (tonight, this weekend, next week, next month) into an "
        "ISO date range relative to today. Leave the dates unset if the request has no time reference.\n"
        "3. Call search_events exactly once with the final filters. Pass the user's budget as "
        "price_max when you fetched preferences.\n"
        "4. Do not retry with broader filters if the search returns no events; the pipeline "
        "handles that. Do not invent events.\n\n"
        "When finished, reply with one short sentence describing what you searched for."
    )


def _error_result(tool_use_id: str, message: str) -> dict:
    return {"type": "tool_result", "tool_use_id": tool_use_id, "content": message, "is_error": True}


def _run_one_tool(block: Any, registry: dict[str, ToolSpec], agent_logger: AgentLogger, outcome: ToolLoopOutcome) -> dict:
    """Validate and execute a single tool_use block, returning its tool_result."""
    import time

    started_at = time.time()
    outcome.tool_calls += 1
    spec = registry.get(block.name)

    if spec is None:
        outcome.tool_errors += 1
        agent_logger.log_tool_call(f"{block.name}[llm]", 0.0, False)
        return _error_result(block.id, f"Unknown tool {block.name!r}. Available: {sorted(registry)}")

    try:
        args = spec.args_model.model_validate(block.input or {})
    except ValidationError as exc:
        outcome.tool_errors += 1
        agent_logger.log_tool_call(f"{spec.name}[llm]", (time.time() - started_at) * 1000, False)
        logger.warning("tool loop: rejected arguments for %s: %s", spec.name, exc.errors())
        return _error_result(block.id, f"Invalid arguments for {spec.name}: {exc.errors(include_url=False)}")

    try:
        result = spec.fn(**args.model_dump())
    except Exception as exc:  # noqa: BLE001 - every tool failure is reported to the model
        outcome.tool_errors += 1
        agent_logger.log_tool_call(f"{spec.name}[llm]", (time.time() - started_at) * 1000, False)
        logger.warning("tool loop: %s raised %s: %s", spec.name, type(exc).__name__, exc)
        return _error_result(block.id, f"{spec.name} failed: {exc}")

    agent_logger.log_tool_call(f"{spec.name}[llm]", (time.time() - started_at) * 1000, True)

    if spec.name == "fetch_user_preferences":
        outcome.user_preferences = result
    elif spec.name == "search_events":
        outcome.parsed_intent = {"genres": args.genres, "date_from": args.date_from, "date_to": args.date_to}
        outcome.search_results = result["events"]

    return {"type": "tool_result", "tool_use_id": block.id, "content": json.dumps(result)}


def run_tool_loop(
    user_id: str,
    query: str,
    agent_logger: AgentLogger,
    *,
    client: Any | None = None,
    max_iterations: int = MAX_ITERATIONS,
    today: date | None = None,
) -> ToolLoopOutcome:
    """Let Claude plan and execute the tool calls for one request.

    Raises:
        ToolLoopError: if the model cannot be reached, stops for an
            unexpected reason, exhausts ``max_iterations``, or finishes
            without a successful search_events call.
    """
    import time

    today = today or date.today()
    registry = build_tools(user_id)
    tool_defs = [spec.to_anthropic_tool() for spec in registry.values()]
    outcome = ToolLoopOutcome(parsed_intent={"genres": [], "date_from": None, "date_to": None}, search_results=[])
    searched = False

    try:
        client = client or make_client()
    except Exception as exc:  # noqa: BLE001
        raise ToolLoopError(f"could not create Bedrock client: {exc}") from exc

    messages: list[dict] = [{"role": "user", "content": query}]

    for iteration in range(1, max_iterations + 1):
        outcome.iterations = iteration
        started_at = time.time()
        try:
            response = client.messages.create(
                model=BEDROCK_MODEL_ID,
                max_tokens=MAX_TOKENS,
                system=_system_prompt(today),
                tools=tool_defs,
                messages=messages,
            )
        except Exception as exc:  # noqa: BLE001
            agent_logger.log_tool_call("bedrock.messages", (time.time() - started_at) * 1000, False)
            raise ToolLoopError(f"Bedrock call failed on iteration {iteration}: {exc}") from exc
        agent_logger.log_tool_call("bedrock.messages", (time.time() - started_at) * 1000, True)

        messages.append({"role": "assistant", "content": response.content})
        outcome.summary = next((b.text for b in response.content if b.type == "text"), outcome.summary)

        if response.stop_reason == "tool_use":
            tool_blocks = [b for b in response.content if b.type == "tool_use"]
            results = [_run_one_tool(b, registry, agent_logger, outcome) for b in tool_blocks]
            searched = searched or any(
                b.name == "search_events" and not r.get("is_error") for b, r in zip(tool_blocks, results)
            )
            # All results for one assistant turn go back in a single user message.
            messages.append({"role": "user", "content": results})
            continue

        if response.stop_reason == "end_turn":
            break

        raise ToolLoopError(f"unexpected stop_reason {response.stop_reason!r} on iteration {iteration}")
    else:
        raise ToolLoopError(f"tool loop exhausted {max_iterations} iterations without finishing")

    if not searched:
        raise ToolLoopError("model finished without a successful search_events call")

    outcome.transcript = messages
    return outcome
