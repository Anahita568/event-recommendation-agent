"""Main LangGraph agent for event recommendations.

Two paths converge on the same fallback routing:

    start ──► llm_tool_loop ─────────────────┐   (Claude on Bedrock picks the
      │            │ failed                  │    tools; see agent/tool_calling.py)
      │            ▼                         │
      └──► parse_query ► fetch_preferences   │   (rule-based, no cloud call)
                              │              │
                   ┌──────────┴───┐          │
                   ▼              ▼          │
   graceful_degradation_fallback  │          │
                   └──────► search_events    │
                                  │◄─────────┘
                       ┌──────────┴───┐
                       ▼              ▼
     alternative_source_fallback      │
                       └──────► rank_events
                                      │
                           ┌──────────┴───┐
                           ▼              ▼
         partial_results_fallback         │
                           └──────► format_output ──► END

Every branch above is a LangGraph conditional edge. Nodes never call a
fallback directly: a node that fails records ``state.last_error`` and
the routing function on its outgoing edge sends the run to the matching
fallback node, which is a first-class node in the graph.
"""

import logging
import time
from typing import Literal

from langgraph.graph import END, StateGraph

from agent import fallbacks, query_parser, tool_calling, tools
from agent.state import AgentState, QueryIntent
from metrics import AgentLogger

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

_agent_logger = AgentLogger()


def _log_latency(node_name: str, started_at: float) -> None:
    latency_ms = (time.time() - started_at) * 1000
    logger.info("%s finished in %.2fms", node_name, latency_ms)


def _apply_preferences(state: AgentState, prefs: dict) -> None:
    """Copy a fetch_user_preferences result (from either path) into the state."""
    state.user_preferences.genres = prefs["genres"]
    state.user_preferences.budget = prefs["budget"]
    history = prefs.get("history") or {}
    state.user_preferences.attended = list(history.get("attended", []))
    state.user_preferences.skipped = list(history.get("skipped", []))
    state.user_preferences.browsed = list(history.get("browsed", []))


def _serialize_transcript(messages: list[dict]) -> list[dict]:
    """Turn the SDK content blocks in a tool-loop transcript into plain dicts."""

    def block(b):
        if isinstance(b, dict):
            return b
        if hasattr(b, "model_dump"):
            return b.model_dump(mode="json", exclude_none=True)
        return vars(b)

    return [
        {"role": m["role"], "content": [block(b) for b in m["content"]] if isinstance(m["content"], list) else m["content"]}
        for m in messages
    ]


# --------------------------------------------------------------------------- #
# Nodes
# --------------------------------------------------------------------------- #


def start_node(state: AgentState) -> AgentState:
    started_at = time.time()
    logger.info("start_node: entering for user_id=%s query=%r", state.user_id, state.query)
    available, reason = tool_calling.bedrock_available()
    if available:
        state.execution_path = "llm_tools"
    else:
        state.execution_path = "deterministic"
        state.llm_error = reason
        logger.info("start_node: LLM path skipped (%s)", reason)
    _log_latency("start_node", started_at)
    return state


def llm_tool_loop_node(state: AgentState) -> AgentState:
    """Agentic path: Claude decides which tools to call and with what arguments."""
    started_at = time.time()
    state.last_error = None
    logger.info("llm_tool_loop_node: entering for user_id=%s", state.user_id)
    try:
        outcome = tool_calling.run_tool_loop(state.user_id, state.query, _agent_logger)
        state.parsed_intent = QueryIntent(**outcome.parsed_intent)
        if outcome.user_preferences:
            _apply_preferences(state, outcome.user_preferences)
        state.search_results = outcome.search_results
        state.llm_summary = outcome.summary
        state.llm_transcript = _serialize_transcript(outcome.transcript)
        logger.info(
            "llm_tool_loop_node: done in %d iteration(s), %d tool call(s), %d tool error(s)",
            outcome.iterations, outcome.tool_calls, outcome.tool_errors,
        )
    except Exception as exc:  # noqa: BLE001 - any failure here routes to the deterministic path
        logger.warning("llm_tool_loop_node: LLM path failed, routing to deterministic path: %s", exc)
        state.execution_path = "deterministic"
        state.llm_error = str(exc)
        state.last_error = str(exc)
    finally:
        _log_latency("llm_tool_loop_node", started_at)
    return state


def parse_query_node(state: AgentState) -> AgentState:
    started_at = time.time()
    state.last_error = None
    logger.info("parse_query_node: entering with query=%r", state.query)
    try:
        intent = query_parser.parse_query_rule_based(state.query)
        state.parsed_intent = QueryIntent(**intent)
        _agent_logger.log_tool_call("parse_query[rule_based]", (time.time() - started_at) * 1000, True)
    except Exception:  # noqa: BLE001
        # Parsing has no fallback of its own: an empty intent just means
        # "use the stored preferences", which the next nodes handle.
        logger.exception("parse_query_node: query parsing failed")
        _agent_logger.log_tool_call("parse_query[rule_based]", (time.time() - started_at) * 1000, False)
        state.parsed_intent = QueryIntent()
    finally:
        _log_latency("parse_query_node", started_at)
    return state


def fetch_preferences_node(state: AgentState) -> AgentState:
    started_at = time.time()
    state.last_error = None
    logger.info("fetch_preferences_node: entering for user_id=%s", state.user_id)
    try:
        _apply_preferences(state, tools.fetch_user_preferences(state.user_id))
        _agent_logger.log_tool_call("fetch_user_preferences", (time.time() - started_at) * 1000, True)
    except Exception as exc:  # noqa: BLE001
        logger.warning("fetch_preferences_node: preference lookup failed: %s", exc)
        _agent_logger.log_tool_call("fetch_user_preferences", (time.time() - started_at) * 1000, False)
        state.last_error = str(exc)
    finally:
        _log_latency("fetch_preferences_node", started_at)
    return state


def search_events_node(state: AgentState) -> AgentState:
    started_at = time.time()
    state.last_error = None
    genres = state.parsed_intent.genres or state.user_preferences.genres
    logger.info(
        "search_events_node: entering with genres=%s date_from=%s date_to=%s",
        genres, state.parsed_intent.date_from, state.parsed_intent.date_to,
    )
    try:
        price_max = state.user_preferences.budget if state.user_preferences.budget > 0 else None
        state.search_results = tools.search_events(
            genres,
            price_max=price_max,
            date_from=state.parsed_intent.date_from,
            date_to=state.parsed_intent.date_to,
        )
        _agent_logger.log_tool_call("search_events", (time.time() - started_at) * 1000, True)
    except Exception as exc:  # noqa: BLE001
        logger.warning("search_events_node: search failed: %s", exc)
        _agent_logger.log_tool_call("search_events", (time.time() - started_at) * 1000, False)
        state.search_results = []
        state.last_error = str(exc)
    finally:
        _log_latency("search_events_node", started_at)
    return state


def rank_events_node(state: AgentState) -> AgentState:
    started_at = time.time()
    state.last_error = None
    history = state.user_preferences.history()
    logger.info(
        "rank_events_node: entering with %d search results, %d attended / %d skipped / %d browsed in history",
        len(state.search_results), len(history["attended"]), len(history["skipped"]), len(history["browsed"]),
    )
    try:
        state.ranked_recommendations = tools.rank_events(state.search_results, history=history, limit=5)
        _agent_logger.log_tool_call("rank_events", (time.time() - started_at) * 1000, True)
    except Exception as exc:  # noqa: BLE001
        logger.warning("rank_events_node: ranking failed: %s", exc)
        _agent_logger.log_tool_call("rank_events", (time.time() - started_at) * 1000, False)
        state.last_error = str(exc)
    finally:
        _log_latency("rank_events_node", started_at)
    return state


def format_output_node(state: AgentState) -> AgentState:
    started_at = time.time()
    state.last_error = None
    logger.info(
        "format_output_node: finalizing %d recommendations via %s path (fallbacks used: %s)",
        len(state.ranked_recommendations), state.execution_path, state.fallback_strategies_used,
    )
    _log_latency("format_output_node", started_at)
    return state


def _fallback_node(strategy):
    """Wrap a fallback strategy as a graph node that also records the metric."""

    def node(state: AgentState) -> AgentState:
        started_at = time.time()
        state = strategy(state)
        state.last_error = None
        _agent_logger.log_fallback(strategy.__name__)
        _log_latency(strategy.__name__, started_at)
        return state

    node.__name__ = strategy.__name__
    return node


graceful_degradation_node = _fallback_node(fallbacks.graceful_degradation_fallback)
alternative_source_node = _fallback_node(fallbacks.alternative_source_fallback)
partial_results_node = _fallback_node(fallbacks.partial_results_fallback)


# --------------------------------------------------------------------------- #
# Routing functions (one per conditional edge)
# --------------------------------------------------------------------------- #


def route_entry(state: AgentState) -> Literal["llm_tool_loop", "parse_query"]:
    return "llm_tool_loop" if state.execution_path == "llm_tools" else "parse_query"


def route_after_llm(state: AgentState) -> Literal["parse_query", "alternative_source_fallback", "rank_events"]:
    if state.last_error:
        return "parse_query"
    if not state.search_results:
        return "alternative_source_fallback"
    return "rank_events"


def route_after_fetch(state: AgentState) -> Literal["graceful_degradation_fallback", "search_events"]:
    return "graceful_degradation_fallback" if state.last_error else "search_events"


def route_after_search(state: AgentState) -> Literal["alternative_source_fallback", "rank_events"]:
    if state.last_error or not state.search_results:
        return "alternative_source_fallback"
    return "rank_events"


def route_after_rank(state: AgentState) -> Literal["partial_results_fallback", "format_output"]:
    return "partial_results_fallback" if state.last_error else "format_output"


# --------------------------------------------------------------------------- #
# Graph
# --------------------------------------------------------------------------- #


def build_graph() -> StateGraph:
    graph = StateGraph(AgentState)

    graph.add_node("start", start_node)
    graph.add_node("llm_tool_loop", llm_tool_loop_node)
    graph.add_node("parse_query", parse_query_node)
    graph.add_node("fetch_preferences", fetch_preferences_node)
    graph.add_node("graceful_degradation_fallback", graceful_degradation_node)
    graph.add_node("search_events", search_events_node)
    graph.add_node("alternative_source_fallback", alternative_source_node)
    graph.add_node("rank_events", rank_events_node)
    graph.add_node("partial_results_fallback", partial_results_node)
    graph.add_node("format_output", format_output_node)

    graph.set_entry_point("start")

    graph.add_conditional_edges("start", route_entry, ["llm_tool_loop", "parse_query"])
    graph.add_conditional_edges(
        "llm_tool_loop", route_after_llm, ["parse_query", "alternative_source_fallback", "rank_events"]
    )

    graph.add_edge("parse_query", "fetch_preferences")
    graph.add_conditional_edges(
        "fetch_preferences", route_after_fetch, ["graceful_degradation_fallback", "search_events"]
    )
    graph.add_edge("graceful_degradation_fallback", "search_events")

    graph.add_conditional_edges("search_events", route_after_search, ["alternative_source_fallback", "rank_events"])
    graph.add_edge("alternative_source_fallback", "rank_events")

    graph.add_conditional_edges("rank_events", route_after_rank, ["partial_results_fallback", "format_output"])
    graph.add_edge("partial_results_fallback", "format_output")

    graph.add_edge("format_output", END)
    return graph


_compiled_graph = build_graph().compile()


def invoke(user_id: str, query: str) -> dict:
    """Run the recommendation agent for a user and query.

    Args:
        user_id: The id of the user requesting recommendations.
        query: The user's natural-language request.

    Returns:
        A dict with the user's recommendations and run metadata.
    """
    global _agent_logger
    _agent_logger = AgentLogger()

    started_at = time.time()
    initial_state = AgentState(user_id=user_id, query=query)
    result = _compiled_graph.invoke(initial_state)
    final_state = AgentState.model_validate(result)
    total_latency_ms = (time.time() - started_at) * 1000

    _agent_logger.log_agent_complete(total_latency_ms, final_state.fallback_count)
    logger.info(
        "invoke: completed for user_id=%s via %s path in %.2fms with %d fallback(s)",
        user_id, final_state.execution_path, total_latency_ms, final_state.fallback_count,
    )

    return {
        "user_id": final_state.user_id,
        "query": final_state.query,
        "execution_path": final_state.execution_path,
        "llm_error": final_state.llm_error,
        "llm_summary": final_state.llm_summary,
        "parsed_intent": final_state.parsed_intent.model_dump(),
        "recommendations": final_state.ranked_recommendations,
        "fallback_count": final_state.fallback_count,
        "fallback_strategies_used": final_state.fallback_strategies_used,
        "fallback_details": final_state.fallback_details,
        "llm_transcript": final_state.llm_transcript,
        "total_latency_ms": total_latency_ms,
        "logs": _agent_logger.get_formatted_logs(),
    }
