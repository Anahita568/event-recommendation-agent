"""Observability primitives for the event recommendation agent."""

import logging
import time
from dataclasses import dataclass, field

logger = logging.getLogger("agent.metrics")


@dataclass
class ToolCallMetrics:
    tool_name: str
    latency_ms: float
    success: bool
    timestamp: float = field(default_factory=time.time)


class AgentLogger:
    """Collects tool call, fallback, and completion events for one agent run."""

    def __init__(self) -> None:
        self.tool_calls: list[ToolCallMetrics] = []
        self.fallbacks: list[str] = []
        self.total_latency_ms: float | None = None
        self.fallback_count: int = 0

    def log_tool_call(self, tool_name: str, latency: float, success: bool) -> None:
        metric = ToolCallMetrics(tool_name=tool_name, latency_ms=latency, success=success)
        self.tool_calls.append(metric)
        level = logging.INFO if success else logging.ERROR
        logger.log(level, "tool=%s latency_ms=%.2f success=%s", tool_name, latency, success)

    def log_fallback(self, strategy_name: str) -> None:
        self.fallbacks.append(strategy_name)
        self.fallback_count += 1
        logger.warning("fallback triggered: %s", strategy_name)

    def log_agent_complete(self, total_latency: float, fallback_count: int) -> None:
        self.total_latency_ms = total_latency
        self.fallback_count = fallback_count
        logger.info(
            "agent run complete: total_latency_ms=%.2f fallback_count=%d",
            total_latency,
            fallback_count,
        )

    def get_formatted_logs(self) -> str:
        lines = ["Tool calls:"]
        for call in self.tool_calls:
            status = "OK" if call.success else "FAILED"
            lines.append(f"  - {call.tool_name}: {call.latency_ms:.2f}ms [{status}]")

        lines.append("Fallbacks triggered:")
        if self.fallbacks:
            lines.extend(f"  - {name}" for name in self.fallbacks)
        else:
            lines.append("  - none")

        if self.total_latency_ms is not None:
            lines.append(f"Total latency: {self.total_latency_ms:.2f}ms")
            lines.append(f"Fallback count: {self.fallback_count}")

        return "\n".join(lines)
