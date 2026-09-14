"""Agent state schema for the event recommendation agent."""

from pydantic import BaseModel, Field


class UserPreferences(BaseModel):
    genres: list[str] = Field(default_factory=list)
    budget: int = 0
    # Event ids from the user's history, used by the ranking step.
    attended: list[str] = Field(default_factory=list)
    skipped: list[str] = Field(default_factory=list)
    browsed: list[str] = Field(default_factory=list)

    def history(self) -> dict[str, list[str]]:
        return {"attended": self.attended, "skipped": self.skipped, "browsed": self.browsed}


class QueryIntent(BaseModel):
    genres: list[str] = Field(default_factory=list)
    date_from: str | None = None
    date_to: str | None = None


class AgentState(BaseModel):
    user_id: str
    query: str
    parsed_intent: QueryIntent = Field(default_factory=QueryIntent)
    user_preferences: UserPreferences = Field(default_factory=UserPreferences)
    search_results: list[dict] = Field(default_factory=list)
    ranked_recommendations: list[dict] = Field(default_factory=list)
    fallback_count: int = 0
    fallback_strategies_used: list[str] = Field(default_factory=list)
    # Per-fallback detail, e.g. which relaxation stage Fallback B settled on.
    fallback_details: dict[str, str] = Field(default_factory=dict)

    # Which path the graph took: "llm_tools" (Claude on Bedrock chose and
    # called the tools) or "deterministic" (rule-based parsing + fixed
    # tool sequence). Set by start_node, downgraded by llm_tool_loop_node
    # if the LLM path fails.
    execution_path: str = "deterministic"
    # Why the LLM path was skipped or abandoned, if it was.
    llm_error: str | None = None
    # The model's closing sentence from the tool loop, if any.
    llm_summary: str | None = None
    # The full message exchange with the model, serialised, when the LLM
    # path succeeded. Printed by ``demo.py --transcript``.
    llm_transcript: list[dict] = Field(default_factory=list)

    # Set by a node when its tool call failed; read by the routing
    # function on that node's outgoing conditional edge, and cleared by
    # whichever node runs next. Never carried across more than one edge.
    last_error: str | None = None
