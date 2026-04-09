"""Agent state definition for the LangGraph job search agent.

Supports the Plan-Act-Observe-Reflect architecture with:
- Strategic planning state (current_plan, plan_step_index)
- Reflection memory (reflections)
- Coverage analytics (coverage)
"""

import operator
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage


class AgentState(TypedDict):
    """State schema for the job search agent.

    All list fields use operator.add as reducer so that node returns
    are APPENDED rather than replacing the entire list.
    """

    # === Core message history ===
    messages: Annotated[list[BaseMessage], add_messages]

    # === Job collection ===
    collected_jobs: Annotated[list[str], operator.add]  # JSON strings of Job objects
    job_count: int

    # === Search tracking ===
    searched_queries: Annotated[list[str], operator.add]
    processed_urls: Annotated[list[str], operator.add]

    # === Strategic planning ===
    current_plan: str       # JSON string: list of sub-goals with rationale
    plan_step_index: int    # which step of the plan we're executing

    # === Reflection memory ===
    reflections: Annotated[list[str], operator.add]  # JSON strings of per-round reflections

    # === Coverage analytics (updated by result_processor) ===
    # JSON string: {by_source, by_location, by_domain, last_round_yield, stale_rounds, ...}
    coverage: str

    # === Control ===
    iteration: int
    status: str  # "running" | "reflecting" | "replanning" | "done"
