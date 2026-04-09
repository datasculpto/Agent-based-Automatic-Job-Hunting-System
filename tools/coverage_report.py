"""Coverage report tool — lets the agent inspect its collection progress.

Uses a module-level variable updated by result_processor so the tool
can be used within LangGraph's standard ToolNode.
"""

import json
from langchain_core.tools import tool

# Module-level coverage data, updated by result_processor node
_current_coverage: dict = {}
# Module-level collected jobs tracker for error recovery
_collected_jobs: list[str] = []


def update_coverage(coverage: dict) -> None:
    """Called by result_processor to update the shared coverage data."""
    global _current_coverage
    _current_coverage = coverage


def track_collected_jobs(jobs: list[str]) -> None:
    """Called by result_processor to track all collected jobs for error recovery."""
    global _collected_jobs
    _collected_jobs = jobs


def get_collected_jobs() -> list[str]:
    """Get all collected jobs (for error recovery)."""
    return _collected_jobs


@tool
def get_coverage_report() -> str:
    """Get the current collection coverage report.

    Shows statistics about collected jobs: distribution by source website,
    location, AI domain, total count, and recent round yield.

    Use this to understand what you've collected so far and identify gaps.

    Returns:
        JSON string with coverage analytics.
    """
    if not _current_coverage:
        return json.dumps({
            "total_jobs": 0,
            "message": "No jobs collected yet. Start searching!",
        }, ensure_ascii=False)

    return json.dumps(_current_coverage, ensure_ascii=False)
