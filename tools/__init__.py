"""Granular tools for the job search agent.

Tools:
- web_search: Search via Baidu (free, no API key)
- fetch_page: Fetch and extract web page content
- analyze_job: LLM-based job posting analysis
- get_coverage_report: View collection coverage statistics
"""

__all__ = ["web_search", "fetch_page", "analyze_job", "get_coverage_report"]


def __getattr__(name: str):
    """Lazy-load tool modules so one missing dependency does not break all tools."""
    if name == "web_search":
        from tools.web_search import web_search
        return web_search
    if name == "fetch_page":
        from tools.page_fetcher import fetch_page
        return fetch_page
    if name == "analyze_job":
        from tools.job_analyzer import analyze_job
        return analyze_job
    if name == "get_coverage_report":
        from tools.coverage_report import get_coverage_report
        return get_coverage_report
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
