"""
AI Engineer Job Search Agent - Main Entry Point

A true autonomous AI Agent with Plan-Act-Observe-Reflect architecture.
The agent autonomously plans search strategies, executes with granular tools,
and reflects on results to adaptively improve its approach.

Architecture: LangGraph (Strategist → Executor → Tools → Processor → Reflector)
"""

import os
import sys
import io
import json
import time
from pathlib import Path
from collections import Counter

# Fix Windows console encoding for Chinese + emoji output
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import pandas as pd
from dotenv import load_dotenv

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from agent.graph import build_graph, get_initial_state
from models.job import Job
from utils.dedup import deduplicate_jobs


def export_results(jobs: list[Job], output_dir: str = "output") -> tuple[str, str]:
    """Export jobs to JSON and CSV files."""
    os.makedirs(output_dir, exist_ok=True)

    # Export JSON
    json_path = os.path.join(output_dir, "jobs.json")
    jobs_dicts = [job.model_dump() for job in jobs]
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(jobs_dicts, f, ensure_ascii=False, indent=2)

    # Export CSV
    csv_path = os.path.join(output_dir, "jobs.csv")
    df = pd.DataFrame(jobs_dicts)
    if "tech_tags" in df.columns:
        df["tech_tags"] = df["tech_tags"].apply(lambda x: ", ".join(x) if isinstance(x, list) else x)
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    return json_path, csv_path


def print_summary(jobs: list[Job]) -> None:
    """Print a summary of collected jobs."""
    print("\n" + "=" * 60)
    print("  FINAL RESULTS")
    print("=" * 60)

    print(f"\n  Total jobs collected: {len(jobs)}")

    # Source breakdown
    source_counts = Counter(job.source for job in jobs)
    print(f"\n  Sources ({len(source_counts)}):")
    for source, count in source_counts.most_common():
        print(f"    - {source}: {count}")

    # Location breakdown
    location_counts = Counter(job.location for job in jobs)
    print(f"\n  Locations (Top 10):")
    for loc, count in location_counts.most_common(10):
        print(f"    - {loc}: {count}")

    # Tech tags breakdown
    all_tags = []
    for job in jobs:
        all_tags.extend(job.tech_tags)
    tag_counts = Counter(all_tags)
    print(f"\n  Top Tech Tags:")
    for tag, count in tag_counts.most_common(15):
        print(f"    - {tag}: {count}")

    # Sample jobs
    print(f"\n  Sample Jobs (first 5):")
    for i, job in enumerate(jobs[:5], 1):
        tags = ", ".join(job.tech_tags[:3]) if job.tech_tags else "N/A"
        print(f"    {i}. [{job.company}] {job.title}")
        print(f"       Location: {job.location} | Salary: {job.salary} | Tags: {tags}")

    print("\n" + "=" * 60)


def main():
    """Run the AI Engineer Job Search Agent."""

    # Load environment variables
    load_dotenv()

    # Validate required environment variables
    if not os.getenv("DEEPSEEK_API_KEY"):
        print("Error: DEEPSEEK_API_KEY not set in .env file")
        sys.exit(1)

    print("=" * 60)
    print("  AI Engineer Job Search Agent")
    print("  Architecture: Plan-Act-Observe-Reflect")
    print("=" * 60)
    print(f"  Target:  50 AI Engineer campus recruitment jobs")
    llm_name = "DeepSeek-Chat"
    print(f"  LLM:     {llm_name} (planning + reflection + analysis)")
    print(f"  Search:  Baidu (free, no API key required)")
    print(f"  Tools:   web_search, fetch_page, analyze_job, get_coverage_report")
    print("=" * 60)

    # Build the agent graph
    print("\nBuilding agent graph (strategist -> executor -> tools -> processor -> reflector)...")
    graph = build_graph()

    # Get initial state
    initial_state = get_initial_state()

    # Run the agent
    print("Agent starting...\n")
    start_time = time.time()

    try:
        final_state = graph.invoke(
            initial_state,
            config={"recursion_limit": 500},
        )
    except Exception as e:
        print(f"\nAgent stopped: {e}")
        print("Attempting to save collected jobs...")
        final_state = initial_state
        # Recover collected jobs from the module-level tracker
        try:
            from tools.coverage_report import get_collected_jobs
            recovered = get_collected_jobs()
            if recovered:
                final_state = {**initial_state, "collected_jobs": recovered}
                print(f"  Recovered {len(recovered)} jobs from tracker")
        except Exception:
            pass

    elapsed = time.time() - start_time
    print(f"\nElapsed time: {elapsed:.1f} seconds")

    # Parse collected jobs
    raw_jobs = []
    for job_json in final_state.get("collected_jobs", []):
        try:
            job_data = json.loads(job_json)
            raw_jobs.append(Job(**job_data))
        except Exception as e:
            print(f"Warning: failed to parse job: {e}")

    # Deduplicate
    jobs = deduplicate_jobs(raw_jobs)
    print(f"\nDedup: {len(raw_jobs)} -> {len(jobs)} jobs")

    if not jobs:
        print("\nNo jobs collected. Please check API configuration and network.")
        sys.exit(1)

    # Export results
    json_path, csv_path = export_results(jobs)
    print(f"\nResults saved:")
    print(f"  JSON: {json_path}")
    print(f"  CSV:  {csv_path}")

    # Print summary
    print_summary(jobs)

    return jobs


if __name__ == "__main__":
    main()
