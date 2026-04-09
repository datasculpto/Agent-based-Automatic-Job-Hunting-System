"""Deduplication utilities for job postings."""

from models.job import Job


def deduplicate_jobs(jobs: list[Job]) -> list[Job]:
    """Remove duplicate jobs by URL and company+title combination."""
    seen_urls: set[str] = set()
    seen_keys: set[str] = set()
    unique_jobs: list[Job] = []

    for job in jobs:
        url = job.job_url.strip().rstrip("/")
        key = job.dedup_key()

        if url in seen_urls or key in seen_keys:
            continue

        seen_urls.add(url)
        seen_keys.add(key)
        unique_jobs.append(job)

    return unique_jobs
