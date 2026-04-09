"""Direct crawling tool for known job listing websites.

Bypasses search engines entirely by directly fetching job listing pages
from known-working websites and extracting individual job URLs.

This is far more reliable than depending on search engines which
frequently trigger captchas or return empty results.
"""

import re
import time
import random
import requests
from langchain_core.tools import tool


_USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
]


def _get_headers(referer: str = "") -> dict:
    return {
        "User-Agent": random.choice(_USER_AGENTS),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
        "Referer": referer or "https://www.google.com/",
    }


def crawl_shixiseng(keywords: list[str], max_pages: int = 2) -> list[dict]:
    """Crawl shixiseng.com (实习僧) for intern job URLs.

    Args:
        keywords: List of search keywords like ["AI算法", "机器学习"]
        max_pages: Max pages to crawl per keyword

    Returns:
        List of {url, title, source} dicts
    """
    all_urls: set[str] = set()
    results: list[dict] = []
    headers = _get_headers("https://www.shixiseng.com/")

    for kw in keywords:
        for page in range(1, max_pages + 1):
            try:
                resp = requests.get(
                    f"https://www.shixiseng.com/interns?keyword={kw}&page={page}",
                    headers=headers,
                    timeout=10,
                )
                if resp.status_code != 200:
                    continue

                # Extract intern URLs
                urls = re.findall(r'(/intern/inn_[a-z0-9]+)', resp.text)
                for u in urls:
                    full_url = "https://www.shixiseng.com" + u
                    if full_url not in all_urls:
                        all_urls.add(full_url)
                        results.append({
                            "link": full_url,
                            "title": f"{kw} 实习岗位",
                            "source": "实习僧",
                            "snippet": "",
                        })
                time.sleep(0.3 + random.uniform(0, 0.3))
            except Exception:
                continue

    return results


def crawl_nowcoder(keywords: list[str], max_pages: int = 2) -> list[dict]:
    """Crawl nowcoder.com (牛客网) for job URLs.

    Args:
        keywords: List of search keywords
        max_pages: Max pages to crawl per keyword

    Returns:
        List of {url, title, source} dicts
    """
    all_urls: set[str] = set()
    results: list[dict] = []
    headers = _get_headers("https://www.nowcoder.com/")

    for kw in keywords:
        for page in range(1, max_pages + 1):
            try:
                resp = requests.get(
                    f"https://www.nowcoder.com/search?type=job&query={kw}&page={page}",
                    headers=headers,
                    timeout=10,
                )
                if resp.status_code != 200:
                    continue

                # Extract job detail URLs
                urls = re.findall(r'(/jobs/detail/\d+)', resp.text)
                for u in urls:
                    full_url = "https://www.nowcoder.com" + u
                    if full_url not in all_urls:
                        all_urls.add(full_url)
                        results.append({
                            "link": full_url,
                            "title": f"{kw} 校招岗位",
                            "source": "牛客网",
                            "snippet": "",
                        })

                # Also check for nowpick URLs
                nowpick_urls = re.findall(r'(https://nowpick\.nowcoder\.com/w/school/detail\?jobId=\d+)', resp.text)
                for u in nowpick_urls:
                    if u not in all_urls:
                        all_urls.add(u)
                        results.append({
                            "link": u,
                            "title": f"{kw} 校招岗位",
                            "source": "牛客网",
                            "snippet": "",
                        })

                time.sleep(0.3 + random.uniform(0, 0.3))
            except Exception:
                continue

    return results


def crawl_ncss(keywords: list[str], max_pages: int = 1) -> list[dict]:
    """Crawl ncss.cn subdomains (国家大学生就业服务平台) for job URLs."""
    all_urls: set[str] = set()
    results: list[dict] = []
    headers = _get_headers("https://www.ncss.cn/")

    # Try known NCSS subdomains (university employment platforms)
    subdomains = ["www", "ce", "shbangde"]
    for subdomain in subdomains:
        for kw in keywords[:3]:  # Limit keywords for NCSS
            try:
                resp = requests.get(
                    f"https://{subdomain}.ncss.cn/student/jobs/search?keyword={kw}",
                    headers=headers,
                    timeout=10,
                )
                if resp.status_code != 200:
                    continue

                # Extract job detail URLs
                urls = re.findall(r'(/student/jobs/[A-Za-z0-9]+/detail\.html)', resp.text)
                for u in urls:
                    full_url = f"https://{subdomain}.ncss.cn{u}"
                    if full_url not in all_urls:
                        all_urls.add(full_url)
                        results.append({
                            "link": full_url,
                            "title": f"{kw} 校招岗位",
                            "source": f"{subdomain}.ncss.cn",
                            "snippet": "",
                        })
                time.sleep(0.3 + random.uniform(0, 0.3))
            except Exception:
                continue

    return results


def crawl_university_jobs(keywords: list[str]) -> list[dict]:
    """Crawl university employment websites for job URLs."""
    all_urls: set[str] = set()
    results: list[dict] = []
    headers = _get_headers()

    # Known university job sites with search APIs
    sites = [
        ("job.xidian.edu.cn", "/job/search?keyword={kw}", r'/job/view/id/(\d+)'),
    ]

    for domain, path_template, pattern in sites:
        for kw in keywords[:3]:
            try:
                url = f"https://{domain}{path_template.format(kw=kw)}"
                resp = requests.get(url, headers=headers, timeout=10)
                if resp.status_code != 200:
                    continue

                ids = re.findall(pattern, resp.text)
                for job_id in ids:
                    full_url = f"https://{domain}/job/view/id/{job_id}"
                    if full_url not in all_urls:
                        all_urls.add(full_url)
                        results.append({
                            "link": full_url,
                            "title": f"{kw} 校招岗位",
                            "source": domain,
                            "snippet": "",
                        })
                time.sleep(0.3 + random.uniform(0, 0.3))
            except Exception:
                continue

    return results


def direct_crawl_all(keywords: list[str] | None = None) -> list[dict]:
    """Crawl all known job sites and return combined results.

    Args:
        keywords: Optional custom keywords. Defaults to common AI terms.

    Returns:
        Combined list of job URLs from all sites.
    """
    if keywords is None:
        keywords = [
            "AI算法", "机器学习", "深度学习", "大模型", "NLP",
            "算法工程师", "推荐算法", "计算机视觉", "LLM", "自然语言处理",
            "自动驾驶", "语音识别", "数据挖掘",
        ]

    all_results: list[dict] = []

    print("  [DirectCrawl] Crawling shixiseng.com...")
    all_results.extend(crawl_shixiseng(keywords, max_pages=2))
    print(f"    -> {len(all_results)} URLs from shixiseng")

    print("  [DirectCrawl] Crawling nowcoder.com...")
    nc_results = crawl_nowcoder(keywords, max_pages=2)
    all_results.extend(nc_results)
    print(f"    -> {len(nc_results)} URLs from nowcoder")

    print("  [DirectCrawl] Crawling ncss.cn...")
    ncss_results = crawl_ncss(keywords)
    all_results.extend(ncss_results)
    print(f"    -> {len(ncss_results)} URLs from ncss")

    print("  [DirectCrawl] Crawling university job sites...")
    uni_results = crawl_university_jobs(keywords)
    all_results.extend(uni_results)
    print(f"    -> {len(uni_results)} URLs from universities")

    print(f"  [DirectCrawl] Total: {len(all_results)} unique URLs collected")
    return all_results
