"""Web search tool using multiple search engines for finding job postings.

This is a granular tool — it ONLY searches and returns results.
The agent decides which results to investigate further.

Primary: SerpAPI (Google, reliable, no captcha)
Fallback: Baidu, Sogou, Bing
Includes rate limiting protection with delays between searches.
"""

import os
import json
import random
import time
import requests
from bs4 import BeautifulSoup
from langchain_core.tools import tool
from urllib.parse import urlparse

# Rate limiting: track last search time
_last_search_time = 0.0
_MIN_SEARCH_INTERVAL = 2.0  # Min seconds between searches


def _identify_source(url: str) -> str:
    """Identify the job site from URL."""
    domain_map = {
        "zhipin.com": "BOSS直聘",
        "nowcoder.com": "牛客网",
        "lagou.com": "拉勾网",
        "zhaopin.com": "智联招聘",
        "51job.com": "前程无忧",
        "liepin.com": "猎聘",
        "maimai.cn": "脉脉",
        "linkedin.com": "LinkedIn",
        "kanzhun.com": "看准网",
        "shixiseng.com": "实习僧",
        "bytedance.com": "字节跳动官网",
        "tencent.com": "腾讯官网",
        "alibaba.com": "阿里巴巴官网",
        "huawei.com": "华为官网",
        "jobs.163.com": "网易招聘",
        "join.qq.com": "腾讯招聘",
        "talent.baidu.com": "百度招聘",
        "campus.meituan.com": "美团校招",
        "xiaomi.com": "小米招聘",
        "weixin.qq.com": "微信公众号",
        "mp.weixin.qq.com": "微信公众号",
    }
    for domain, name in domain_map.items():
        if domain in url:
            return name
    try:
        parts = url.split("/")
        if len(parts) >= 3:
            return parts[2]
    except Exception:
        pass
    return "未知"


def _resolve_baidu_url(baidu_link: str, headers: dict) -> str:
    """Resolve a Baidu redirect URL to the actual target URL."""
    if not baidu_link.startswith("http://www.baidu.com/link"):
        return baidu_link
    try:
        resp = requests.head(baidu_link, headers=headers, timeout=5, allow_redirects=True)
        return resp.url
    except Exception:
        try:
            resp = requests.get(baidu_link, headers=headers, timeout=5, allow_redirects=True, stream=True)
            real_url = resp.url
            resp.close()
            return real_url
        except Exception:
            return baidu_link


def _get_headers() -> dict:
    """Get randomized request headers."""
    ua = random.choice([
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:123.0) Gecko/20100101 Firefox/123.0",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    ])
    return {
        "User-Agent": ua,
        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }


def _normalize_url(url: str) -> str:
    """Normalize URLs so results from multiple engines deduplicate cleanly."""
    if not url:
        return ""
    try:
        parsed = urlparse(url)
        path = parsed.path.rstrip("/")
        return f"{parsed.scheme}://{parsed.netloc}{path}"
    except Exception:
        return url.split("?")[0].rstrip("/")


def _merge_results(engine_results: list[tuple[str, list[dict]]], limit: int = 20) -> list[dict]:
    """Merge multiple engine result sets without keeping duplicate URLs."""
    merged: list[dict] = []
    seen_urls: set[str] = set()

    for engine_name, results in engine_results:
        for item in results:
            normalized = _normalize_url(item.get("link", ""))
            if not normalized or normalized in seen_urls:
                continue
            seen_urls.add(normalized)
            merged.append({**item, "engine": engine_name})
            if len(merged) >= limit:
                return merged

    return merged


def _search_baidu(query: str, num_results: int = 15) -> list[dict]:
    """Search Baidu and parse results."""
    headers = _get_headers()

    # Generate a random BAIDUID cookie to avoid captcha
    import hashlib
    rand_id = hashlib.md5(str(random.random()).encode()).hexdigest().upper()
    cookies = {
        "BAIDUID": f"{rand_id}:FG=1",
        "BIDUPSID": rand_id,
    }

    resp = requests.get(
        "https://www.baidu.com/s",
        params={"wd": query, "rn": str(num_results)},
        headers=headers,
        cookies=cookies,
        timeout=15,
    )
    resp.raise_for_status()

    html = resp.content.decode("utf-8", errors="replace")

    # Check for captcha/verification
    if "百度安全验证" in html or "verify" in html.lower()[:500]:
        raise Exception("Baidu captcha triggered, switching engine")

    soup = BeautifulSoup(html, "html.parser")
    items = soup.select(".result") or soup.select(".c-container")

    results = []
    for item in items:
        title_el = item.select_one("h3 a") or item.select_one("a")
        if not title_el:
            continue

        title = title_el.get_text(strip=True)
        baidu_link = title_el.get("href", "")

        snippet_el = (
            item.select_one(".c-abstract")
            or item.select_one("span.content-right_8Zs40")
            or item.select_one("p")
        )
        snippet = snippet_el.get_text(strip=True)[:200] if snippet_el else ""

        if title and baidu_link:
            real_url = _resolve_baidu_url(baidu_link, headers)
            results.append({
                "title": title,
                "link": real_url,
                "snippet": snippet,
                "source": _identify_source(real_url),
            })

    return results[:num_results]


def _resolve_sogou_url(sogou_link: str, headers: dict) -> str:
    """Resolve a Sogou redirect URL by parsing the JS redirect."""
    if "sogou.com/link" not in sogou_link:
        return sogou_link
    try:
        resp = requests.get(sogou_link, headers=headers, timeout=5, allow_redirects=False)
        content = resp.content.decode("utf-8", errors="replace")

        # Extract URL from window.location.replace("...")
        import re
        match = re.search(r'window\.location\.replace\("([^"]+)"\)', content)
        if match:
            return match.group(1)

        # Try meta refresh
        match = re.search(r'URL=\'([^\']+)\'', content)
        if match:
            return match.group(1)

        # Try Location header
        if resp.headers.get("Location"):
            return resp.headers["Location"]

        return sogou_link
    except Exception:
        return sogou_link


def _search_sogou(query: str, num_results: int = 15) -> list[dict]:
    """Search Sogou (Chinese search engine, good fallback)."""
    headers = _get_headers()

    resp = requests.get(
        "https://www.sogou.com/web",
        params={"query": query, "num": str(num_results)},
        headers=headers,
        timeout=15,
    )
    resp.raise_for_status()

    html = resp.content.decode("utf-8", errors="replace")
    soup = BeautifulSoup(html, "html.parser")

    results = []
    # Sogou result selectors
    for item in soup.select(".vrwrap") or soup.select(".rb"):
        title_el = item.select_one("h3 a") or item.select_one("a")
        if not title_el:
            continue

        title = title_el.get_text(strip=True)
        link = title_el.get("href", "")

        snippet_el = item.select_one(".str-info") or item.select_one("p")
        snippet = snippet_el.get_text(strip=True)[:200] if snippet_el else ""

        if title and link:
            # Resolve Sogou redirect URLs (they use JS redirect)
            if link.startswith("/link"):
                link = "https://www.sogou.com" + link
            if "sogou.com/link" in link:
                link = _resolve_sogou_url(link, headers)
            # Skip relative URLs that aren't proper links
            if link.startswith("?") or not link.startswith("http"):
                continue
            results.append({
                "title": title,
                "link": link,
                "snippet": snippet,
                "source": _identify_source(link),
            })

    return results[:num_results]


def _search_bing(query: str, num_results: int = 15) -> list[dict]:
    """Search Bing (international, may work as fallback)."""
    headers = _get_headers()
    headers["Accept-Language"] = "zh-CN,zh;q=0.9"

    resp = requests.get(
        "https://www.bing.com/search",
        params={"q": query, "count": str(num_results)},
        headers=headers,
        timeout=15,
    )
    resp.raise_for_status()

    html = resp.content.decode("utf-8", errors="replace")
    soup = BeautifulSoup(html, "html.parser")

    results = []
    for item in soup.select("li.b_algo"):
        title_el = item.select_one("h2 a")
        if not title_el:
            continue

        title = title_el.get_text(strip=True)
        link = title_el.get("href", "")
        snippet_el = item.select_one(".b_caption p") or item.select_one("p")
        snippet = snippet_el.get_text(strip=True)[:200] if snippet_el else ""

        if link and title:
            results.append({
                "title": title,
                "link": link,
                "snippet": snippet,
                "source": _identify_source(link),
            })

    return results[:num_results]


def _search_duckduckgo(query: str, num_results: int = 15) -> list[dict]:
    """Search using DuckDuckGo HTML version. Reliable, no captcha, no JS needed."""
    from urllib.parse import unquote

    headers = _get_headers()
    resp = requests.get(
        "https://html.duckduckgo.com/html/",
        params={"q": query},
        headers=headers,
        timeout=8,
    )
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")
    results = []

    for item in soup.select(".result"):
        title_el = item.select_one(".result__a")
        if not title_el:
            continue

        title = title_el.get_text(strip=True)
        href = title_el.get("href", "")

        # Extract actual URL from DDG redirect
        if "uddg=" in href:
            actual_url = href.split("uddg=")[1].split("&")[0]
            actual_url = unquote(actual_url)
        else:
            actual_url = href

        if not actual_url.startswith("http"):
            continue

        snippet_el = item.select_one(".result__snippet")
        snippet = snippet_el.get_text(strip=True)[:200] if snippet_el else ""

        results.append({
            "title": title,
            "link": actual_url,
            "snippet": snippet,
            "source": _identify_source(actual_url),
        })

    return results[:num_results]


# Track engine availability (skip engines that have been rate-limited/captcha'd)
_serpapi_available = True
_baidu_available = True
_duckduckgo_available = True
_duckduckgo_fail_count = 0


def _search_serpapi(query: str, num_results: int = 15) -> list[dict]:
    """Search using SerpAPI (Google). Reliable, no captcha issues."""
    api_key = os.getenv("SERPAPI_API_KEY")
    if not api_key:
        raise Exception("SERPAPI_API_KEY not set")

    resp = requests.get(
        "https://serpapi.com/search",
        params={
            "q": query,
            "engine": "google",
            "hl": "zh-cn",
            "gl": "cn",
            "num": str(num_results),
            "api_key": api_key,
        },
        timeout=20,
    )
    resp.raise_for_status()
    data = resp.json()

    results = []
    for item in data.get("organic_results", []):
        title = item.get("title", "")
        link = item.get("link", "")
        snippet = item.get("snippet", "")[:200]
        if title and link:
            results.append({
                "title": title,
                "link": link,
                "snippet": snippet,
                "source": _identify_source(link),
            })

    return results[:num_results]


@tool
def web_search(query: str) -> str:
    """Search the web for job postings. Returns titles, links, and snippets.

    Uses multiple search engines (Baidu primary, Sogou/Bing fallback).
    Includes automatic rate limiting to avoid blocks.

    Args:
        query: Search query string, e.g. "AI算法工程师 校招 2025"

    Returns:
        JSON string with search results: {query, result_count, results: [{title, link, snippet, source}]}
    """
    global _last_search_time

    # Rate limiting: wait between searches
    elapsed = time.time() - _last_search_time
    if elapsed < _MIN_SEARCH_INTERVAL:
        wait = _MIN_SEARCH_INTERVAL - elapsed + random.uniform(0.5, 1.5)
        time.sleep(wait)
    _last_search_time = time.time()

    global _serpapi_available, _baidu_available, _duckduckgo_available, _duckduckgo_fail_count
    errors = []

    engine_results: list[tuple[str, list[dict]]] = []

    # Strategy 1: DuckDuckGo HTML (no captcha, no API key)
    if _duckduckgo_available:
        try:
            results = _search_duckduckgo(query)
            if results:
                engine_results.append(("duckduckgo", results))
                _duckduckgo_fail_count = 0
        except Exception as e:
            errors.append(f"duckduckgo: {str(e)[:80]}")
            _duckduckgo_fail_count += 1
            if _duckduckgo_fail_count >= 3:
                _duckduckgo_available = False

    # Strategy 2: Bing (good for Chinese queries, rarely blocks)
    try:
        results = _search_bing(query)
        if results:
            engine_results.append(("bing", results))
    except Exception as e:
        errors.append(f"bing: {str(e)[:80]}")

    # Strategy 3: Baidu (best recall for Chinese queries, skip if captcha'd previously)
    if _baidu_available:
        try:
            time.sleep(1)
            results = _search_baidu(query)
            if results:
                engine_results.append(("baidu", results))
        except Exception as e:
            err_str = str(e)
            errors.append(f"baidu: {err_str[:80]}")
            if "captcha" in err_str.lower() or "验证" in err_str:
                _baidu_available = False  # Don't retry after captcha

    # Strategy 4: Sogou (Chinese fallback)
    try:
        time.sleep(0.5)
        results = _search_sogou(query)
        if results:
            engine_results.append(("sogou", results))
    except Exception as e:
        errors.append(f"sogou: {str(e)[:80]}")

    # Strategy 5: SerpAPI (skip if rate-limited or no key)
    if _serpapi_available:
        try:
            results = _search_serpapi(query)
            if results:
                engine_results.append(("serpapi", results))
        except Exception as e:
            err_str = str(e)
            errors.append(f"serpapi: {err_str[:80]}")
            if "429" in err_str or "Too Many" in err_str or "not set" in err_str:
                _serpapi_available = False  # Don't retry on rate limit or missing key

    merged_results = _merge_results(engine_results, limit=20)
    if merged_results:
        return json.dumps({
            "query": query,
            "engine": ",".join(engine for engine, _ in engine_results),
            "result_count": len(merged_results),
            "results": merged_results,
        }, ensure_ascii=False)

    # All failed
    return json.dumps({
        "error": f"All search engines failed or returned empty: {'; '.join(errors)}",
        "query": query,
        "result_count": 0,
        "results": [],
    }, ensure_ascii=False)
