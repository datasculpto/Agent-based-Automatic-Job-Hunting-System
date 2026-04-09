"""Web page fetching and content extraction tool.

This is a granular tool — it ONLY fetches and extracts text.
The agent decides whether the content is worth analyzing with LLM.

Supports two modes:
- requests + BeautifulSoup for static HTML pages (fast, lightweight)
- Playwright headless Chromium for JS-rendered pages (猎聘, 智联招聘, etc.)

Performance optimizations:
- Playwright browser instance is reused across calls (no repeated startup)
- Domains that fail repeatedly are auto-blacklisted at runtime
"""

import json
import random
import atexit
import threading
import requests
from bs4 import BeautifulSoup
from langchain_core.tools import tool

# Track the main thread ID so we can detect when we're in a worker thread
_main_thread_id = threading.main_thread().ident


# Rotate User-Agent to reduce blocking
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
]

MAX_CONTENT_LENGTH = 8000

# Domains that require JavaScript rendering
JS_RENDER_DOMAINS = [
    "zhipin.com",           # BOSS直聘
    "liepin.com",           # 猎聘
    "lagou.com",            # 拉勾网
    "zhaopin.com",          # 智联招聘
    "careers.tencent.com",  # 腾讯招聘
    "jobs.bytedance.com",   # 字节跳动
    "join.qq.com",          # QQ招聘
    "campus.meituan.com",   # 美团校招
    "kanzhun.com",          # 看准网
]

# Runtime blacklist: domains that fail repeatedly are auto-skipped
# Key: domain, Value: consecutive failure count
_domain_fail_count: dict[str, int] = {}
_DOMAIN_FAIL_THRESHOLD = 5  # Skip after 5 consecutive failures


# ============================================================
# Shared Playwright browser (lazy init, reused across calls)
# ============================================================
_playwright_instance = None
_browser_instance = None


def _get_browser():
    """Get or create a shared Playwright browser instance."""
    global _playwright_instance, _browser_instance
    if _browser_instance is None:
        try:
            from playwright.sync_api import sync_playwright
        except ImportError:
            return None
        _playwright_instance = sync_playwright().start()
        _browser_instance = _playwright_instance.chromium.launch(headless=True)
    return _browser_instance


def _cleanup_browser():
    """Cleanup browser on process exit."""
    global _playwright_instance, _browser_instance
    try:
        if _browser_instance:
            _browser_instance.close()
            _browser_instance = None
        if _playwright_instance:
            _playwright_instance.stop()
            _playwright_instance = None
    except Exception:
        pass


atexit.register(_cleanup_browser)


# ============================================================
# Helper functions
# ============================================================

def _get_domain(url: str) -> str:
    """Extract domain from URL for blacklist tracking."""
    try:
        from urllib.parse import urlparse
        return urlparse(url).netloc.lower()
    except Exception:
        return url


def _is_blacklisted(url: str) -> bool:
    """Check if a domain has been auto-blacklisted due to repeated failures."""
    domain = _get_domain(url)
    return _domain_fail_count.get(domain, 0) >= _DOMAIN_FAIL_THRESHOLD


def _record_failure(url: str):
    """Record a fetch failure for runtime blacklist tracking."""
    domain = _get_domain(url)
    _domain_fail_count[domain] = _domain_fail_count.get(domain, 0) + 1
    count = _domain_fail_count[domain]
    if count == _DOMAIN_FAIL_THRESHOLD:
        print(f"      [Blacklist] {domain} auto-blacklisted after {count} consecutive failures")


def _record_success(url: str):
    """Reset failure count on success."""
    domain = _get_domain(url)
    _domain_fail_count[domain] = 0


def _get_headers() -> dict:
    """Get request headers with randomized User-Agent."""
    return {
        "User-Agent": random.choice(USER_AGENTS),
        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Encoding": "gzip, deflate",
        "Connection": "keep-alive",
    }


def _is_main_thread() -> bool:
    """Check if we're running in the main thread (Playwright is not thread-safe)."""
    return threading.current_thread().ident == _main_thread_id


def _needs_js_render(url: str) -> bool:
    """Check if a URL requires JavaScript rendering.
    Returns False when called from a worker thread since Playwright is not thread-safe.
    """
    if not _is_main_thread():
        return False
    return any(domain in url for domain in JS_RENDER_DOMAINS)


def _extract_text(html: str) -> str:
    """Parse HTML and extract clean text content."""
    soup = BeautifulSoup(html, "html.parser")

    # Remove script/style/nav elements
    for tag in soup(["script", "style", "nav", "header", "footer", "iframe", "noscript"]):
        tag.decompose()

    text = soup.get_text(separator="\n", strip=True)

    # Collapse multiple blank lines
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    text = "\n".join(lines)

    # Truncate to fit LLM context
    if len(text) > MAX_CONTENT_LENGTH:
        text = text[:MAX_CONTENT_LENGTH] + "\n...[内容已截断]"

    return text


def _looks_like_verification_page(text: str) -> bool:
    """Detect anti-bot / verification pages that should not be sent to the LLM."""
    signals = [
        "网站访客身份验证",
        "点击按钮进行验证",
        "完成验证后即可正常使用",
        "滑动验证",
        "请拖动滑块",
        "请完成下方验证后继续操作",
        "访问异常",
        "captcha",
        "verify-slider",
    ]
    lowered = text.lower()
    return any(signal.lower() in lowered for signal in signals)


def _fetch_with_requests(url: str) -> dict:
    """Fetch page using requests (fast, for static HTML)."""
    resp = requests.get(
        url,
        headers=_get_headers(),
        timeout=20,
        allow_redirects=True,
    )

    if resp.status_code != 200:
        return {"error": f"HTTP {resp.status_code}", "url": url}

    # Decode raw bytes with correct encoding to avoid garbled Chinese
    try:
        html_text = resp.content.decode("utf-8")
    except (UnicodeDecodeError, LookupError):
        encoding = resp.apparent_encoding or resp.encoding or "utf-8"
        html_text = resp.content.decode(encoding, errors="replace")

    text = _extract_text(html_text)
    return {"url": url, "content": text, "content_length": len(text)}


def _fetch_with_playwright(url: str) -> dict:
    """Fetch page using shared Playwright browser (reused across calls)."""
    browser = _get_browser()
    if browser is None:
        return {
            "error": "Playwright not installed. Run: pip install playwright && playwright install chromium",
            "url": url,
        }

    context = None
    try:
        context = browser.new_context(
            user_agent=random.choice(USER_AGENTS),
            viewport={"width": 1280, "height": 800},
            locale="zh-CN",
        )
        page = context.new_page()

        # Navigate and wait for network to settle
        page.goto(url, wait_until="networkidle", timeout=30000)

        # Extra wait for lazy-loaded content
        page.wait_for_timeout(2000)

        html_content = page.content()

        text = _extract_text(html_content)
        return {"url": url, "content": text, "content_length": len(text)}

    except Exception as e:
        return {"error": f"Playwright fetch failed: {str(e)}", "url": url}
    finally:
        if context:
            try:
                context.close()
            except Exception:
                pass


@tool
def fetch_page(url: str) -> str:
    """Fetch a web page and extract its text content.

    Automatically uses Playwright for JS-rendered sites (猎聘, 智联招聘, etc.)
    and falls back to Playwright if requests returns too little content.
    Domains that fail repeatedly are auto-blacklisted to save time.

    Args:
        url: The URL of the web page to fetch.

    Returns:
        JSON string with {url, content, content_length} or {error, url}.
    """
    try:
        # Check runtime blacklist first
        if _is_blacklisted(url):
            domain = _get_domain(url)
            return json.dumps({
                "error": f"Domain {domain} auto-blacklisted (failed {_DOMAIN_FAIL_THRESHOLD}+ times)",
                "url": url,
            }, ensure_ascii=False)

        # Route 1: Known JS-rendered sites → go straight to Playwright
        if _needs_js_render(url):
            print(f"      [Playwright] JS site detected: {url[:60]}")
            result = _fetch_with_playwright(url)
            if result.get("error"):
                _record_failure(url)
                return json.dumps(result, ensure_ascii=False)
            if _looks_like_verification_page(result.get("content", "")):
                _record_failure(url)
                return json.dumps({
                    "error": "Encountered anti-bot verification page",
                    "url": url,
                }, ensure_ascii=False)
            if len(result.get("content", "")) < 50:
                _record_failure(url)
                return json.dumps({
                    "error": "Page content too short even with Playwright rendering",
                    "url": url,
                    "content": result.get("content", ""),
                    "content_length": len(result.get("content", "")),
                }, ensure_ascii=False)
            _record_success(url)
            return json.dumps(result, ensure_ascii=False)

        # Route 2: Normal sites → try requests first
        result = _fetch_with_requests(url)

        if result.get("error"):
            _record_failure(url)
            return json.dumps(result, ensure_ascii=False)

        if _looks_like_verification_page(result.get("content", "")):
            _record_failure(url)
            return json.dumps({
                "error": "Encountered anti-bot verification page",
                "url": url,
            }, ensure_ascii=False)

        # Fallback: if requests got too little content, retry with Playwright (main thread only)
        if len(result.get("content", "")) < 50:
            if _is_main_thread():
                print(f"      [Playwright fallback] Content too short, retrying with JS rendering")
                pw_result = _fetch_with_playwright(url)
                if not pw_result.get("error") and len(pw_result.get("content", "")) >= 50:
                    _record_success(url)
                    return json.dumps(pw_result, ensure_ascii=False)
            _record_failure(url)
            return json.dumps({
                "error": "Page content too short",
                "url": url,
                "content": result.get("content", ""),
                "content_length": len(result.get("content", "")),
            }, ensure_ascii=False)

        _record_success(url)
        return json.dumps(result, ensure_ascii=False)

    except requests.exceptions.Timeout:
        return json.dumps({"error": "Request timed out (20s)", "url": url}, ensure_ascii=False)
    except requests.exceptions.ConnectionError:
        return json.dumps({"error": "Connection failed (site may be blocked or down)", "url": url}, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": f"Fetch failed: {str(e)}", "url": url}, ensure_ascii=False)
