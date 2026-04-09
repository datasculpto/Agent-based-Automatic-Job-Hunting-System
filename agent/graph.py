"""LangGraph-based job search agent with Plan-Act-Observe-Reflect architecture.

Graph topology (hybrid code-driven executor):
  strategist → executor → reflector
                              ↓
                strategist (replan) or executor (continue)

The executor calls tools DIRECTLY in code (web_search, fetch_page, analyze_job)
rather than relying on LLM tool-binding. The LLM is used only to generate
search queries based on the current strategy. This makes execution
deterministic and efficient.
"""

import os
import sys
import io
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter
from urllib.parse import urlparse

# Fix Windows console encoding
if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END

from agent.state import AgentState
from agent.prompts import STRATEGIST_PROMPT, REFLECTOR_PROMPT
from tools.web_search import web_search
from tools.page_fetcher import fetch_page
from tools.job_analyzer import analyze_job
from tools.coverage_report import get_coverage_report, update_coverage, track_collected_jobs
from tools.direct_crawl import direct_crawl_all

# Configuration
MAX_ITERATIONS = 40
TARGET_JOBS = 50
REPLAN_INTERVAL = 3
MAX_URLS_PER_ROUND = 20
MAX_SEARCH_QUERIES = 5
MAX_QUERY_ATTEMPTS = 10
MAX_PARALLEL_URLS = 8

# Tools list (kept for reference, but executor calls them directly)
TOOLS = [web_search, fetch_page, analyze_job, get_coverage_report]

RELIABLE_SITE_PATTERNS = [
    "牛客网 {term} 校招 招聘",
    "实习僧 {term} 实习 招聘",
    "{term} 校招 招聘详情 岗位职责",
    "{term} 实习 招聘 任职要求",
    "{term} 校园招聘 2025 应届",
    "{term} 招聘 岗位描述 技能要求",
    "nowcoder {term} 校招",
    "shixiseng {term} 实习",
]

CITY_POOL = ["北京", "上海", "深圳", "杭州", "广州", "成都", "南京", "武汉", "西安", "合肥", "长沙", "苏州"]
COMPANY_POOL = [
    "字节跳动", "腾讯", "美团", "阿里巴巴", "百度", "京东", "华为", "旷视", "商汤", "科大讯飞",
    "网易", "小米", "滴滴", "快手", "蚂蚁集团", "大疆", "小红书", "哔哩哔哩", "蔚来", "理想汽车",
    "地平线", "寒武纪", "第四范式", "云从科技", "依图科技", "深信服", "中兴",
]
FOCUS_TERM_GROUPS = [
    (("大模型", "llm", "aigc", "多模态", "生成式"), ["大模型", "LLM", "AIGC"]),
    (("nlp", "自然语言处理", "语音", "asr", "tts", "对话"), ["自然语言处理", "语音识别", "对话系统"]),
    (("cv", "视觉", "感知", "自动驾驶", "机器人", "图像"), ["计算机视觉", "自动驾驶", "图像识别"]),
    (("推荐", "搜索排序", "ranking", "广告", "数据挖掘"), ["推荐系统", "搜索排序", "数据挖掘"]),
    (("infra", "mlops", "基础设施", "训练推理", "部署", "平台"), ["AI基础设施", "MLOps", "AI平台"]),
    (("强化学习", "rl", "决策", "博弈"), ["强化学习", "智能决策"]),
    (("安全", "隐私", "联邦"), ["AI安全", "隐私计算"]),
]

# Prompt for LLM to generate search queries only
QUERY_GEN_PROMPT = """你是一个搜索关键词生成专家。根据当前策略目标，生成2-3个高质量的搜索关键词。

## 当前策略目标
{current_goal}

## 策略理由
{goal_rationale}

## 建议方向
{suggested_approach}

## 进度: {job_count}/{target} 条

## 已搜索过的关键词（不要重复使用！）
{searched_queries}

## 已失败的信息
{failed_info}

## 要求
1. 生成2-3个**全新**的搜索关键词，不要和已搜索过的重复
2. 关键词应该能在Bing/搜狗/百度上找到AI校招/实习岗位
3. **禁止使用 site: 操作符**，这在大多数搜索引擎上不工作
4. 高效关键词模板参考：
   - "美团 算法工程师 校招 2025"
   - "深圳 AI算法 校园招聘 招聘详情"
   - "大模型 应届 招聘 岗位职责"
   - "推荐系统 算法 校招 任职要求"
   - "商汤科技 校园招聘 AI 岗位详情"
   - "牛客网 NLP算法 校招"
   - "实习僧 机器学习 实习"
   - "字节跳动 算法 校招 2025 招聘"
5. 只返回JSON数组格式，每个元素是一个关键词字符串

```json
["关键词1", "关键词2", "关键词3"]
```
"""


def create_llm(temperature: float = 0.3, max_tokens: int = 4096) -> ChatOpenAI:
    """Create a DeepSeek LLM instance (used for all nodes)."""
    return ChatOpenAI(
        model="deepseek-chat",
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        temperature=temperature,
        max_tokens=max_tokens,
        max_retries=3,
        request_timeout=120,
    )


def _compute_coverage(collected_jobs: list[str]) -> dict:
    """Compute coverage analytics from collected jobs."""
    if not collected_jobs:
        return {"total_jobs": 0}

    by_source: Counter = Counter()
    by_location: Counter = Counter()
    by_domain: Counter = Counter()
    companies: list[str] = []

    domain_keywords = {
        "NLP": ["nlp", "自然语言", "文本", "对话", "语义"],
        "CV": ["cv", "计算机视觉", "图像", "视觉", "目标检测"],
        "大模型/LLM": ["大模型", "llm", "gpt", "生成式", "aigc", "多模态"],
        "推荐系统": ["推荐", "搜索", "广告", "ranking"],
        "语音": ["语音", "asr", "tts", "音频"],
        "自动驾驶": ["自动驾驶", "无人驾驶", "自驾"],
        "机器学习": ["机器学习", "ml", "数据挖掘", "深度学习"],
        "AI基础设施": ["平台", "框架", "mlops", "训练", "推理", "部署"],
    }

    for job_json in collected_jobs:
        try:
            job = json.loads(job_json)
        except Exception:
            continue
        by_source[job.get("source", "未知")] += 1
        by_location[job.get("location", "未知")] += 1
        companies.append(job.get("company", "未知"))
        title = job.get("title", "").lower()
        tags = " ".join(job.get("tech_tags", [])).lower()
        combined = f"{title} {tags}"
        for domain, keywords in domain_keywords.items():
            if any(kw in combined for kw in keywords):
                by_domain[domain] += 1

    return {
        "total_jobs": len(collected_jobs),
        "by_source": dict(by_source.most_common()),
        "by_location": dict(by_location.most_common(15)),
        "by_domain": dict(by_domain.most_common()),
        "unique_companies": len(set(companies)),
        "top_companies": dict(Counter(companies).most_common(10)),
    }


# ============================================================
# Node 1: STRATEGIST
# ============================================================

def strategist_node(state: AgentState) -> dict:
    llm = create_llm(temperature=0.4)

    coverage = json.loads(state.get("coverage") or "{}")
    coverage_str = json.dumps(coverage, ensure_ascii=False, indent=2) if coverage else "尚无数据（首次规划）"

    reflections = state.get("reflections", [])
    reflections_str = "\n".join(f"- Round {i+1}: {r}" for i, r in enumerate(reflections[-5:])) if reflections else "尚无反思记录（首次规划）"

    searched = state.get("searched_queries", [])
    queries_str = ", ".join(searched[-15:]) if searched else "尚无"

    prompt = STRATEGIST_PROMPT.format(
        target=TARGET_JOBS, job_count=state["job_count"], iteration=state["iteration"],
        coverage=coverage_str, reflections=reflections_str, searched_queries=queries_str,
    )

    print(f"\n{'='*60}")
    print(f"[STRATEGY] Round {state['iteration'] + 1} | Jobs: {state['job_count']}/{TARGET_JOBS}")
    print(f"{'='*60}")

    response = llm.invoke([
        SystemMessage(content="你是一个搜索策略规划专家。只返回JSON数组格式的策略计划。"),
        HumanMessage(content=prompt),
    ])

    plan_text = response.content.strip()
    if "```json" in plan_text:
        plan_text = plan_text.split("```json")[1].split("```")[0].strip()
    elif "```" in plan_text:
        plan_text = plan_text.split("```")[1].split("```")[0].strip()

    try:
        plan = json.loads(plan_text)
        if not isinstance(plan, list):
            plan = [plan]
    except json.JSONDecodeError:
        plan = [
            {"goal": "搜索主流招聘网站的AI校招岗位", "rationale": "广撒网获取基础数据", "suggested_approach": "使用不同网站搜索"},
            {"goal": "搜索特定AI方向的岗位", "rationale": "提高覆盖领域多样性", "suggested_approach": "使用NLP/CV/大模型等方向关键词"},
        ]

    for i, step in enumerate(plan):
        print(f"  Plan {i+1}: {step.get('goal', '?')}")
        print(f"    Why: {step.get('rationale', '?')}")

    return {"current_plan": json.dumps(plan, ensure_ascii=False), "plan_step_index": 0, "status": "running"}



def _is_duplicate_url(url: str, processed_set: set[str]) -> bool:
    """Check if a URL (normalized) has already been processed."""
    base = url.split("?")[0].rstrip("/")
    return any(p.split("?")[0].rstrip("/") == base for p in processed_set)


def _get_url_host(url: str) -> str:
    """Extract the host part of a URL for diversity scoring."""
    try:
        return urlparse(url).netloc.lower()
    except Exception:
        return ""


def _unique_preserve_order(items: list[str]) -> list[str]:
    """Deduplicate strings while preserving their original order."""
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        normalized = item.strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        result.append(normalized)
    return result


def _extract_queries_from_response(text: str) -> list[str]:
    """Parse a list of queries from an LLM response."""
    raw = text.strip()
    if "```json" in raw:
        raw = raw.split("```json", 1)[1].split("```", 1)[0].strip()
    elif "```" in raw:
        raw = raw.split("```", 1)[1].split("```", 1)[0].strip()

    try:
        queries = json.loads(raw)
        if not isinstance(queries, list):
            queries = [str(queries)]
        return _unique_preserve_order([str(query) for query in queries])
    except json.JSONDecodeError:
        return _unique_preserve_order([raw[:80]]) if raw else []


def _pick_focus_terms(current_step: dict) -> list[str]:
    """Choose search focus terms from the current strategic goal."""
    context = " ".join([
        current_step.get("goal", ""),
        current_step.get("rationale", ""),
        current_step.get("suggested_approach", ""),
    ]).lower()

    focus_terms: list[str] = []
    for triggers, terms in FOCUS_TERM_GROUPS:
        if any(trigger in context for trigger in triggers):
            focus_terms.extend(terms)

    focus_terms.extend(["AI算法工程师", "算法工程师", "机器学习", "大模型"])
    return _unique_preserve_order(focus_terms)


def _pick_gap_city(coverage: dict, iteration: int) -> str:
    """Pick a city that is underrepresented in the current coverage."""
    existing_locations = " ".join((coverage.get("by_location") or {}).keys())
    for city in CITY_POOL:
        if city not in existing_locations:
            return city
    return CITY_POOL[iteration % len(CITY_POOL)]


def _pick_company_hint(current_step: dict, iteration: int) -> str:
    """Choose a company keyword for targeted official-channel style searches."""
    context = " ".join([
        current_step.get("goal", ""),
        current_step.get("rationale", ""),
        current_step.get("suggested_approach", ""),
    ])
    for company in COMPANY_POOL:
        if company in context:
            return company
    return COMPANY_POOL[iteration % len(COMPANY_POOL)]


def _build_template_queries(state: AgentState, current_step: dict, coverage: dict) -> list[str]:
    """Generate deterministic, reliable-site-first queries."""
    focus_terms = _pick_focus_terms(current_step)
    city = _pick_gap_city(coverage, state["iteration"])
    company = _pick_company_hint(current_step, state["iteration"])

    query_candidates: list[str] = []
    base_terms = ["AI算法工程师", "算法工程师", "机器学习", "深度学习", "大模型"]
    for index, pattern in enumerate(RELIABLE_SITE_PATTERNS):
        term = base_terms[index % len(base_terms)]
        query_candidates.append(pattern.format(term=term))

    primary_term = focus_terms[0]
    secondary_term = focus_terms[min(1, len(focus_terms) - 1)]
    query_candidates.extend([
        f"牛客网 {primary_term} 校招 招聘",
        f"实习僧 {secondary_term} 实习 招聘",
        f"{company} {primary_term} 校园招聘",
        f"{company} 算法工程师 校招 2025",
        f"{city} AI算法工程师 校园招聘",
        f"{city} {primary_term} 实习 招聘",
        f"{city} {secondary_term} 校招 招聘",
        f"{primary_term} 应届生 招聘 岗位详情",
        f"{secondary_term} 校招 2025 岗位职责 任职要求",
        f"{company} {secondary_term} 实习 2025",
    ])

    searched = set(state.get("searched_queries", []))
    return [query for query in _unique_preserve_order(query_candidates) if query not in searched]


def _looks_like_ai_job(job: dict) -> bool:
    """Final sanity check to prevent obviously non-AI roles from slipping through.

    Default to True (trust DeepSeek's judgment) unless the title clearly
    indicates a non-AI role.
    """
    negative_terms = [
        "前端", "后端", "测试", "运维", "产品经理", "运营", "销售", "财务", "行政",
        "hr", "人力", "市场", "客服", "会计", "法务",
    ]

    title = (job.get("title", "") or "").lower()

    # Only reject if the title clearly matches a non-AI role
    if any(term in title for term in negative_terms):
        return False
    # Trust LLM judgment for everything else
    return True


def _generate_llm_queries(state: AgentState, current_step: dict, searched_str: str, failed_str: str) -> list[str]:
    """Generate supplemental search queries with the LLM."""
    llm = create_llm(temperature=0.3, max_tokens=1024)
    query_prompt = QUERY_GEN_PROMPT.format(
        current_goal=current_step.get("goal", "搜索AI校招岗位"),
        goal_rationale=current_step.get("rationale", ""),
        suggested_approach=current_step.get("suggested_approach", ""),
        job_count=state["job_count"], target=TARGET_JOBS,
        searched_queries=searched_str, failed_info=failed_str,
    )

    try:
        response = llm.invoke([
            SystemMessage(content="你是一个搜索关键词生成专家。只返回JSON数组格式的关键词列表。"),
            HumanMessage(content=query_prompt),
        ])
    except Exception as exc:
        print(f"  [!] Query generation error: {exc}")
        return []

    return _extract_queries_from_response(response.content)


def _process_candidate_url(candidate: tuple[str, str, str, str, int]) -> dict:
    """Fetch and analyze a single candidate URL."""
    url, title, source, snippet, score = candidate
    logs = [f"  -> fetch_page({url[:80]})"]

    try:
        fetch_raw = fetch_page.invoke({"url": url})
        fetch_result = json.loads(fetch_raw) if isinstance(fetch_raw, str) else fetch_raw
    except Exception as exc:
        logs.append(f"    [!] Fetch exception: {exc}")
        return {"url": url, "status": "fetch_error", "logs": logs}

    if fetch_result.get("error"):
        logs.append(f"    [!] Fetch error: {fetch_result['error']}")
        return {"url": url, "status": "fetch_error", "logs": logs}

    content = fetch_result.get("content", "")
    if not content or len(content) < 50:
        logs.append("    [!] Content too short, skipping")
        return {"url": url, "status": "fetch_error", "logs": logs}

    logs.append(f"    -> analyze_job(content_len={len(content)}, source={source})")
    try:
        analyze_raw = analyze_job.invoke({
            "content": content,
            "url": url,
            "source": source,
        })
        analyze_result = json.loads(analyze_raw) if isinstance(analyze_raw, str) else analyze_raw
    except Exception as exc:
        logs.append(f"    [!] Analyze exception: {exc}")
        return {"url": url, "status": "analyze_error", "logs": logs}

    if analyze_result.get("error"):
        logs.append(f"    [!] Analyze error: {analyze_result['error']}")
        return {"url": url, "status": "analyze_error", "logs": logs}

    return {
        "url": url,
        "status": "ok",
        "logs": logs,
        "analyze_result": analyze_result,
    }


# ============================================================
# Node 2: EXECUTOR — Hybrid code-driven approach
# ============================================================

def executor_node(state: AgentState) -> dict:
    """Executor: uses LLM only for query generation, calls tools directly in code.

    Flow per round:
      1. LLM generates 2-3 search queries
      2. Code calls web_search.invoke() for each query
      3. Code selects up to MAX_URLS_PER_ROUND promising URLs
      4. Code calls fetch_page.invoke() + analyze_job.invoke() for each URL
      5. Returns all results as state updates
    """
    # --- Parse current plan step ---
    try:
        plan = json.loads(state.get("current_plan") or "[]")
    except json.JSONDecodeError:
        plan = []

    step_idx = state.get("plan_step_index", 0)
    current_step = plan[step_idx] if step_idx < len(plan) else (plan[0] if plan else {
        "goal": "搜索AI校招岗位", "rationale": "继续收集", "suggested_approach": "尝试不同关键词",
    })
    coverage = json.loads(state.get("coverage") or "{}")

    searched = list(state.get("searched_queries", []))
    searched_str = "\n".join(f"  - {q}" for q in searched[-30:]) if searched else "  （尚无搜索记录）"

    failed_info_parts = []
    for r_str in state.get("reflections", [])[-3:]:
        try:
            r = json.loads(r_str)
            if r.get("what_failed"):
                failed_info_parts.append(r["what_failed"])
            if r.get("pattern_detected"):
                failed_info_parts.append(r["pattern_detected"])
        except Exception:
            pass
    failed_str = "\n".join(f"  - {f}" for f in failed_info_parts) if failed_info_parts else "  （暂无失败记录）"

    processed_set = set(state.get("processed_urls", []))

    print(f"\n[EXECUTE] Goal: {current_step.get('goal', '?')}")

    # === Step 1: Build reliable template queries first, then let the LLM supplement ===
    template_queries = _build_template_queries(state, current_step, coverage)
    llm_queries = _generate_llm_queries(state, current_step, searched_str, failed_str)
    query_pool = _unique_preserve_order(template_queries + llm_queries)
    candidate_queries = [query for query in query_pool if query not in searched][:MAX_QUERY_ATTEMPTS]
    if not candidate_queries:
        candidate_queries = query_pool[:1] if query_pool else template_queries[:1]

    print(f"  Generated queries: {candidate_queries}")

    # === Step 2: Execute searches and collect URLs ===
    all_search_results = []
    new_searched_queries = []

    for query in candidate_queries:
        if len(new_searched_queries) >= MAX_SEARCH_QUERIES:
            break

        print(f"  -> web_search({query})")
        try:
            raw_result = web_search.invoke({"query": query})
            result = json.loads(raw_result) if isinstance(raw_result, str) else raw_result

            if result.get("error"):
                print(f"    [!] Search error: {result['error']}")
            else:
                results_list = result.get("results", [])
                print(f"    Got {len(results_list)} results from {result.get('engine', '?')}")
                new_searched_queries.append(query)
                all_search_results.extend(results_list)
        except Exception as e:
            print(f"    [!] Search exception: {e}")

    # === Step 2.5: Direct crawl known sites (bypass search engines) ===
    # Always use direct crawling with rotation of keywords per round
    focus_terms = _pick_focus_terms(current_step)
    # Rotate keywords based on iteration to get different results each round
    all_crawl_keywords = [
        "AI算法", "机器学习", "深度学习", "大模型", "NLP", "算法工程师",
        "推荐算法", "计算机视觉", "LLM", "自然语言处理", "自动驾驶",
        "语音识别", "数据挖掘", "强化学习", "图像识别", "智能驾驶",
        "搜索推荐", "多模态", "AIGC", "机器人算法",
    ]
    # Pick a different subset each round
    start = (state["iteration"] * 4) % len(all_crawl_keywords)
    round_keywords = (all_crawl_keywords[start:start+6] + focus_terms[:4])
    round_keywords = _unique_preserve_order(round_keywords)

    direct_results = direct_crawl_all(round_keywords[:8])
    added_direct = 0
    for item in direct_results:
        if not _is_duplicate_url(item["link"], processed_set):
            all_search_results.append(item)
            added_direct += 1
    if added_direct:
        print(f"  After direct crawl: +{added_direct} new URLs ({len(all_search_results)} total)")

    # === Step 3: Select promising URLs (filter duplicates and JS-rendered sites) ===
    candidate_urls = []
    seen_bases = set()
    preferred_hosts = [
        "zhipin.com",
        "liepin.com",
        "lagou.com",
        "zhaopin.com",
        "nowcoder.com",
        "shixiseng.com",
        "51job.com",
        "jobs.bytedance.com",
        "careers.tencent.com",
        "join.qq.com",
        "campus.meituan.com",
        "talent.baidu.com",
        "xiaomi.com",
        "huawei.com",
        "ncss.cn",
        "job.163.com",
        "kanzhun.com",
        "maimai.cn",
    ]
    blocked_hosts = [
        "cloud.tencent.com",
        "xinhuanet.com",
        "baijiahao.baidu.com",
        "wappass.baidu.com",
        "pconline.com.cn",
        "book118.com",
        "sohu.com",
    ]
    low_value_patterns = [
        "/discuss/",
        "/feed/main/detail/",
        "/news/",
        "/users/",
        "/subject/index/",
        "/school/schedule",
        "/login",
        "safe/verify-slider",
        "wappass.baidu.com/static/captcha",
    ]

    # Priority keywords for relevance filtering
    priority_keywords = ["AI", "算法", "机器学习", "深度学习", "大模型", "校招", "实习",
                         "NLP", "CV", "推荐", "自动驾驶", "语音", "校园招聘", "应届",
                         "engineer", "intern", "campus"]

    for item in all_search_results:
        url = item.get("link", "")
        title = item.get("title", "")
        if not url:
            continue

        # Skip already-processed URLs
        if _is_duplicate_url(url, processed_set):
            continue

        # Normalize to avoid duplicates within this batch
        base = url.split("?")[0].rstrip("/")
        if base in seen_bases:
            continue
        seen_bases.add(base)

        url_lower = url.lower()
        host = _get_url_host(url)
        if any(blocked in host for blocked in blocked_hosts):
            continue

        # Score by relevance using both title and snippet.
        title_lower = title.lower()
        snippet_lower = item.get("snippet", "").lower()
        combined_text = f"{title_lower} {snippet_lower}"
        score = sum(1 for kw in priority_keywords if kw.lower() in combined_text)

        # Boost URLs that look like job detail pages
        detail_patterns = ["/jobs/detail/", "/job_detail/", "/intern/inn_", "/position/",
                           "/jobdetail/", "/job/detail/", "/joinus/detail/", "/post-read",
                           "jobid=", "positionid=", "/campus_apply/", "/detail?jobid="]
        if any(p in url_lower for p in detail_patterns):
            score += 3

        # Penalize URLs that look like listing/aggregation pages
        listing_patterns = ["/schedule", "/campus", "/company/", "/recommend/",
                            "/category/", "/zhaopin/", "/interns?", "/search"]
        if any(p in url_lower for p in listing_patterns):
            score -= 2

        if any(host in url_lower for host in preferred_hosts):
            score += 3

        if any(pattern in url_lower for pattern in low_value_patterns):
            score -= 4

        if not any(hostname in host for hostname in preferred_hosts) and score < 1:
            continue

        candidate_urls.append((url, title, item.get("source", "未知"), item.get("snippet", ""), score))

    # Sort by relevance and keep per-host diversity so one site cannot dominate the round.
    candidate_urls.sort(key=lambda x: x[4], reverse=True)
    selected_urls = []
    host_counts: Counter = Counter()
    for candidate in candidate_urls:
        host = _get_url_host(candidate[0])
        host_cap = 3 if any(domain in host for domain in preferred_hosts) else 2
        if host and host_counts[host] >= host_cap:
            continue
        selected_urls.append(candidate)
        if host:
            host_counts[host] += 1
        if len(selected_urls) >= MAX_URLS_PER_ROUND:
            break

    print(f"  Selected {len(selected_urls)} URLs to process (from {len(candidate_urls)} candidates)")

    # === Step 4: Fetch and analyze each URL ===
    # Track dedup state for this round
    existing_keys: set[str] = set()
    existing_urls: set[str] = set()
    for jstr in state["collected_jobs"]:
        try:
            j = json.loads(jstr)
            existing_keys.add(f"{j.get('company', '').strip().lower()}|{j.get('title', '').strip().lower()}")
            existing_urls.add(j.get("job_url", "").strip().rstrip("/"))
        except Exception:
            pass

    new_jobs = []
    new_processed_urls = []
    new_count = 0
    dupes = 0
    fetch_errors = 0
    analyze_errors = 0

    ordered_results: dict[int, dict] = {}
    if selected_urls:
        max_workers = min(MAX_PARALLEL_URLS, len(selected_urls))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_map = {
                executor.submit(_process_candidate_url, candidate): index
                for index, candidate in enumerate(selected_urls)
            }
            for future in as_completed(future_map):
                index = future_map[future]
                try:
                    ordered_results[index] = future.result()
                except Exception as exc:
                    ordered_results[index] = {
                        "url": selected_urls[index][0],
                        "status": "analyze_error",
                        "logs": [f"    [!] Worker exception: {exc}"],
                    }

    for index in range(len(selected_urls)):
        result = ordered_results.get(index)
        if not result:
            continue

        for log in result.get("logs", []):
            print(log)

        status = result.get("status")
        if status == "fetch_error":
            fetch_errors += 1
            continue
        if status == "analyze_error":
            analyze_errors += 1
            continue

        url = result["url"]
        if url not in processed_set:
            new_processed_urls.append(url)
            processed_set.add(url)

        analyze_result = result.get("analyze_result", {})
        if analyze_result.get("is_relevant") and analyze_result.get("job"):
            job = analyze_result["job"]
            if not _looks_like_ai_job(job):
                print(f"    [-] Filtered after sanity check: {job.get('title', '?')}")
                continue
            job_url = job.get("job_url", "").strip().rstrip("/")
            job_key = f"{job.get('company', '').strip().lower()}|{job.get('title', '').strip().lower()}"

            if job_url in existing_urls or job_key in existing_keys:
                dupes += 1
                print(f"    [=] Duplicate: {job.get('title', '?')} @ {job.get('company', '?')}")
            elif state["job_count"] + new_count < TARGET_JOBS:
                existing_urls.add(job_url)
                existing_keys.add(job_key)
                new_jobs.append(json.dumps(job, ensure_ascii=False))
                new_count += 1
                print(f"    [+] {job.get('title', '?')} @ {job.get('company', '?')} ({job.get('source', '?')})")
        else:
            reason = analyze_result.get("reason", "not relevant")
            print(f"    [-] Not relevant: {reason[:60]}")

    # === Step 5: Build state updates ===
    error_info = f", {fetch_errors} fetch errors, {analyze_errors} analyze errors" if (fetch_errors or analyze_errors) else ""
    print(f"  Round Summary: {len(new_searched_queries)} searches, +{new_count} new jobs, {dupes} dupes{error_info} (total: {state['job_count'] + new_count})")

    # Compute coverage
    all_jobs = list(state["collected_jobs"]) + new_jobs
    coverage = _compute_coverage(all_jobs)
    coverage["last_round_yield"] = new_count
    update_coverage(coverage)
    track_collected_jobs(all_jobs)

    # Build a summary message for the message history
    summary = (
        f"本轮执行完成。搜索了 {len(new_searched_queries)} 个关键词，"
        f"处理了 {len(selected_urls)} 个URL，"
        f"新增 {new_count} 条岗位，{dupes} 条重复。"
        f"当前总计: {state['job_count'] + new_count}/{TARGET_JOBS} 条。"
    )

    return {
        "messages": [AIMessage(content=summary)],
        "collected_jobs": new_jobs,
        "job_count": state["job_count"] + new_count,
        "searched_queries": new_searched_queries,
        "processed_urls": new_processed_urls,
        "coverage": json.dumps(coverage, ensure_ascii=False),
        "plan_step_index": state.get("plan_step_index", 0) + 1,
        "iteration": state["iteration"] + 1,
    }


# ============================================================
# Node 3: REFLECTOR
# ============================================================

def reflector_node(state: AgentState) -> dict:
    llm = create_llm(temperature=0.3, max_tokens=1000)

    try:
        plan = json.loads(state.get("current_plan") or "[]")
    except json.JSONDecodeError:
        plan = []

    step_idx = max(0, state.get("plan_step_index", 1) - 1)
    executed_step = plan[step_idx] if step_idx < len(plan) else {"goal": "搜索AI岗位"}

    coverage = json.loads(state.get("coverage") or "{}")
    coverage_str = json.dumps(coverage, ensure_ascii=False, indent=2)
    round_queries = state.get("searched_queries", [])[-3:]
    queries_str = ", ".join(round_queries) if round_queries else "无"
    new_jobs = coverage.get("last_round_yield", 0)

    prompt = REFLECTOR_PROMPT.format(
        executed_goal=executed_step.get("goal", "?"), new_jobs_count=new_jobs,
        round_queries=queries_str, job_count=state["job_count"],
        target=TARGET_JOBS, iteration=state["iteration"], coverage=coverage_str,
    )

    print(f"\n[REFLECT] Round {state['iteration']} | +{new_jobs} jobs this round")

    response = llm.invoke([
        SystemMessage(content="你是一个搜索效果分析专家。只返回JSON格式的反思分析。"),
        HumanMessage(content=prompt),
    ])

    reflect_text = response.content.strip()
    if "```json" in reflect_text:
        reflect_text = reflect_text.split("```json")[1].split("```")[0].strip()
    elif "```" in reflect_text:
        reflect_text = reflect_text.split("```")[1].split("```")[0].strip()

    try:
        reflection = json.loads(reflect_text)
    except json.JSONDecodeError:
        reflection = {"yield_assessment": "medium", "what_worked": "", "what_failed": "",
                      "pattern_detected": "", "coverage_gap": "", "needs_replan": False,
                      "replan_reason": "", "next_suggestion": "继续执行当前策略"}

    yield_emoji = {"high": "++", "medium": "+-", "low": "--"}.get(reflection.get("yield_assessment", "medium"), "+-")
    print(f"  Yield: {yield_emoji} ({reflection.get('yield_assessment', '?')})")
    if reflection.get("what_worked"):
        print(f"  Worked: {reflection['what_worked']}")
    if reflection.get("coverage_gap"):
        print(f"  Gap: {reflection['coverage_gap']}")
    if reflection.get("needs_replan"):
        print(f"  >>> REPLAN: {reflection.get('replan_reason', '?')}")

    return {"reflections": [json.dumps(reflection, ensure_ascii=False)], "status": "reflecting"}


# ============================================================
# Conditional Edges
# ============================================================

def after_reflector(state):
    if state["job_count"] >= TARGET_JOBS:
        print(f"\n{'='*60}\nTARGET REACHED: {state['job_count']} jobs collected!\n{'='*60}")
        return "end"
    if state["iteration"] >= MAX_ITERATIONS:
        print(f"\n{'='*60}\nMAX ITERATIONS: {state['iteration']} rounds, {state['job_count']} jobs\n{'='*60}")
        return "end"

    needs_replan = False
    reflections = state.get("reflections", [])

    # Check if reflector explicitly requested replan
    if reflections:
        try:
            if json.loads(reflections[-1]).get("needs_replan"):
                needs_replan = True
        except Exception:
            pass

    # Periodic replan every REPLAN_INTERVAL rounds
    if state["iteration"] > 0 and state["iteration"] % REPLAN_INTERVAL == 0:
        needs_replan = True

    # Zero yield this round → replan
    coverage = json.loads(state.get("coverage") or "{}")
    if coverage.get("last_round_yield", 0) == 0 and state["iteration"] > 1:
        needs_replan = True

    return "strategist" if needs_replan else "executor"


# ============================================================
# Build Graph
# ============================================================

def build_graph():
    """Build the PAOR agent graph: strategist -> executor -> reflector (simplified, no ToolNode)."""
    graph = StateGraph(AgentState)

    graph.add_node("strategist", strategist_node)
    graph.add_node("executor", executor_node)
    graph.add_node("reflector", reflector_node)

    graph.set_entry_point("strategist")
    graph.add_edge("strategist", "executor")
    graph.add_edge("executor", "reflector")
    graph.add_conditional_edges("reflector", after_reflector, {"strategist": "strategist", "executor": "executor", "end": END})

    return graph.compile()


def get_initial_state() -> AgentState:
    return AgentState(
        messages=[], collected_jobs=[], job_count=0, searched_queries=[],
        processed_urls=[], current_plan="", plan_step_index=0,
        reflections=[], coverage="{}", iteration=0, status="running",
    )
