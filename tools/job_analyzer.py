"""LLM-based job analysis tool for extracting structured info and judging relevance.

This is a granular tool — the agent explicitly chooses which page content
to analyze, rather than having it done automatically in a batch.
"""

import os
import json
from langchain_core.tools import tool


ANALYSIS_PROMPT = """你是一个专业的AI岗位分析助手。请分析以下职位信息，完成两个任务：

**任务1：判断相关性**
判断这个岗位是否同时满足以下条件：
1. 与 AI / 机器学习 / 深度学习 / 大模型(LLM) / 数据智能 / 算法工程 / NLP / CV / 推荐系统 相关
2. 面向 应届生 / 校招 / 实习生 / 校园招聘（如果无法确定，但岗位本身是AI相关的，也可以保留）
3. **必须是具体的岗位详情页**，而不是岗位列表页、导航页、分类索引页或搜索结果页。如果页面包含多个不同岗位的链接/标题但没有某一个岗位的详细描述，则判定为不相关，reason填写"这是聚合页/导航页，不是具体岗位详情"

**任务2：提取结构化信息**
如果岗位相关，请提取以下字段：

请以JSON格式返回，格式如下：
```json
{{
    "is_relevant": true,
    "reason": "判断理由（简短）",
    "job": {{
        "title": "职位名称",
        "company": "公司名称",
        "location": "工作地点，未知则填'未知'",
        "salary": "薪资范围，未知则填'未公开'",
        "tech_tags": ["技术标签1", "技术标签2"],
        "requirements": "核心技能要求摘要（50字以内）"
    }}
}}
```

如果不相关，只需返回：
```json
{{
    "is_relevant": false,
    "reason": "判断理由"
}}
```

---
**职位来源网站**: {source}
**职位链接**: {url}
**职位页面内容**:
{content}
"""


@tool
def analyze_job(content: str, url: str, source: str) -> str:
    """Use LLM to analyze a job posting and extract structured information.

    Judges whether the posting is relevant to AI Engineer campus recruitment,
    and if so, extracts title, company, location, salary, tech tags, etc.

    Args:
        content: The raw text content of the job posting page.
        url: The URL of the job posting.
        source: The source website name (e.g. "BOSS直聘", "牛客网").

    Returns:
        JSON string with {is_relevant, reason, job} or {is_relevant: false, reason}.
    """
    try:
        from openai import OpenAI
    except ImportError:
        return json.dumps({
            "error": "openai package is not installed. Run: pip install -r requirements.txt",
        }, ensure_ascii=False)

    client = OpenAI(
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
    )
    model = "deepseek-chat"

    prompt = ANALYSIS_PROMPT.format(
        source=source,
        url=url,
        content=content[:5000],
    )

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "你是一个精确的岗位信息分析助手，只返回JSON格式的结果。"},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            max_tokens=1000,
            timeout=120,
        )

        result_text = response.choices[0].message.content.strip()

        # Extract JSON from possible markdown code block
        if "```json" in result_text:
            result_text = result_text.split("```json")[1].split("```")[0].strip()
        elif "```" in result_text:
            result_text = result_text.split("```")[1].split("```")[0].strip()

        result = json.loads(result_text)

        # Attach url and source to the job info
        if result.get("is_relevant") and result.get("job"):
            result["job"]["job_url"] = url
            result["job"]["source"] = source

        return json.dumps(result, ensure_ascii=False)

    except json.JSONDecodeError:
        return json.dumps({
            "error": "LLM returned invalid JSON",
            "raw_response": result_text[:500],
        }, ensure_ascii=False)
    except Exception as e:
        return json.dumps({
            "error": f"Analysis failed: {str(e)}",
        }, ensure_ascii=False)
