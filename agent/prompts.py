"""Role-specific prompts for the Plan-Act-Observe-Reflect agent architecture.

Two prompts for two LLM-driven roles:
- Strategist: high-level planning based on coverage gaps and reflections
- Reflector: structured self-evaluation after each round

(The executor node is code-driven and uses QUERY_GEN_PROMPT in graph.py)
"""

STRATEGIST_PROMPT = """你是一个AI岗位搜索Agent的**战略规划模块**。你的任务是制定和调整搜索策略。

## 目标
收集至少 {target} 条高质量的 AI Engineer 校园招聘/实习岗位，要求：
- **来源多样**: 覆盖多个招聘网站（牛客网、智联招聘、猎聘、前程无忧、拉勾网、BOSS直聘、实习僧等）
- **地域多样**: 不能只集中在北京/上海，应覆盖深圳、杭州、成都、广州、南京、武汉等城市
- **领域多样**: 覆盖 NLP、CV、推荐系统、大模型、自动驾驶、语音、数据挖掘等 AI 子方向
- **公司多样**: 大厂（BAT、字节、华为）、中厂（美团、京东、网易）、创业公司（商汤、旷视）、研究院

## 当前进展
- 已收集: {job_count}/{target} 条
- 搜索轮次: {iteration}

## 覆盖率分析
{coverage}

## 过去的反思记录
{reflections}

## 已使用的搜索关键词
{searched_queries}

## 你的任务
基于以上信息，输出一个搜索策略，包含 3-5 个**子目标**。

要求：
1. 每个子目标是一个**战略方向**，不是具体的搜索关键词
2. 每个子目标要有明确的**理由**（基于覆盖率缺口或反思发现）
3. 给出**建议的搜索方向**（网站、领域、地域等，但不是具体query）
4. **避免重复**已经搜索过的方向

**重要经验：** 优先选择稳定可抓取的职位详情来源，如牛客网、实习僧、国家大学生就业服务平台、高校就业信息网、公司官方招聘页。BOSS直聘、拉勾网、知乎、公众号、新闻页常出现验证码、聚合页或资讯页，除非明确是职位详情链接，否则不要优先依赖。

请以JSON格式输出：
```json
[
  {{
    "goal": "子目标描述",
    "rationale": "为什么需要这个目标",
    "suggested_approach": "建议的搜索方向"
  }}
]
```
"""

REFLECTOR_PROMPT = """你是一个AI岗位搜索Agent的**反思模块**。每轮执行后，你负责分析效果并提供改进建议。

## 本轮执行摘要
- 执行的策略目标: {executed_goal}
- 新增岗位数: {new_jobs_count}
- 本轮搜索的关键词: {round_queries}

## 累计进度
- 总计收集: {job_count}/{target} 条
- 总搜索轮次: {iteration}

## 当前覆盖率
{coverage}

## 你的任务
分析本轮执行效果，输出结构化反思。请以JSON格式返回：

```json
{{
  "yield_assessment": "high/medium/low",
  "what_worked": "本轮成功的方面",
  "what_failed": "本轮不成功的方面（如果有）",
  "pattern_detected": "发现的规律（如某类关键词效果好、某些网站内容可以正常抓取等）",
  "coverage_gap": "当前最大的覆盖率缺口是什么（来源/地域/领域/公司哪方面最缺）",
  "needs_replan": true或false,
  "replan_reason": "如果需要重新规划，原因是什么",
  "next_suggestion": "对下一轮执行的1-2个具体搜索关键词建议"
}}
```

## 何时建议重新规划（needs_replan: true）
- 连续2轮收益都很低（每轮新增 < 2 条）
- 发现覆盖率严重失衡（某维度占比超过60%）
- 当前策略的剩余目标已经不再有效
- 发现了更好的搜索方向
"""
