# AI Engineer Job Search Agent

基于 LangGraph 的 Agentic AI 求职助手，自动搜索、筛选、整理 AI Engineer 校招岗位信息。

## 架构设计

```
┌──────────────────────────────────────────────────────────┐
│              Plan-Act-Observe-Reflect Architecture        │
│              (LangGraph StateGraph)                       │
│                                                          │
│   Strategist → Executor ⇄ Tools → Processor → Reflector │
│      ↑                                        │          │
│      └────────────────────────────────────────┘          │
├──────────────┬──────────────┬────────────────────────────┤
│  web_search  │  fetch_page  │  analyze_job               │
│  (Baidu)     │  (requests)  │  (DeepSeek LLM)            │
└──────────────┴──────────────┴────────────────────────────┘
```

### Agent 核心能力

| 能力 | 实现方式 |
|------|---------|
| 🧭 任务规划 | Strategist 节点根据覆盖率分析自主制定搜索策略 |
| 🌐 工具调用 | 4个LangChain Tool: web_search, fetch_page, analyze_job, get_coverage_report |
| 🔁 迭代搜索 | Reflector 循环评估，不足50条自动调整关键词继续搜索 |
| 🧠 语义判断 | DeepSeek LLM 判断岗位是否属于AI Engineer方向 |
| 🧹 数据清洗 | Pydantic模型 + LLM结构化提取 + URL/公司+职位双重去重 |
| 📦 结果汇总 | 输出标准化 JSON + CSV，附覆盖率分析 |
| 🔄 自我反思 | Reflector 分析每轮效果，检测搜索模式，建议策略调整 |

### 技术栈

- **Agent框架**: LangGraph (Plan-Act-Observe-Reflect 状态机)
- **LLM**: DeepSeek (OpenAI兼容API，用于规划/执行/反思/分析)
- **搜索引擎**: Baidu (免费，无需API Key，中文结果最优)
- **网页解析**: requests + BeautifulSoup4
- **数据模型**: Pydantic v2

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置环境变量

编辑 `.env` 文件，填入你的API密钥：

```
DEEPSEEK_API_KEY=your_deepseek_api_key
DEEPSEEK_BASE_URL=https://api.deepseek.com
```

> 注：搜索使用百度，无需额外API Key。

### 3. 运行Agent

```bash
python main.py
```

### 4. 查看结果

结果保存在 `output/` 目录下：
- `jobs.json` - 结构化JSON格式
- `jobs.csv` - CSV表格格式

## 输出数据结构

```json
{
  "title": "AI算法工程师（校招）",
  "company": "字节跳动",
  "location": "北京",
  "salary": "30-60K",
  "tech_tags": ["LLM", "NLP", "深度学习", "PyTorch"],
  "requirements": "熟悉深度学习框架，有NLP/CV项目经验",
  "source": "BOSS直聘",
  "job_url": "https://..."
}
```

## 项目结构

```
Job_Search_Agent/
├── main.py                 # 入口文件
├── agent/
│   ├── graph.py            # LangGraph Agent 核心逻辑 (PAOR架构)
│   ├── state.py            # Agent 状态定义
│   └── prompts.py          # 三角色提示词 (Strategist/Executor/Reflector)
├── tools/
│   ├── web_search.py       # 百度搜索工具 (免费)
│   ├── page_fetcher.py     # 网页抓取工具
│   ├── job_analyzer.py     # LLM 岗位分析工具
│   └── coverage_report.py  # 覆盖率报告工具
├── models/
│   └── job.py              # 岗位数据模型 (Pydantic)
├── utils/
│   └── dedup.py            # 去重工具
└── output/                 # 输出结果目录
```

## Agent 工作流程

1. **规划阶段 (Strategist)**: 分析覆盖率缺口和过往反思，制定3-5个搜索子目标
2. **执行阶段 (Executor)**: 使用百度搜索+网页抓取+LLM分析的工具链执行策略
3. **处理阶段 (Processor)**: 提取新增岗位，URL/标题双重去重，更新覆盖率统计
4. **反思阶段 (Reflector)**: 评估本轮效果，发现模式，决定是否需要重新规划
5. **迭代**: 不足50条则回到规划或继续执行，直到达成目标
