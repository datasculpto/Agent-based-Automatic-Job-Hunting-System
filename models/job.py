"""Job data model for structured job information."""

from typing import Optional
from pydantic import BaseModel, Field


class Job(BaseModel):
    """Structured job posting information."""

    title: str = Field(description="职位名称")
    company: str = Field(description="公司名称")
    location: str = Field(default="未知", description="工作地点")
    salary: str = Field(default="未公开", description="薪资范围")
    tech_tags: list[str] = Field(default_factory=list, description="技术关键词，如 LLM / CV / NLP / 推荐系统")
    requirements: str = Field(default="", description="岗位核心技能摘要")
    source: str = Field(description="招聘网站来源")
    job_url: str = Field(description="岗位链接")

    def dedup_key(self) -> str:
        """Generate a deduplication key based on company + title."""
        return f"{self.company.strip().lower()}|{self.title.strip().lower()}"
