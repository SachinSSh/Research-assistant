from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, validator
from enum import Enum


class QueryDepth(int, Enum):
    SHALLOW = 1
    MEDIUM = 2
    DEEP = 3


class SearchResult(BaseModel):
    title: str
    url: str
    snippet: str
    published_date: Optional[datetime] = None
    relevance_score: float = Field(ge=0.0, le=1.0)


class ResearchPlan(BaseModel):
    query: str
    search_queries: List[str] = Field(min_items=1, max_items=10)
    expected_sources: int = Field(ge=1, le=20)
    focus_areas: List[str]
    estimated_duration: int = Field(description="Estimated duration in seconds")


class SourceSummary(BaseModel):
    url: str
    title: str
    content_snippet: str = Field(max_length=500)
    key_points: List[str] = Field(min_items=1, max_items=10)
    relevance_score: float = Field(ge=0.0, le=1.0)
    credibility_score: float = Field(ge=0.0, le=1.0)
    word_count: int = Field(ge=0)
    processed_at: datetime = Field(default_factory=datetime.utcnow)


class Reference(BaseModel):
    title: str
    url: str
    author: Optional[str] = None
    published_date: Optional[datetime] = None
    excerpt: str = Field(max_length=200)


class FinalBrief(BaseModel):
    topic: str
    summary: str = Field(min_length=100)
    key_findings: List[str] = Field(min_items=3, max_items=15)
    detailed_analysis: str = Field(min_length=500)
    references: List[Reference] = Field(min_items=1)
    confidence_score: float = Field(ge=0.0, le=1.0)
    generated_at: datetime = Field(default_factory=datetime.utcnow)
    processing_time: float = Field(description="Processing time in seconds")
    token_usage: Dict[str, int] = Field(default_factory=dict)


class BriefRequest(BaseModel):
    topic: str = Field(min_length=10, max_length=500)
    depth: QueryDepth = Field(default=QueryDepth.MEDIUM)
    follow_up: bool = Field(default=False)
    user_id: str = Field(min_length=1, max_length=100)
    
    @validator('topic')
    def validate_topic(cls, v):
        if not v.strip():
            raise ValueError('Topic cannot be empty or whitespace only')
        return v.strip()


class ContextSummary(BaseModel):
    user_id: str
    previous_topics: List[str]
    key_insights: List[str]
    recurring_themes: List[str]
    last_interaction: datetime
    total_interactions: int


class UserHistory(BaseModel):
    user_id: str
    briefs: List[FinalBrief]
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class APIResponse(BaseModel):
    success: bool
    data: Optional[Any] = None
    error: Optional[str] = None
    trace_id: Optional[str] = None
    processing_time: Optional[float] = None