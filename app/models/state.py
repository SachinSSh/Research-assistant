from typing import List, Optional, Dict, Any, TypedDict
from langchain_core.messages import BaseMessage
from .schemas import (
    ResearchPlan, SearchResult, SourceSummary, 
    FinalBrief, ContextSummary, BriefRequest
)


class GraphState(TypedDict):
    # Input
    request: BriefRequest
    
    # Context
    context_summary: Optional[ContextSummary]
    
    # Planning
    research_plan: Optional[ResearchPlan]
    
    # Search and Retrieval
    search_results: List[SearchResult]
    fetched_content: Dict[str, str]
    
    # Processing
    source_summaries: List[SourceSummary]
    
    # Output
    final_brief: Optional[FinalBrief]
    
    # Metadata
    messages: List[BaseMessage]
    errors: List[str]
    retry_count: int
    processing_start: float
    token_usage: Dict[str, int]
    trace_id: Optional[str]


class NodeOutput(TypedDict):
    success: bool
    data: Optional[Any]
    error: Optional[str]
    tokens_used: int
    processing_time: float