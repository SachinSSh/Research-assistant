import time
import uuid
from typing import Dict, Any, Literal
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_core.messages import HumanMessage

from ..models.state import GraphState
from ..models.schemas import BriefRequest
from ..services.llm_service import LLMService
from ..services.search_service import SearchService
from ..services.context_service import ContextService
from ..services.storage_service import StorageService
from .nodes import ResearchNodes
from ..utils.monitoring import setup_tracing


class ResearchWorkflow:
    def __init__(self):
        self.llm_service = LLMService()
        self.storage_service = StorageService()
        self.context_service = ContextService(self.llm_service, self.storage_service)
        
        self.nodes = ResearchNodes(
            self.llm_service,
            SearchService(),
            self.context_service
        )
        
        self.checkpointer = SqliteSaver.from_conn_string(":memory:")
        
        self.workflow = self._build_workflow()

    def _build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow."""
        workflow = StateGraph(GraphState)
        
        workflow.add_node("context_summarization", self.nodes.context_summarization_node)
        workflow.add_node("planning", self.nodes.planning_node)
        workflow.add_node("search", self.nodes.search_node)
        workflow.add_node("content_fetching", self.nodes.content_fetching_node)
        workflow.add_node("source_summarization", self.nodes.source_summarization_node)
        workflow.add_node("synthesis", self.nodes.synthesis_node)
        workflow.add_node("post_processing", self.nodes.post_processing_node)
        
        workflow.set_entry_point("context_summarization")
        
        workflow.add_edge("context_summarization", "planning")
        workflow.add_edge("planning", "search")
        workflow.add_conditional_edges(
            "search",
            self._should_retry_search,
            {
                "continue": "content_fetching",
                "retry": "search",
                "skip": "synthesis"
            }
        )
        workflow.add_edge("content_fetching", "source_summarization")
        workflow.add_conditional_edges(
            "source_summarization",
            self._should_retry_summarization,
            {
                "continue": "synthesis",
                "retry": "source_summarization",
                "skip": "synthesis"
            }
        )
        workflow.add_edge("synthesis", "post_processing")
        workflow.add_edge("post_processing", END)
        
        return workflow.compile(checkpointer=self.checkpointer)

    def _should_retry_search(self, state: GraphState) -> Literal["continue", "retry", "skip"]:
        """Determine if search should be retried."""
        if len(state["search_results"]) == 0:
            if state["retry_count"] < 2:
                state["retry_count"] += 1
                return "retry"
            else:
                return "skip"
        return "continue"

    def _should_retry_summarization(self, state: GraphState) -> Literal["continue", "retry", "skip"]:
        """Determine if summarization should be retried."""
        if len(state["source_summaries"]) == 0:
            if state["retry_count"] < 2:
                state["retry_count"] += 1
                return "retry"
            else:
                return "skip"
        return "continue"

    async def execute(self, request: BriefRequest) -> Dict[str, Any]:
        """Execute the research workflow."""
        trace_id = str(uuid.uuid4())
        setup_tracing(trace_id)
        
        initial_state: GraphState = {
            "request": request,
            "context_summary": None,
            "research_plan": None,
            "search_results": [],
            "fetched_content": {},
            "source_summaries": [],
            "final_brief": None,
            "messages": [HumanMessage(content=f"Starting research for: {request.topic}")],
            "errors": [],
            "retry_count": 0,
            "processing_start": time.time(),
            "token_usage": {},
            "trace_id": trace_id
        }
        
        try:
            config = {"configurable": {"thread_id": f"user_{request.user_id}_{int(time.time())}"}}
            final_state = await self.workflow.ainvoke(initial_state, config)
            
            return {
                "success": True,
                "brief": final_state["final_brief"],
                "trace_id": trace_id,
                "processing_time": final_state["final_brief"].processing_time if final_state["final_brief"] else 0,
                "errors": final_state["errors"]
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "trace_id": trace_id,
                "processing_time": time.time() - initial_state["processing_start"]
            }

    async def resume_execution(self, thread_id: str) -> Dict[str, Any]:
        """Resume a checkpointed execution."""
        try:
            config = {"configurable": {"thread_id": thread_id}}
            final_state = await self.workflow.ainvoke(None, config)
            
            return {
                "success": True,
                "brief": final_state["final_brief"],
                "resumed": True
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Resume failed: {str(e)}"
            }