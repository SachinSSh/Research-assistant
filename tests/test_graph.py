from httpx import patch
import pytest
from unittest.mock import AsyncMock, MagicMock
from app.graph.nodes import ResearchNodes
from app.models.state import GraphState
from app.models.schemas import BriefRequest, QueryDepth, SearchResult


@pytest.mark.asyncio
class TestResearchNodes:
    async def test_context_summarization_node_no_followup(
        self, mock_llm_service, mock_search_service, mock_context_service
    ):
        """Test context summarization when not a follow-up."""
        nodes = ResearchNodes(mock_llm_service, mock_search_service, mock_context_service)
        
        state = GraphState(
            request=BriefRequest(
                topic="AI in Healthcare",
                depth=QueryDepth.MEDIUM,
                follow_up=False,
                user_id="test_user"
            ),
            context_summary=None,
            research_plan=None,
            search_results=[],
            fetched_content={},
            source_summaries=[],
            final_brief=None,
            messages=[],
            errors=[],
            retry_count=0,
            processing_start=0.0,
            token_usage={},
            trace_id=None
        )
        
        result = await nodes.context_summarization_node(state)
        
        assert result["context_summary"] is None
        mock_context_service.get_context_summary.assert_not_called()

    async def test_context_summarization_node_with_followup(
        self, mock_llm_service, mock_search_service, mock_context_service
    ):
        """Test context summarization for follow-up query."""
        nodes = ResearchNodes(mock_llm_service, mock_search_service, mock_context_service)
        
        state = GraphState(
            request=BriefRequest(
                topic="AI in Healthcare",
                depth=QueryDepth.MEDIUM,
                follow_up=True,
                user_id="test_user"
            ),
            context_summary=None,
            research_plan=None,
            search_results=[],
            fetched_content={},
            source_summaries=[],
            final_brief=None,
            messages=[],
            errors=[],
            retry_count=0,
            processing_start=0.0,
            token_usage={},
            trace_id=None
        )
        
        result = await nodes.context_summarization_node(state)
        
        mock_context_service.get_context_summary.assert_called_once_with("test_user")

    async def test_planning_node(
        self, mock_llm_service, mock_search_service, mock_context_service
    ):
        """Test research planning node."""
        nodes = ResearchNodes(mock_llm_service, mock_search_service, mock_context_service)
        
        state = GraphState(
            request=BriefRequest(
                topic="AI in Healthcare",
                depth=QueryDepth.MEDIUM,
                follow_up=False,
                user_id="test_user"
            ),
            context_summary=None,
            research_plan=None,
            search_results=[],
            fetched_content={},
            source_summaries=[],
            final_brief=None,
            messages=[],
            errors=[],
            retry_count=0,
            processing_start=0.0,
            token_usage={},
            trace_id=None
        )
        
        result = await nodes.planning_node(state)
        
        assert result["research_plan"] is not None
        assert result["research_plan"].query == "AI in Healthcare"
        mock_llm_service.generate_structured.assert_called_once()

    async def test_search_node(
        self, mock_llm_service, mock_search_service, mock_context_service, sample_final_brief
    ):
        """Test search node."""
        from app.models.schemas import ResearchPlan
        
        nodes = ResearchNodes(mock_llm_service, mock_search_service, mock_context_service)
        
        research_plan = ResearchPlan(
            query="AI in Healthcare",
            search_queries=["AI healthcare applications", "machine learning medicine"],
            expected_sources=5,
            focus_areas=["diagnostics", "treatment"],
            estimated_duration=300
        )
        
        state = GraphState(
            request=BriefRequest(
                topic="AI in Healthcare",
                depth=QueryDepth.MEDIUM,
                follow_up=False,
                user_id="test_user"
            ),
            context_summary=None,
            research_plan=research_plan,
            search_results=[],
            fetched_content={},
            source_summaries=[],
            final_brief=None,
            messages=[],
            errors=[],
            retry_count=0,
            processing_start=0.0,
            token_usage={},
            trace_id=None
        )
        
        with patch('app.graph.nodes.SearchService') as MockSearchService:
            mock_search_instance = AsyncMock()
            MockSearchService.return_value.__aenter__.return_value = mock_search_instance
            mock_search_instance.search_web = AsyncMock(return_value=[
                SearchResult(
                    title="AI in Healthcare Applications",
                    url="https://example.com/ai-healthcare",
                    snippet="AI applications in healthcare...",
                    relevance_score=0.9
                )
            ])
            
            result = await nodes.search_node(state)
            
            assert len(result["search_results"]) > 0
            assert result["search_results"][0].title == "AI in Healthcare Applications"
