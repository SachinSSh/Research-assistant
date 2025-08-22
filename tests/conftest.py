import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock
from app.services.llm_service import LLMService
from app.services.search_service import SearchService
from app.services.context_service import ContextService
from app.services.storage_service import StorageService
from app.models.schemas import (
    BriefRequest, QueryDepth, FinalBrief, SourceSummary,
    SearchResult, ResearchPlan, ContextSummary
)


@pytest.fixture
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def sample_brief_request():
    """Sample brief request for testing."""
    return BriefRequest(
        topic="Artificial Intelligence in Healthcare",
        depth=QueryDepth.MEDIUM,
        follow_up=False,
        user_id="test_user_123"
    )


@pytest.fixture
def sample_final_brief():
    """Sample final brief for testing."""
    return FinalBrief(
        topic="Artificial Intelligence in Healthcare",
        summary="AI is transforming healthcare through various applications.",
        key_findings=[
            "AI improves diagnostic accuracy",
            "Machine learning enables personalized treatment",
            "Natural language processing assists in clinical documentation"
        ],
        detailed_analysis="Detailed analysis of AI applications in healthcare...",
        references=[],
        confidence_score=0.85,
        processing_time=45.3,
        token_usage={"total_tokens": 1500, "prompt_tokens": 800, "completion_tokens": 700}
    )


@pytest.fixture
def mock_llm_service():
    """Mock LLM service."""
    service = AsyncMock(spec=LLMService)
    
    service.generate_structured = AsyncMock(return_value=ResearchPlan(
        query="AI in Healthcare",
        search_queries=["AI healthcare applications", "machine learning medicine"],
        expected_sources=10,
        focus_areas=["diagnostics", "treatment", "automation"],
        estimated_duration=300
    ))
    
    service.generate_text = AsyncMock(return_value="Generated text response")
    
    return service


@pytest.fixture
def mock_search_service():
    """Mock search service."""
    service = AsyncMock(spec=SearchService)
    
    service.search_web = AsyncMock(return_value=[
        SearchResult(
            title="AI in Healthcare: Current Applications",
            url="https://example.com/ai-healthcare",
            snippet="AI is revolutionizing healthcare...",
            relevance_score=0.9
        )
    ])
    
    service.fetch_content = AsyncMock(return_value="Detailed article content...")
    service.batch_fetch_content = AsyncMock(return_value={
        "https://example.com/ai-healthcare": "Detailed article content..."
    })
    
    return service


@pytest.fixture
def mock_storage_service():
    """Mock storage service."""
    service = AsyncMock(spec=StorageService)
    service.initialize = AsyncMock()
    service.get_user_history = AsyncMock(return_value=None)
    service.save_user_brief = AsyncMock()
    service.get_user_stats = AsyncMock(return_value={
        "total_briefs": 5,
        "avg_processing_time": 42.5,
        "total_tokens": 7500
    })
    return service


@pytest.fixture
def mock_context_service():
    """Mock context service."""
    service = AsyncMock(spec=ContextService)
    service.get_context_summary = AsyncMock(return_value=None)
    service.incorporate_context_into_planning = AsyncMock(return_value="Context guidance")
    service.save_brief = AsyncMock()
    return service
