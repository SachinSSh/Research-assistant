import pytest
from datetime import datetime
from pydantic import ValidationError
from app.models.schemas import (
    BriefRequest, QueryDepth, FinalBrief, SourceSummary,
    SearchResult, ResearchPlan, Reference, APIResponse
)


class TestBriefRequest:
    def test_valid_request(self):
        """Test valid brief request creation."""
        request = BriefRequest(
            topic="Climate Change Impact",
            depth=QueryDepth.DEEP,
            follow_up=True,
            user_id="user123"
        )
        
        assert request.topic == "Climate Change Impact"
        assert request.depth == QueryDepth.DEEP
        assert request.follow_up is True
        assert request.user_id == "user123"

    def test_topic_validation(self):
        """Test topic validation."""
        with pytest.raises(ValidationError):
            BriefRequest(topic="AI", depth=QueryDepth.MEDIUM, follow_up=False, user_id="user123")
        
        with pytest.raises(ValidationError):
            BriefRequest(topic="", depth=QueryDepth.MEDIUM, follow_up=False, user_id="user123")
        
        with pytest.raises(ValidationError):
            BriefRequest(topic="   ", depth=QueryDepth.MEDIUM, follow_up=False, user_id="user123")

    def test_topic_sanitization(self):
        """Test topic whitespace stripping."""
        request = BriefRequest(
            topic="  Climate Change  ",
            depth=QueryDepth.MEDIUM,
            follow_up=False,
            user_id="user123"
        )
        
        assert request.topic == "Climate Change"


class TestFinalBrief:
    def test_valid_brief(self, sample_final_brief):
        """Test valid final brief creation."""
        assert sample_final_brief.topic == "Artificial Intelligence in Healthcare"
        assert len(sample_final_brief.key_findings) == 3
        assert sample_final_brief.confidence_score == 0.85
        assert sample_final_brief.processing_time == 45.3

    def test_confidence_score_validation(self):
        """Test confidence score range validation."""
        brief = FinalBrief(
            topic="Test Topic",
            summary="Test summary" * 10,
            key_findings=["Finding 1", "Finding 2", "Finding 3"],
            detailed_analysis="Detailed analysis" * 20,
            references=[],
            confidence_score=0.75,
            processing_time=30.0
        )
        assert brief.confidence_score == 0.75
        
        with pytest.raises(ValidationError):
            FinalBrief(
                topic="Test Topic",
                summary="Test summary" * 10,
                key_findings=["Finding 1", "Finding 2", "Finding 3"],
                detailed_analysis="Detailed analysis" * 20,
                references=[],
                confidence_score=1.5, 
                processing_time=30.0
            )

    def test_minimum_requirements(self):
        """Test minimum content requirements."""
        with pytest.raises(ValidationError):
            FinalBrief(
                topic="Test Topic",
                summary="Test summary" * 10,
                key_findings=["Finding 1", "Finding 2"],  
                detailed_analysis="Detailed analysis" * 20,
                references=[],
                confidence_score=0.75,
                processing_time=30.0
            )


        with pytest.raises(ValidationError):
            FinalBrief(
                topic="Test Topic",
                summary="Short",
                key_findings=["Finding 1", "Finding 2", "Finding 3"],
                detailed_analysis="Detailed analysis" * 20,
                references=[],
                confidence_score=0.75,
                processing_time=30.0
            )


class TestSourceSummary:
    def test_valid_source_summary(self):
        """Test valid source summary creation."""
        summary = SourceSummary(
            url="https://example.com/article",
            title="Test Article",
            content_snippet="This is a test article snippet.",
            key_points=["Point 1", "Point 2", "Point 3"],
            relevance_score=0.8,
            credibility_score=0.9,
            word_count=1500
        )
        
        assert summary.url == "https://example.com/article"
        assert len(summary.key_points) == 3
        assert summary.relevance_score == 0.8