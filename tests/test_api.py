import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, patch
from app.main import app
from app.models.schemas import QueryDepth


@pytest.fixture
def client():
    """Test client."""
    return TestClient(app)


class TestBriefEndpoint:
    def test_valid_brief_request(self, client):
        """Test valid brief request."""
        with patch('app.main.workflow_instance') as mock_workflow:
            mock_workflow.execute = AsyncMock(return_value={
                "success": True,
                "brief": {
                    "topic": "AI in Healthcare",
                    "summary": "AI is transforming healthcare through various applications.",
                    "key_findings": [
                        "AI improves diagnostic accuracy",
                        "Machine learning enables personalized treatment",
                        "Natural language processing assists in clinical documentation"
                    ],
                    "detailed_analysis": "Detailed analysis of AI applications in healthcare...",
                    "references": [],
                    "confidence_score": 0.85,
                    "generated_at": "2024-01-01T12:00:00",
                    "processing_time": 45.3,
                    "token_usage": {"total_tokens": 1500}
                },
                "trace_id": "test-trace-123",
                "processing_time": 45.3
            })
            
            response = client.post("/brief", json={
                "topic": "AI in Healthcare",
                "depth": 2,
                "follow_up": False,
                "user_id": "test_user"
            })
            
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert data["data"]["topic"] == "AI in Healthcare"
            assert "trace_id" in data

    def test_invalid_brief_request(self, client):
        """Test invalid brief request."""
        response = client.post("/brief", json={
            "topic": "AI" 
        })
        
        assert response.status_code == 400
        assert "Invalid request data" in response.json()["detail"]

    def test_brief_request_validation(self, client):
        """Test request validation."""
        response = client.post("/brief", json={
            "topic": "AI", 
            "depth": 2,
            "follow_up": False,
            "user_id": "test_user"
        })
        
        assert response.status_code == 400


class TestHistoryEndpoint:
    def test_get_user_history(self, client):
        """Test getting user history."""
        with patch('app.main.storage_service') as mock_storage:
            mock_storage.get_user_history = AsyncMock(return_value={
                "user_id": "test_user",
                "briefs": [],
                "created_at": "2024-01-01T00:00:00",
                "updated_at": "2024-01-01T12:00:00"
            })
            
            response = client.get("/history/test_user")
            
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert data["data"]["user_id"] == "test_user"

    def test_get_nonexistent_user_history(self, client):
        """Test getting history for non-existent user."""
        with patch('app.main.storage_service') as mock_storage:
            mock_storage.get_user_history = AsyncMock(return_value=None)
            
            response = client.get("/history/nonexistent_user")
            
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert data["data"] is None


class TestHealthEndpoint:
    def test_health_check(self, client):
        """Test health check endpoint."""
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
