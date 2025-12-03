"""Tests for the RD5 FastAPI application.

This module tests the REST API endpoints.
"""

import pytest
import sys
import os
from unittest.mock import AsyncMock, MagicMock, patch

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

# Ensure mock sandbox is used
os.environ["RD5_USE_DOCKER_SANDBOX"] = "false"

from fastapi.testclient import TestClient


@pytest.fixture
def client():
    """Create test client for the API."""
    from rd5.api.main import app
    return TestClient(app)


class TestRootEndpoint:
    """Test the root endpoint."""

    def test_root_returns_info(self, client):
        """Test root endpoint returns API info."""
        response = client.get("/")

        assert response.status_code == 200
        data = response.json()
        assert "name" in data
        assert "version" in data
        assert "docs" in data


class TestHealthEndpoint:
    """Test the health check endpoint."""

    def test_health_check(self, client):
        """Test health check endpoint."""
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "version" in data
        assert "services" in data


class TestStatusEndpoint:
    """Test the status endpoint."""

    def test_status(self, client):
        """Test status endpoint returns configuration."""
        response = client.get("/status")

        assert response.status_code == 200
        data = response.json()
        assert "app_name" in data
        assert "version" in data
        assert "settings" in data
        assert "openai_model" in data["settings"]


class TestValidateEndpoint:
    """Test the code validation endpoint."""

    def test_validate_safe_code(self, client):
        """Test validating safe code."""
        safe_code = '''
import json
import asyncio

async def main():
    return {"success": True}
'''
        response = client.post("/validate", params={"code": safe_code})

        assert response.status_code == 200
        data = response.json()
        assert data["is_valid"] is True
        assert len(data["errors"]) == 0

    def test_validate_unsafe_code(self, client):
        """Test validating unsafe code."""
        unsafe_code = '''
import os
os.system("rm -rf /")
'''
        response = client.post("/validate", params={"code": unsafe_code})

        assert response.status_code == 200
        data = response.json()
        assert data["is_valid"] is False
        assert len(data["errors"]) > 0


class TestPlansEndpoint:
    """Test the plan generation endpoint."""

    def test_generate_plan_with_mocked_llm(self, client):
        """Test plan generation with mocked LLM."""
        mock_llm = MagicMock()
        mock_llm.extract_intent = AsyncMock(return_value={
            "intent": "turn on lights",
            "action_verb": "turn on",
            "target_objects": ["lights"],
            "parameters": {},
            "location": "living room",
        })
        mock_llm.generate_plan_code = AsyncMock(return_value='''
import json
print(json.dumps({"success": True}))
''')
        mock_llm.summarize_execution = AsyncMock(return_value="Lights turned on")

        mock_weaviate = MagicMock()
        mock_weaviate.connect = MagicMock()
        mock_weaviate.close = MagicMock()
        mock_weaviate.ensure_schema = MagicMock()
        mock_weaviate.store_plan = MagicMock(return_value="plan-test-123")
        mock_weaviate.search_similar_plans = MagicMock(return_value=[])

        with patch("rd5.workflow.nodes.intent_extraction.get_llm_client", return_value=mock_llm), \
             patch("rd5.workflow.nodes.code_generation.get_llm_client", return_value=mock_llm), \
             patch("rd5.workflow.nodes.code_generation.WeaviateClient", return_value=mock_weaviate), \
             patch("rd5.workflow.nodes.result_feedback.get_llm_client", return_value=mock_llm), \
             patch("rd5.workflow.nodes.plan_storage.WeaviateClient", return_value=mock_weaviate):

            response = client.post(
                "/plans",
                json={
                    "user_request": "Turn on the living room lights",
                    "request_id": "test-api-001",
                },
            )

            assert response.status_code == 201
            data = response.json()
            assert data["request_id"] == "test-api-001"
            assert data["status"] == "completed"
            assert data["extracted_intent"] is not None
            assert data["extracted_intent"]["intent"] == "turn on lights"

    def test_generate_plan_missing_request(self, client):
        """Test plan generation with missing request."""
        response = client.post("/plans", json={})

        assert response.status_code == 422  # Validation error

    def test_generate_plan_empty_request(self, client):
        """Test plan generation with empty request."""
        response = client.post(
            "/plans",
            json={"user_request": ""},
        )

        assert response.status_code == 422  # Validation error


class TestPlanSearchEndpoint:
    """Test the plan search endpoint."""

    def test_search_plans_no_weaviate(self, client):
        """Test plan search when Weaviate is not available."""
        response = client.get("/plans", params={"query": "turn on lights"})

        # Should return 500 when Weaviate is not available
        assert response.status_code == 500

    def test_search_plans_with_mocked_weaviate(self, client):
        """Test plan search with mocked Weaviate."""
        mock_weaviate = MagicMock()
        mock_weaviate.connect = MagicMock()
        mock_weaviate.close = MagicMock()
        mock_weaviate.search_similar_plans = MagicMock(return_value=[
            {
                "plan_id": "plan-1",
                "intent_text": "turn on lights",
                "certainty": 0.95,
            },
        ])

        with patch("rd5.storage.weaviate_client.WeaviateClient", return_value=mock_weaviate):
            response = client.get("/plans", params={"query": "turn on lights"})

            assert response.status_code == 200
            data = response.json()
            assert data["count"] == 1
            assert data["plans"][0]["plan_id"] == "plan-1"


class TestGetPlanEndpoint:
    """Test the get plan by ID endpoint."""

    def test_get_plan_not_found(self, client):
        """Test getting a non-existent plan."""
        mock_weaviate = MagicMock()
        mock_weaviate.connect = MagicMock()
        mock_weaviate.close = MagicMock()
        mock_weaviate.get_plan_by_id = MagicMock(return_value=None)

        with patch("rd5.storage.weaviate_client.WeaviateClient", return_value=mock_weaviate):
            response = client.get("/plans/nonexistent-id")

            assert response.status_code == 404

    def test_get_plan_found(self, client):
        """Test getting an existing plan."""
        mock_weaviate = MagicMock()
        mock_weaviate.connect = MagicMock()
        mock_weaviate.close = MagicMock()
        mock_weaviate.get_plan_by_id = MagicMock(return_value={
            "plan_id": "plan-123",
            "intent_text": "turn on lights",
            "plan_code": "print('hello')",
        })

        with patch("rd5.storage.weaviate_client.WeaviateClient", return_value=mock_weaviate):
            response = client.get("/plans/plan-123")

            assert response.status_code == 200
            data = response.json()
            assert data["plan_id"] == "plan-123"


class TestAPIModels:
    """Test API model validation."""

    def test_plan_request_validation(self):
        """Test PlanRequest model validation."""
        from rd5.api.main import PlanRequest

        # Valid request
        request = PlanRequest(user_request="Turn on lights")
        assert request.user_request == "Turn on lights"
        assert request.execute is True

        # With optional fields
        request = PlanRequest(
            user_request="Set temperature",
            request_id="test-123",
            execute=False,
        )
        assert request.request_id == "test-123"
        assert request.execute is False

    def test_plan_request_too_long(self):
        """Test PlanRequest rejects too-long requests."""
        from rd5.api.main import PlanRequest
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            PlanRequest(user_request="x" * 1001)

    def test_validation_result_model(self):
        """Test ValidationResult model."""
        from rd5.api.main import ValidationResult

        result = ValidationResult(
            is_valid=True,
            errors=[],
            warnings=["minor issue"],
            imports_found=["json", "asyncio"],
        )

        assert result.is_valid is True
        assert len(result.warnings) == 1
        assert "json" in result.imports_found


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
