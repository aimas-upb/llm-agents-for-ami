"""
Unit tests for CommunitySignifierClient with mocked HTTP.
"""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from aiohttp import ClientError

from ami_agents.shared.community.community_client import CommunitySignifierClient


@pytest.fixture
def client():
    return CommunitySignifierClient(api_url="http://localhost:9000")


class TestMatchSignifiers:
    """Tests for CommunitySignifierClient.match_signifiers()."""

    @pytest.mark.asyncio
    async def test_match_success(self, client):
        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.json = AsyncMock(return_value={
            "matches": [
                {"signifier_id": "sig-001", "affordance_uri": "http://test/turn_on", "intent_similarity": 0.9}
            ],
            "final_matches": ["sig-001"],
        })
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=False)

        mock_session = AsyncMock()
        mock_session.post = MagicMock(return_value=mock_resp)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            result = await client.match_signifiers("increase light")

        assert "matches" in result
        assert len(result["matches"]) == 1

    @pytest.mark.asyncio
    async def test_match_empty(self, client):
        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.json = AsyncMock(return_value={"matches": [], "final_matches": []})
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=False)

        mock_session = AsyncMock()
        mock_session.post = MagicMock(return_value=mock_resp)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            result = await client.match_signifiers("unknown intent")

        assert result.get("matches", []) == []

    @pytest.mark.asyncio
    async def test_match_server_error(self, client):
        mock_resp = AsyncMock()
        mock_resp.status = 500
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=False)

        mock_session = AsyncMock()
        mock_session.post = MagicMock(return_value=mock_resp)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            result = await client.match_signifiers("test")

        assert result == {}

    @pytest.mark.asyncio
    async def test_match_connection_error(self, client):
        mock_session = AsyncMock()
        mock_session.post = MagicMock(side_effect=ClientError("Connection refused"))
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            result = await client.match_signifiers("test")

        assert result == {}


class TestPublishSignifier:
    """Tests for CommunitySignifierClient.publish_signifier()."""

    @pytest.mark.asyncio
    async def test_publish_success(self, client):
        mock_resp = AsyncMock()
        mock_resp.status = 201
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=False)

        mock_session = AsyncMock()
        mock_session.post = MagicMock(return_value=mock_resp)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            result = await client.publish_signifier({"intent": "turn on light"})

        assert result is True

    @pytest.mark.asyncio
    async def test_publish_failure(self, client):
        mock_resp = AsyncMock()
        mock_resp.status = 400
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=False)

        mock_session = AsyncMock()
        mock_session.post = MagicMock(return_value=mock_resp)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            result = await client.publish_signifier({"intent": "test"})

        assert result is False

    @pytest.mark.asyncio
    async def test_publish_connection_error(self, client):
        mock_session = AsyncMock()
        mock_session.post = MagicMock(side_effect=ClientError("fail"))
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            result = await client.publish_signifier({"intent": "test"})

        assert result is False
