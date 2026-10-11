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


def _session(status=200, body=None, error=None):
    """An aiohttp session whose POST answers `status` / `body`, or raises."""
    mock_resp = AsyncMock()
    mock_resp.status = status
    mock_resp.json = AsyncMock(return_value=body)
    mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
    mock_resp.__aexit__ = AsyncMock(return_value=False)
    mock_session = AsyncMock()
    mock_session.post = (MagicMock(side_effect=error) if error
                         else MagicMock(return_value=mock_resp))
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)
    return mock_session


KEY = {"goal_specificity": "explicit", "artifact_name": "ac"}


class TestMatchSignifiers:
    """match_signifiers: the record under an exact {kind, key}, or None."""

    @pytest.mark.asyncio
    async def test_a_match_is_returned(self, client):
        record = {"signifier_id": "sig-001", "kind": "explicit", "tree": {"type": "action"}}
        session = _session(body={"match": record})
        with patch("aiohttp.ClientSession", return_value=session):
            result = await client.match_signifiers("explicit", KEY)
        assert result == record
        _, kwargs = session.post.call_args
        assert kwargs["json"] == {"kind": "explicit", "key": KEY}

    @pytest.mark.asyncio
    async def test_no_match_is_none(self, client):
        with patch("aiohttp.ClientSession", return_value=_session(body={"match": None})):
            assert await client.match_signifiers("explicit", KEY) is None

    @pytest.mark.asyncio
    async def test_a_server_error_is_none(self, client):
        with patch("aiohttp.ClientSession", return_value=_session(status=500)):
            assert await client.match_signifiers("explicit", KEY) is None

    @pytest.mark.asyncio
    async def test_a_connection_error_is_none(self, client):
        with patch("aiohttp.ClientSession",
                   return_value=_session(error=ClientError("Connection refused"))):
            assert await client.match_signifiers("explicit", KEY) is None


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
