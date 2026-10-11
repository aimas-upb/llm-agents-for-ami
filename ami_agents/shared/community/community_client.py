"""
Community Signifier Client

HTTP client for querying the shared community signifier registry
(the Experience Engine signifier API running as a standalone service).

Used by InteractionSolver to discover cross-environment signifiers.
"""

from typing import Any, Optional

import aiohttp
from ..utils.logger import LoggerFactory

logger = LoggerFactory.get_logger(__name__)


class CommunitySignifierClient:
    """
    Client for querying the community signifier registry.

    The community API is the Experience Engine signifier API (shared/memory/src/api/)
    running as a shared service accessible by all environments.

    Endpoints used (the memory engine's API, `shared/memory/src/api/routes/signifiers.py`):
    - POST /matching/match - the record under an exact {kind, key}
    - POST /signifiers - publish a goal signifier
    - GET /signifiers - list all signifiers
    """

    def __init__(self, api_url: str, config: dict = None, timeout: float = None):
        """
        Args:
            api_url: Base URL of the community signifier API
            config: Configuration dictionary containing timeouts
            timeout: Request timeout in seconds (overrides config)
        """
        self.api_url = api_url.rstrip("/")
        self.config = config or {}

        # Get timeout from config or use provided/default value
        if timeout is None:
            timeout = self.config.get("timeouts", {}).get("community", {}).get("api", 10.0)

        self.timeout = aiohttp.ClientTimeout(total=timeout)

    async def match_signifiers(self, kind: str, key: dict) -> Optional[dict]:
        """The community's record under exactly this kind and key, or None
        (also on any failure: the community is a fallback, never required)."""
        try:
            async with aiohttp.ClientSession(timeout=self.timeout) as session:
                async with session.post(f"{self.api_url}/matching/match",
                                        json={"kind": kind, "key": key}) as resp:
                    if resp.status != 200:
                        logger.warning(f"Community match failed: HTTP {resp.status}")
                        return None
                    data = await resp.json()
                    match = data.get("match") if isinstance(data, dict) else None
                    return match if isinstance(match, dict) else None
        except Exception as e:
            logger.warning(f"Community match request failed: {e}")
            return None

    async def publish_signifier(self, signifier_data: dict) -> bool:
        """
        Publish a signifier to the community.

        Args:
            signifier_data: Signifier dict to publish

        Returns:
            True if published successfully
        """
        try:
            async with aiohttp.ClientSession(timeout=self.timeout) as session:
                async with session.post(
                    f"{self.api_url}/signifiers",
                    json=signifier_data,
                ) as resp:
                    if resp.status in (200, 201):
                        logger.info(
                            f"Published {signifier_data.get('kind', '?')} signifier to "
                            f"community: {signifier_data.get('intent_text', 'unknown')}"
                        )
                        return True
                    else:
                        logger.warning(
                            f"Failed to publish signifier: HTTP {resp.status}"
                        )
                        return False
        except aiohttp.ClientError as e:
            logger.warning(f"Community publish request failed: {e}")
            return False
        except Exception as e:
            logger.warning(f"Unexpected error in community publish: {e}")
            return False

    async def list_signifiers(self) -> list[dict]:
        """
        List all signifiers in the community.

        Returns:
            List of signifier dicts
        """
        try:
            async with aiohttp.ClientSession(timeout=self.timeout) as session:
                async with session.get(f"{self.api_url}/signifiers") as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        return data.get("signifiers", []) if isinstance(data, dict) else []
                    return []
        except Exception as e:
            logger.warning(f"Community list request failed: {e}")
            return []

    async def health_check(self) -> bool:
        """Check if the community API is reachable."""
        try:
            async with aiohttp.ClientSession(timeout=self.timeout) as session:
                async with session.get(f"{self.api_url}/") as resp:
                    return resp.status == 200
        except Exception:
            return False
