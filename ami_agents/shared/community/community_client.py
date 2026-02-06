"""
Community Signifier Client

HTTP client for querying the shared community signifier registry
(the RD4 signifier API running as a standalone service).

Used by InteractionSolver to discover cross-environment signifiers.
"""

import logging
from typing import Any, Optional

import aiohttp

logger = logging.getLogger(__name__)


class CommunitySignifierClient:
    """
    Client for querying the community signifier registry.

    The community API is the RD4 signifier API (shared/memory/src/api/)
    running as a shared service accessible by all environments.

    Endpoints used:
    - POST /matching/match - match signifiers by intent
    - POST /signifiers - create/publish a signifier
    - GET /signifiers - list all signifiers
    """

    def __init__(self, api_url: str, timeout: float = 10.0):
        """
        Args:
            api_url: Base URL of the community signifier API
            timeout: Request timeout in seconds
        """
        self.api_url = api_url.rstrip("/")
        self.timeout = aiohttp.ClientTimeout(total=timeout)

    async def match_signifiers(
        self,
        intent: str,
        k: int = 5,
        min_similarity: float = 0.5,
    ) -> dict:
        """
        Query community for signifier matches.

        Args:
            intent: Natural language intent to match
            k: Maximum number of matches to return
            min_similarity: Minimum similarity threshold

        Returns:
            Dict with match results (matches, final_matches, etc.)
        """
        try:
            async with aiohttp.ClientSession(timeout=self.timeout) as session:
                async with session.post(
                    f"{self.api_url}/matching/match",
                    json={
                        "intent": intent,
                        "k": k,
                        "min_similarity": min_similarity,
                    },
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        logger.debug(
                            f"Community match for '{intent}': "
                            f"{len(data.get('matches', []))} matches"
                        )
                        return data
                    else:
                        logger.warning(
                            f"Community match failed: HTTP {resp.status}"
                        )
                        return {}
        except aiohttp.ClientError as e:
            logger.warning(f"Community match request failed: {e}")
            return {}
        except Exception as e:
            logger.warning(f"Unexpected error in community match: {e}")
            return {}

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
                            f"Published signifier to community: "
                            f"{signifier_data.get('intent', 'unknown')}"
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
                        return await resp.json()
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
