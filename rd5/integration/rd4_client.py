"""RD4 Signifier API client for integration.

This module provides a client to interact with the RD4 Signifier API
for querying and creating signifiers based on plan execution.
"""

import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import httpx

from rd5.config.settings import get_settings

logger = logging.getLogger(__name__)


@dataclass
class SignifierMatch:
    """Result of signifier matching.

    Attributes:
        signifier_id: Matched signifier ID.
        intent_similarity: Similarity score (0.0 to 1.0).
        shacl_conforms: Whether SHACL validation passed.
        shacl_violations: List of violation messages.
    """

    signifier_id: str
    intent_similarity: float
    shacl_conforms: bool
    shacl_violations: List[str]


@dataclass
class SignifierInfo:
    """Basic signifier information.

    Attributes:
        signifier_id: Unique identifier.
        version: Version number.
        status: Active or deprecated.
        intent: Natural language intent.
        affordance_uri: URI of the affordance.
    """

    signifier_id: str
    version: int
    status: str
    intent: str
    affordance_uri: str


class RD4Client:
    """Client for RD4 Signifier API.

    Provides methods to query, create, and manage signifiers
    through the RD4 API endpoints.
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        timeout: Optional[float] = None,
    ):
        """Initialize RD4 client.

        Args:
            base_url: API base URL. Defaults to settings.
            timeout: Request timeout in seconds. Defaults to settings.
        """
        settings = get_settings()
        self.base_url = (base_url or settings.rd4_api_url).rstrip("/")
        self.timeout = timeout or settings.rd4_api_timeout
        self._client: Optional[httpx.AsyncClient] = None

    async def __aenter__(self) -> "RD4Client":
        """Async context manager entry."""
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=self.timeout,
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        if self._client:
            await self._client.aclose()
            self._client = None

    @property
    def client(self) -> httpx.AsyncClient:
        """Get HTTP client instance.

        Returns:
            Configured httpx client.

        Raises:
            RuntimeError: If not in async context.
        """
        if not self._client:
            raise RuntimeError(
                "RD4Client must be used as async context manager"
            )
        return self._client

    async def list_signifiers(self) -> List[SignifierInfo]:
        """List all signifiers.

        Returns:
            List of signifier information.

        Raises:
            httpx.HTTPStatusError: On API error.
        """
        response = await self.client.get("/signifiers")
        response.raise_for_status()

        data = response.json()
        signifiers = []

        for s in data.get("signifiers", []):
            signifiers.append(
                SignifierInfo(
                    signifier_id=s["signifier_id"],
                    version=s["version"],
                    status=s["status"],
                    intent=s["intent"],
                    affordance_uri=s["affordance_uri"],
                )
            )

        logger.info(f"Retrieved {len(signifiers)} signifiers from RD4")
        return signifiers

    async def match_intent(
        self,
        intent: str,
        context: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> List[SignifierMatch]:
        """Match signifiers based on intent and context.

        Args:
            intent: Natural language intent query.
            context: Optional context as nested dict
                    {artifact_uri: {property_uri: value}}.

        Returns:
            List of matching signifiers sorted by similarity.

        Raises:
            httpx.HTTPStatusError: On API error.
        """
        params = {"intent": intent}
        if context:
            params["context"] = json.dumps(context)

        response = await self.client.get("/signifiers/match", params=params)
        response.raise_for_status()

        data = response.json()
        matches = []

        for m in data.get("matches", []):
            matches.append(
                SignifierMatch(
                    signifier_id=m["signifier_id"],
                    intent_similarity=m["intent_similarity"],
                    shacl_conforms=m["shacl_conforms"],
                    shacl_violations=m.get("shacl_violations", []),
                )
            )

        # Filter to only those that passed SHACL if specified
        final_ids = set(data.get("final_matches", []))
        logger.info(
            f"Matched {len(matches)} signifiers for intent, "
            f"{len(final_ids)} passed SHACL validation"
        )

        return matches

    async def create_signifier(self, rdf_data: str) -> str:
        """Create a new signifier from RDF data.

        Args:
            rdf_data: RDF data in Turtle format.

        Returns:
            Created signifier ID.

        Raises:
            httpx.HTTPStatusError: On API error.
        """
        response = await self.client.post(
            "/signifiers",
            json={"rdf_data": rdf_data},
        )
        response.raise_for_status()

        data = response.json()
        signifier_id = data["signifier_id"]

        logger.info(f"Created signifier via RD4: {signifier_id}")
        return signifier_id

    async def delete_all_signifiers(self) -> int:
        """Delete all signifiers.

        Returns:
            Number of signifiers deleted.

        Raises:
            httpx.HTTPStatusError: On API error.
        """
        response = await self.client.delete("/signifiers")
        response.raise_for_status()

        data = response.json()
        deleted_count = data.get("deleted_count", 0)

        logger.info(f"Deleted {deleted_count} signifiers from RD4")
        return deleted_count

    async def health_check(self) -> bool:
        """Check if RD4 API is healthy.

        Returns:
            True if API is responsive.
        """
        try:
            response = await self.client.get("/health")
            return response.status_code == 200
        except Exception as e:
            logger.warning(f"RD4 health check failed: {e}")
            return False


def generate_signifier_rdf(
    signifier_id: str,
    intent_text: str,
    affordance_uri: str,
    created_by: str = "rd5-orchestrator",
    context_conditions: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """Generate RDF (Turtle) for a new signifier.

    Args:
        signifier_id: Unique identifier for the signifier.
        intent_text: Natural language intent description.
        affordance_uri: URI of the affordance to signify.
        created_by: Creator identifier.
        context_conditions: Optional list of context conditions.

    Returns:
        RDF data in Turtle format.
    """
    # Build base RDF
    rdf = f"""@prefix signifier: <http://example.org/signifier/> .
@prefix intent: <http://example.org/intent/> .
@prefix affordance: <http://example.org/affordance/> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .
@prefix prov: <http://www.w3.org/ns/prov#> .

signifier:{signifier_id} a signifier:Signifier ;
    signifier:hasIntent [
        intent:nlText "{intent_text}" ;
    ] ;
    signifier:signifiesAffordance <{affordance_uri}> ;
    prov:wasAttributedTo "{created_by}" .
"""

    # Add context conditions if provided
    if context_conditions:
        conditions_rdf = []
        for i, cond in enumerate(context_conditions):
            artifact = cond.get("artifact", "")
            prop = cond.get("property", "")
            operator = cond.get("operator", "equals")
            value = cond.get("value", "")

            conditions_rdf.append(f"""
    signifier:hasContextCondition [
        signifier:artifact <{artifact}> ;
        signifier:property <{prop}> ;
        signifier:operator "{operator}" ;
        signifier:value "{value}"
    ]""")

        if conditions_rdf:
            # Insert before the final period
            rdf = rdf.rstrip(". \n") + " ;" + " ;".join(conditions_rdf) + " .\n"

    return rdf
