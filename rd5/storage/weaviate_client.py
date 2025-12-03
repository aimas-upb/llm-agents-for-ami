"""Weaviate client for RD5 plan storage and retrieval.

This module provides high-level operations for storing and searching
execution plans using Weaviate's vector database.
"""

import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import weaviate
from weaviate.classes.config import Configure, Property, DataType
from weaviate.classes.query import MetadataQuery
from weaviate.exceptions import WeaviateConnectionError

from rd5.config.settings import get_settings
from rd5.storage.weaviate_schema import EXECUTION_PLAN_SCHEMA

logger = logging.getLogger(__name__)


class WeaviateClient:
    """Client for Weaviate vector database operations.

    Handles connection management, schema initialization, and
    CRUD operations for execution plans.
    """

    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        grpc_port: Optional[int] = None,
    ):
        """Initialize Weaviate client.

        Args:
            host: Weaviate host. Defaults to settings.
            port: Weaviate HTTP port. Defaults to settings.
            grpc_port: Weaviate gRPC port. Defaults to settings.
        """
        settings = get_settings()
        self.host = host or settings.weaviate_host
        self.port = port or settings.weaviate_port
        self.grpc_port = grpc_port or settings.weaviate_grpc_port
        self._client: Optional[weaviate.WeaviateClient] = None

    def connect(self) -> None:
        """Establish connection to Weaviate.

        Raises:
            WeaviateConnectionError: If connection fails.
        """
        try:
            self._client = weaviate.connect_to_local(
                host=self.host,
                port=self.port,
                grpc_port=self.grpc_port,
            )
            logger.info(f"Connected to Weaviate at {self.host}:{self.port}")
        except Exception as e:
            logger.error(f"Failed to connect to Weaviate: {e}")
            raise WeaviateConnectionError(f"Connection failed: {e}")

    def close(self) -> None:
        """Close connection to Weaviate."""
        if self._client:
            self._client.close()
            self._client = None
            logger.info("Weaviate connection closed")

    @property
    def client(self) -> weaviate.WeaviateClient:
        """Get the Weaviate client instance.

        Returns:
            Connected Weaviate client.

        Raises:
            RuntimeError: If not connected.
        """
        if not self._client:
            raise RuntimeError("Not connected to Weaviate. Call connect() first.")
        return self._client

    def ensure_schema(self) -> None:
        """Ensure the ExecutionPlan schema exists.

        Creates the schema if it doesn't exist.
        """
        collection_name = EXECUTION_PLAN_SCHEMA["class"]

        if self.client.collections.exists(collection_name):
            logger.info(f"Collection '{collection_name}' already exists")
            return

        # Create collection with properties
        self.client.collections.create(
            name=collection_name,
            description=EXECUTION_PLAN_SCHEMA["description"],
            vectorizer_config=Configure.Vectorizer.text2vec_transformers(),
            properties=[
                Property(
                    name=prop["name"],
                    data_type=self._map_data_type(prop["dataType"][0]),
                    description=prop.get("description"),
                    skip_vectorization=prop.get("moduleConfig", {})
                    .get("text2vec-transformers", {})
                    .get("skip", False),
                )
                for prop in EXECUTION_PLAN_SCHEMA["properties"]
            ],
        )
        logger.info(f"Created collection '{collection_name}'")

    def _map_data_type(self, weaviate_type: str) -> DataType:
        """Map Weaviate schema type to DataType enum.

        Args:
            weaviate_type: Weaviate type string.

        Returns:
            Corresponding DataType enum value.
        """
        type_map = {
            "text": DataType.TEXT,
            "text[]": DataType.TEXT_ARRAY,
            "int": DataType.INT,
            "number": DataType.NUMBER,
            "boolean": DataType.BOOL,
            "date": DataType.DATE,
        }
        return type_map.get(weaviate_type, DataType.TEXT)

    def store_plan(
        self,
        intent_text: str,
        extracted_intent: str,
        plan_code: str,
        plan_explanation: str,
        affordance_uris: List[str],
        device_types: List[str],
        execution_context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Store a new execution plan.

        Args:
            intent_text: Original user intent.
            extracted_intent: Cleaned/structured intent.
            plan_code: Generated Python code.
            plan_explanation: Human-readable explanation.
            affordance_uris: URIs of used affordances.
            device_types: Types of devices involved.
            execution_context: Optional context dictionary.

        Returns:
            Generated plan_id.
        """
        plan_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()

        collection = self.client.collections.get("ExecutionPlan")
        collection.data.insert(
            properties={
                "plan_id": plan_id,
                "intent_text": intent_text,
                "extracted_intent": extracted_intent,
                "plan_code": plan_code,
                "plan_explanation": plan_explanation,
                "affordance_uris": affordance_uris,
                "device_types": device_types,
                "success_count": 0,
                "failure_count": 0,
                "created_at": now,
                "last_executed_at": now,
                "execution_context": json.dumps(execution_context or {}),
            }
        )

        logger.info(f"Stored plan with ID: {plan_id}")
        return plan_id

    def search_similar_plans(
        self,
        query: str,
        limit: int = 5,
        min_certainty: float = 0.7,
    ) -> List[Dict[str, Any]]:
        """Search for similar plans using semantic search.

        Args:
            query: Natural language query for semantic search.
            limit: Maximum number of results to return.
            min_certainty: Minimum certainty threshold (0.0 to 1.0).

        Returns:
            List of matching plans with metadata.
        """
        collection = self.client.collections.get("ExecutionPlan")

        response = collection.query.near_text(
            query=query,
            limit=limit,
            certainty=min_certainty,
            return_metadata=MetadataQuery(certainty=True),
        )

        results = []
        for obj in response.objects:
            plan_data = dict(obj.properties)
            plan_data["certainty"] = obj.metadata.certainty
            results.append(plan_data)

        logger.info(f"Found {len(results)} similar plans for query: {query[:50]}...")
        return results

    def get_plan_by_id(self, plan_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a plan by its ID.

        Args:
            plan_id: The plan identifier.

        Returns:
            Plan data dictionary or None if not found.
        """
        collection = self.client.collections.get("ExecutionPlan")

        response = collection.query.fetch_objects(
            filters=weaviate.classes.query.Filter.by_property("plan_id").equal(plan_id),
            limit=1,
        )

        if response.objects:
            return dict(response.objects[0].properties)
        return None

    def update_execution_stats(
        self,
        plan_id: str,
        success: bool,
    ) -> None:
        """Update execution statistics for a plan.

        Args:
            plan_id: The plan identifier.
            success: Whether the execution was successful.
        """
        collection = self.client.collections.get("ExecutionPlan")

        # Find the object by plan_id
        response = collection.query.fetch_objects(
            filters=weaviate.classes.query.Filter.by_property("plan_id").equal(plan_id),
            limit=1,
            include_vector=False,
        )

        if not response.objects:
            logger.warning(f"Plan not found for stats update: {plan_id}")
            return

        obj = response.objects[0]
        props = dict(obj.properties)

        # Update counts
        if success:
            props["success_count"] = props.get("success_count", 0) + 1
        else:
            props["failure_count"] = props.get("failure_count", 0) + 1

        props["last_executed_at"] = datetime.now(timezone.utc).isoformat()

        # Update the object
        collection.data.update(
            uuid=obj.uuid,
            properties=props,
        )

        logger.info(f"Updated stats for plan {plan_id}: success={success}")

    def delete_plan(self, plan_id: str) -> bool:
        """Delete a plan by its ID.

        Args:
            plan_id: The plan identifier.

        Returns:
            True if deleted, False if not found.
        """
        collection = self.client.collections.get("ExecutionPlan")

        response = collection.query.fetch_objects(
            filters=weaviate.classes.query.Filter.by_property("plan_id").equal(plan_id),
            limit=1,
        )

        if response.objects:
            collection.data.delete_by_id(response.objects[0].uuid)
            logger.info(f"Deleted plan: {plan_id}")
            return True

        logger.warning(f"Plan not found for deletion: {plan_id}")
        return False


# Module-level client instance for convenience
_client: Optional[WeaviateClient] = None


def get_weaviate_client() -> WeaviateClient:
    """Get or create the global Weaviate client.

    Returns:
        Connected WeaviateClient instance.
    """
    global _client
    if _client is None:
        _client = WeaviateClient()
        _client.connect()
        _client.ensure_schema()
    return _client
