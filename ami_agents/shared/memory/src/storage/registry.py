"""Signifier Registry module for CRUD operations.

This module provides high-level operations for managing signifiers,
coordinating Memory Store and Representation Service.
"""

import logging
from typing import Dict, List, Optional

from src.models.signifier import Signifier, SignifierStatus
from src.storage.memory_store import MemoryStore
from src.storage.representation import RepresentationService
from src.validation import AuthoringValidationError, AuthoringValidator

logger = logging.getLogger(__name__)


class SignifierRegistry:
    """Registry for managing signifier lifecycle.

    This class provides:
    - CRUD operations for signifiers
    - Versioning support
    - Status management
    - Querying and filtering
    """

    def __init__(
        self,
        storage_dir: Optional[str] = None,
        enable_authoring_validation: bool = False,
    ):
        """Initialize the registry.

        Args:
            storage_dir: Directory for file-based storage (optional)
            enable_authoring_validation: Enable SHACL validation at ingest
        """
        self.store = MemoryStore(storage_dir)
        self.repr_service = RepresentationService()
        self.enable_authoring_validation = enable_authoring_validation
        self.authoring_validator = AuthoringValidator(strict_mode=False)
        logger.info(
            f"Initialized SignifierRegistry "
            f"(authoring_validation={enable_authoring_validation})"
        )

    def create(self, signifier: Signifier, rdf_data: Optional[str] = None) -> Signifier:
        """Create a new signifier.

        Args:
            signifier: Signifier instance
            rdf_data: Optional RDF representation

        Returns:
            Created signifier

        Raises:
            ValueError: If signifier already exists or validation fails
        """
        existing = self.store.get_json_document(signifier.signifier_id)
        if existing:
            raise ValueError(
                f"Signifier {signifier.signifier_id} already exists. Use update instead."
            )

        if self.enable_authoring_validation:
            try:
                self.authoring_validator.validate_and_raise(signifier)
                logger.info(
                    f"Authoring validation passed for {signifier.signifier_id}"
                )
            except AuthoringValidationError as e:
                logger.error(f"Authoring validation failed: {e}")
                raise ValueError(f"Authoring validation failed: {e}")

        normalized = self.repr_service.normalize_signifier(signifier)

        self.store.store_json_document(normalized)

        self.store.update_property_index(normalized)

        if rdf_data:
            try:
                self.store.store_rdf_graph(
                    normalized.signifier_id, normalized.version, rdf_data
                )
            except ValueError as e:
                logger.warning(f"Failed to store RDF data: {e}")
        else:
            rdf_generated = self.repr_service.generate_rdf(normalized)
            self.store.store_rdf_graph(
                normalized.signifier_id,
                normalized.version,
                rdf_generated,
            )

        logger.info(f"Created signifier: {normalized.signifier_id}")
        return normalized

    def create_from_rdf(self, rdf_data: str, format: str = "turtle") -> Signifier:
        """Create signifier from RDF data.

        Args:
            rdf_data: RDF data as string
            format: RDF serialization format

        Returns:
            Created signifier

        Raises:
            ValueError: If parsing or creation fails
        """
        signifier = self.repr_service.parse_rdf_signifier(rdf_data, format)

        return self.create(signifier, rdf_data)

    def get(self, signifier_id: str) -> Optional[Signifier]:
        """Retrieve signifier by ID.

        Args:
            signifier_id: Signifier identifier

        Returns:
            Signifier instance or None if not found
        """
        doc = self.store.get_json_document(signifier_id)
        if not doc:
            return None

        try:
            signifier = Signifier(**doc)
            logger.debug(f"Retrieved signifier: {signifier_id}")
            return signifier
        except Exception as e:
            logger.error(f"Failed to deserialize signifier {signifier_id}: {e}")
            return None

    def update(self, signifier: Signifier, create_new_version: bool = False) -> Signifier:
        """Update existing signifier.

        Args:
            signifier: Updated signifier instance
            create_new_version: If True, increment version for breaking changes

        Returns:
            Updated signifier

        Raises:
            ValueError: If signifier doesn't exist
        """
        existing_doc = self.store.get_json_document(signifier.signifier_id)
        if not existing_doc:
            raise ValueError(
                f"Signifier {signifier.signifier_id} not found. Use create instead."
            )

        if create_new_version:
            signifier.version = existing_doc.get("version", 1) + 1
            logger.info(
                f"Creating new version {signifier.version} for {signifier.signifier_id}"
            )

        normalized = self.repr_service.normalize_signifier(signifier)

        self.store.store_json_document(normalized)

        self.store.update_property_index(normalized)

        rdf_generated = self.repr_service.generate_rdf(normalized)
        self.store.store_rdf_graph(
            normalized.signifier_id,
            normalized.version,
            rdf_generated,
        )

        logger.info(f"Updated signifier: {normalized.signifier_id} v{normalized.version}")
        return normalized

    def delete(self, signifier_id: str) -> bool:
        """Delete signifier and all its versions.

        Args:
            signifier_id: Signifier identifier

        Returns:
            True if deletion succeeded
        """
        success = self.store.delete_signifier(signifier_id)
        if success:
            logger.info(f"Deleted signifier: {signifier_id}")
        else:
            logger.warning(f"Failed to delete signifier: {signifier_id}")
        return success

    def list_signifiers(
        self,
        status: Optional[SignifierStatus] = None,
        affordance_uri: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Signifier]:
        """List signifiers with optional filtering.

        Args:
            status: Filter by status (active or deprecated)
            affordance_uri: Filter by affordance URI
            limit: Maximum number of results
            offset: Number of results to skip

        Returns:
            List of signifiers matching criteria
        """
        all_ids = self.store.list_all_signifiers()

        signifiers = []
        for signifier_id in all_ids:
            signifier = self.get(signifier_id)
            if not signifier:
                continue

            if status and signifier.status != status:
                continue

            if affordance_uri and signifier.affordance_uri != affordance_uri:
                continue

            signifiers.append(signifier)

        signifiers = signifiers[offset : offset + limit]

        logger.debug(f"Listed {len(signifiers)} signifiers (total: {len(all_ids)})")
        return signifiers

    def find_by_property(
        self, artifact_uri: str, property_uri: str
    ) -> List[Signifier]:
        """Find signifiers that reference a specific artifact property.

        Args:
            artifact_uri: Artifact URI
            property_uri: Property URI

        Returns:
            List of signifiers referencing this property
        """
        signifier_ids = self.store.find_by_property(artifact_uri, property_uri)

        signifiers = []
        for signifier_id in signifier_ids:
            signifier = self.get(signifier_id)
            if signifier:
                signifiers.append(signifier)

        logger.debug(
            f"Found {len(signifiers)} signifiers for property {artifact_uri}, {property_uri}"
        )
        return signifiers

    def get_rdf_representation(
        self, signifier_id: str, version: Optional[int] = None
    ) -> Optional[str]:
        """Get RDF representation of a signifier.

        Args:
            signifier_id: Signifier identifier
            version: Specific version (None uses current version)

        Returns:
            RDF data as Turtle string or None if not found
        """
        if version is None:
            signifier = self.get(signifier_id)
            if not signifier:
                return None
            version = signifier.version

        graph = self.store.get_rdf_graph(signifier_id, version)
        if not graph:
            return None

        return graph.serialize(format="turtle")

    def update_status(
        self, signifier_id: str, status: SignifierStatus
    ) -> Optional[Signifier]:
        """Update signifier status.

        Args:
            signifier_id: Signifier identifier
            status: New status

        Returns:
            Updated signifier or None if not found
        """
        signifier = self.get(signifier_id)
        if not signifier:
            return None

        signifier.status = status
        return self.update(signifier, create_new_version=False)
