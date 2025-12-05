"""Memory Store module for RDF and JSON storage.

This module provides dual storage:
- RDF store with named graphs for each signifier version
- JSON document store for canonical signifiers
- Property index catalog for fast candidate prefiltering
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from rdflib import Graph, Namespace, URIRef

from src.config import get_settings
from src.models.signifier import Signifier

logger = logging.getLogger(__name__)


class MemoryStore:
    """Memory store for signifiers with dual RDF and JSON storage.

    This class manages:
    - RDF named graphs (one per signifier version)
    - JSON document store
    - Property index catalog for (artifact, property) lookups
    """

    def __init__(self, storage_dir: Optional[str] = None):
        """Initialize the memory store.

        Args:
            storage_dir: Directory for file-based storage (optional)
        """
        settings = get_settings()
        self.storage_dir = Path(storage_dir or settings.storage_dir)
        self.rdf_dir = self.storage_dir / "rdf"
        self.json_dir = self.storage_dir / "json"
        self.index_dir = self.storage_dir / "indexes"

        self.rdf_dir.mkdir(parents=True, exist_ok=True)
        self.json_dir.mkdir(parents=True, exist_ok=True)
        self.index_dir.mkdir(parents=True, exist_ok=True)

        self.property_index: Dict[Tuple[str, str], Set[str]] = {}
        self._load_property_index()

        logger.info(f"Initialized MemoryStore at {self.storage_dir}")

    def _get_graph_uri(self, signifier_id: str, version: int) -> str:
        """Generate named graph URI for a signifier version.

        Args:
            signifier_id: Signifier identifier
            version: Version number

        Returns:
            Named graph URI
        """
        return f"graph:signifier/{signifier_id}/{version}"

    def _get_rdf_path(self, signifier_id: str, version: int) -> Path:
        """Get file path for RDF graph.

        Args:
            signifier_id: Signifier identifier
            version: Version number

        Returns:
            Path to RDF file
        """
        return self.rdf_dir / f"{signifier_id}_v{version}.ttl"

    def _get_json_path(self, signifier_id: str) -> Path:
        """Get file path for JSON document.

        Args:
            signifier_id: Signifier identifier

        Returns:
            Path to JSON file
        """
        return self.json_dir / f"{signifier_id}.json"

    def _load_property_index(self) -> None:
        """Load property index from disk."""
        index_path = self.index_dir / "property_index.json"
        if index_path.exists():
            try:
                with open(index_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.property_index = {
                        tuple(k.split("|")): set(v) for k, v in data.items()
                    }
                logger.info(
                    f"Loaded property index with {len(self.property_index)} entries"
                )
            except Exception as e:
                logger.error(f"Failed to load property index: {e}")
                self.property_index = {}

    def _save_property_index(self) -> None:
        """Save property index to disk."""
        index_path = self.index_dir / "property_index.json"
        try:
            data = {
                "|".join(k): list(v) for k, v in self.property_index.items()
            }
            with open(index_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            logger.debug("Saved property index to disk")
        except Exception as e:
            logger.error(f"Failed to save property index: {e}")

    def store_rdf_graph(
        self, signifier_id: str, version: int, rdf_data: str, format: str = "turtle"
    ) -> str:
        """Store RDF graph for a signifier version.

        Args:
            signifier_id: Signifier identifier
            version: Version number
            rdf_data: RDF data as string
            format: RDF serialization format (default: turtle)

        Returns:
            Named graph URI

        Raises:
            ValueError: If RDF data is invalid
        """
        graph_uri = self._get_graph_uri(signifier_id, version)

        try:
            graph = Graph()
            graph.parse(data=rdf_data, format=format)

            rdf_path = self._get_rdf_path(signifier_id, version)
            graph.serialize(destination=str(rdf_path), format="turtle")

            logger.info(
                f"Stored RDF graph for {signifier_id} v{version} at {rdf_path}"
            )
            return graph_uri

        except Exception as e:
            logger.error(f"Failed to store RDF graph: {e}")
            raise ValueError(f"Invalid RDF data: {e}")

    def get_rdf_graph(
        self, signifier_id: str, version: int
    ) -> Optional[Graph]:
        """Retrieve RDF graph for a signifier version.

        Args:
            signifier_id: Signifier identifier
            version: Version number

        Returns:
            RDF Graph or None if not found
        """
        rdf_path = self._get_rdf_path(signifier_id, version)

        if not rdf_path.exists():
            logger.debug(
                f"RDF graph not found for {signifier_id} v{version}"
            )
            return None

        try:
            graph = Graph()
            graph.parse(str(rdf_path), format="turtle")
            logger.debug(f"Retrieved RDF graph for {signifier_id} v{version}")
            return graph
        except Exception as e:
            logger.error(f"Failed to load RDF graph: {e}")
            return None

    def store_json_document(self, signifier: Signifier) -> None:
        """Store JSON document for a signifier.

        Args:
            signifier: Signifier instance

        Raises:
            ValueError: If storage fails
        """
        json_path = self._get_json_path(signifier.signifier_id)

        try:
            doc = signifier.to_json_doc()
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(doc, f, indent=2)

            logger.info(
                f"Stored JSON document for {signifier.signifier_id} at {json_path}"
            )
        except Exception as e:
            logger.error(f"Failed to store JSON document: {e}")
            raise ValueError(f"Failed to store signifier: {e}")

    def get_json_document(self, signifier_id: str) -> Optional[Dict]:
        """Retrieve JSON document for a signifier.

        Args:
            signifier_id: Signifier identifier

        Returns:
            JSON document dictionary or None if not found
        """
        json_path = self._get_json_path(signifier_id)

        if not json_path.exists():
            logger.debug(f"JSON document not found for {signifier_id}")
            return None

        try:
            with open(json_path, "r", encoding="utf-8") as f:
                doc = json.load(f)
            logger.debug(f"Retrieved JSON document for {signifier_id}")
            return doc
        except Exception as e:
            logger.error(f"Failed to load JSON document: {e}")
            return None

    def update_property_index(self, signifier: Signifier) -> None:
        """Update property index catalog with signifier's properties.

        Args:
            signifier: Signifier instance
        """
        property_keys = signifier.get_property_keys()

        for artifact_uri, property_uri in property_keys:
            key = (artifact_uri, property_uri)
            if key not in self.property_index:
                self.property_index[key] = set()
            self.property_index[key].add(signifier.signifier_id)

        self._save_property_index()
        logger.debug(
            f"Updated property index for {signifier.signifier_id}: {property_keys}"
        )

    def find_by_property(
        self, artifact_uri: str, property_uri: str
    ) -> List[str]:
        """Find signifier IDs by artifact and property.

        Args:
            artifact_uri: Artifact URI
            property_uri: Property URI

        Returns:
            List of signifier IDs that reference this property
        """
        key = (artifact_uri, property_uri)
        signifier_ids = list(self.property_index.get(key, set()))
        logger.debug(
            f"Found {len(signifier_ids)} signifiers for {artifact_uri}, {property_uri}"
        )
        return signifier_ids

    def list_all_signifiers(self) -> List[str]:
        """List all signifier IDs in the store.

        Returns:
            List of signifier IDs
        """
        signifier_ids = []
        for json_file in self.json_dir.glob("*.json"):
            signifier_ids.append(json_file.stem)

        logger.debug(f"Found {len(signifier_ids)} signifiers in store")
        return signifier_ids

    def delete_signifier(self, signifier_id: str, version: Optional[int] = None) -> bool:
        """Delete signifier data.

        Args:
            signifier_id: Signifier identifier
            version: Optional specific version to delete (None deletes all)

        Returns:
            True if deletion succeeded
        """
        try:
            if version is not None:
                rdf_path = self._get_rdf_path(signifier_id, version)
                if rdf_path.exists():
                    rdf_path.unlink()
                    logger.info(f"Deleted RDF for {signifier_id} v{version}")
            else:
                for rdf_file in self.rdf_dir.glob(f"{signifier_id}_v*.ttl"):
                    rdf_file.unlink()
                logger.info(f"Deleted all RDF versions for {signifier_id}")

                json_path = self._get_json_path(signifier_id)
                if json_path.exists():
                    json_path.unlink()
                    logger.info(f"Deleted JSON for {signifier_id}")

                for key in list(self.property_index.keys()):
                    self.property_index[key].discard(signifier_id)
                    if not self.property_index[key]:
                        del self.property_index[key]
                self._save_property_index()

            return True

        except Exception as e:
            logger.error(f"Failed to delete signifier {signifier_id}: {e}")
            return False
