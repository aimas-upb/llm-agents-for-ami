"""
Ontology loader utilities for working with OWL files.

Provides functionality to load and cache HMAS and CASHMERE ontologies
using owlready2.
"""

import logging
import os
from pathlib import Path
from typing import Optional

from owlready2 import get_ontology, Ontology

logger = logging.getLogger(__name__)

# Path to the ontologies directory
ONTOLOGIES_DIR = Path(__file__).parent


class OntologyLoader:
    """
    Utility class for loading and caching OWL ontologies.

    This class provides methods to load ontologies from local files
    and maintains a cache to avoid reloading the same ontology multiple times.
    """

    _cache: dict[str, Ontology] = {}

    @classmethod
    def load_ontology(cls, file_name: str, reload: bool = False) -> Optional[Ontology]:
        """
        Load an ontology from a file.

        Args:
            file_name: The name of the ontology file (e.g., "hmas.owl").
            reload: If True, reload the ontology even if it's cached.

        Returns:
            The loaded Ontology object, or None if loading fails.
        """
        # Check cache first
        if not reload and file_name in cls._cache:
            logger.debug(f"Returning cached ontology: {file_name}")
            return cls._cache[file_name]

        # Build the full path to the ontology file
        ontology_path = ONTOLOGIES_DIR / file_name

        if not ontology_path.exists():
            logger.error(f"Ontology file not found: {ontology_path}")
            return None

        try:
            logger.info(f"Loading ontology from: {ontology_path}")

            # Load the ontology using owlready2
            # Convert to file:// URI format
            file_uri = ontology_path.as_uri()

            # owlready2 supports: OWL/XML (.owl), RDF/XML (.rdf), and NTriples (.nt)
            # Turtle, N3, and other formats are NOT supported
            file_ext = ontology_path.suffix.lower()
            logger.debug(f"Loading ontology with extension: {file_ext}")

            # Load the ontology - owlready2 will auto-detect the format for supported types
            ontology = get_ontology(file_uri).load()

            # Cache the loaded ontology
            cls._cache[file_name] = ontology

            logger.info(f"Successfully loaded ontology: {file_name} (IRI: {ontology.base_iri})")
            return ontology

        except Exception as e:
            logger.error(f"Failed to load ontology {file_name}: {e}", exc_info=True)
            return None

    @classmethod
    def get_ontology_path(cls, file_name: str) -> Path:
        """
        Get the full path to an ontology file.

        Args:
            file_name: The name of the ontology file.

        Returns:
            Path object representing the full path to the ontology file.
        """
        return ONTOLOGIES_DIR / file_name

    @classmethod
    def clear_cache(cls):
        """Clear the ontology cache."""
        cls._cache.clear()
        logger.info("Ontology cache cleared")


def get_hmas_ontology(reload: bool = False) -> Optional[Ontology]:
    """
    Load the HMAS (Hypermedia MAS) ontology.

    Args:
        reload: If True, reload the ontology even if it's cached.

    Returns:
        The HMAS Ontology object, or None if loading fails.
    """
    # Try supported formats: .owl (OWL/XML), .rdf (RDF/XML), .nt (NTriples)
    for ext in ["hmas.owl", "hmas.rdf", "hmas.nt"]:
        if (ONTOLOGIES_DIR / ext).exists():
            return OntologyLoader.load_ontology(ext, reload=reload)

    logger.error("HMAS ontology file not found (tried hmas.owl, hmas.rdf, hmas.nt)")
    return None


def get_cashmere_ontology(reload: bool = False) -> Optional[Ontology]:
    """
    Load the CASHMERE ontology for signifier representation.

    Args:
        reload: If True, reload the ontology even if it's cached.

    Returns:
        The CASHMERE Ontology object, or None if loading fails.
    """
    # Try supported formats: .owl (OWL/XML), .rdf (RDF/XML), .nt (NTriples)
    for ext in ["cashmere.owl", "cashmere.rdf", "cashmere.nt"]:
        if (ONTOLOGIES_DIR / ext).exists():
            return OntologyLoader.load_ontology(ext, reload=reload)

    logger.error("CASHMERE ontology file not found (tried cashmere.owl, cashmere.rdf, cashmere.nt)")
    return None


def get_td_ontology(reload: bool = False) -> Optional[Ontology]:
    """
    Load the W3C Thing Description (TD) ontology.

    Args:
        reload: If True, reload the ontology even if it's cached.

    Returns:
        The TD Ontology object, or None if loading fails.
    """
    # Try supported formats: .owl (OWL/XML), .rdf (RDF/XML), .nt (NTriples)
    for ext in ["td.owl", "td.rdf", "td.nt"]:
        if (ONTOLOGIES_DIR / ext).exists():
            return OntologyLoader.load_ontology(ext, reload=reload)

    logger.error("TD ontology file not found (tried td.owl, td.rdf, td.nt)")
    return None


def get_hctl_ontology(reload: bool = False) -> Optional[Ontology]:
    """
    Load the HCTL (Hypermedia Control Transfer Language) ontology.

    Args:
        reload: If True, reload the ontology even if it's cached.

    Returns:
        The HCTL Ontology object, or None if loading fails.
    """
    # Try supported formats: .owl (OWL/XML), .rdf (RDF/XML), .nt (NTriples)
    for ext in ["hctl.owl", "hctl.rdf", "hctl.nt"]:
        if (ONTOLOGIES_DIR / ext).exists():
            return OntologyLoader.load_ontology(ext, reload=reload)

    logger.error("HCTL ontology file not found (tried hctl.owl, hctl.rdf, hctl.nt)")
    return None


def get_http_ontology(reload: bool = False) -> Optional[Ontology]:
    """
    Load the HTTP (Hypertext Transfer Protocol) ontology.

    Args:
        reload: If True, reload the ontology even if it's cached.
    Returns:
        The HTTP Ontology object, or None if loading fails.
    """
    # Try supported formats: .owl (OWL/XML), .rdf (RDF/XML), .nt (NTriples)
    for ext in ["http.owl", "http.rdf", "http.nt"]:
        if (ONTOLOGIES_DIR / ext).exists():
            return OntologyLoader.load_ontology(ext, reload=reload)

    logger.error("HTTP ontology file not found (tried http.owl, http.rdf, http.nt)")
    return None