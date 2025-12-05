"""Context Graph Builder for converting KV maps to RDF graphs.

This module converts key-value context maps into canonical RDF graphs
suitable for SHACL validation.
"""

import logging
from typing import Any, Dict, List, Tuple

from rdflib import Graph, Literal, Namespace, RDF, URIRef
from rdflib.namespace import XSD

logger = logging.getLogger(__name__)

RDF_NS = Namespace("http://www.w3.org/1999/02/22-rdf-syntax-ns#")
XSD_NS = Namespace("http://www.w3.org/2001/XMLSchema#")


class ContextGraphBuilder:
    """Builder for converting context snapshots to RDF graphs.

    This builder takes key-value context maps and converts them into
    RDF graphs with proper typing and structure for SHACL validation.
    """

    def __init__(self):
        """Initialize the Context Graph Builder."""
        logger.info("Context Graph Builder initialized")

    def build_from_kv(
        self, context_features: Dict[str, Dict[str, Any]]
    ) -> Tuple[Graph, Dict[Tuple[str, str], Any]]:
        """Build RDF graph from key-value context features.

        Args:
            context_features: Dictionary mapping artifact URIs to their properties
                             Format: {artifact_uri: {property_uri: value, ...}, ...}

        Returns:
            Tuple of (RDF Graph, extracted features dict)

        Raises:
            ValueError: If context features are invalid

        Example:
            >>> features = {
            ...     "http://example.org/artifacts/sensor1": {
            ...         "http://example.org/LightSensor#hasLuminosityLevel": 15000
            ...     }
            ... }
            >>> graph, extracted = builder.build_from_kv(features)
        """
        if not isinstance(context_features, dict):
            raise ValueError("context_features must be a dictionary")

        graph = Graph()
        graph.bind("rdf", RDF)
        graph.bind("xsd", XSD)

        extracted_features: Dict[Tuple[str, str], Any] = {}

        for artifact_uri, properties in context_features.items():
            if not isinstance(artifact_uri, str):
                logger.warning(f"Skipping non-string artifact URI: {artifact_uri}")
                continue

            artifact_node = URIRef(artifact_uri)

            if not isinstance(properties, dict):
                logger.warning(
                    f"Skipping non-dict properties for {artifact_uri}"
                )
                continue

            for property_uri, value in properties.items():
                if not isinstance(property_uri, str):
                    logger.warning(
                        f"Skipping non-string property URI: {property_uri}"
                    )
                    continue

                property_node = URIRef(property_uri)

                literal_value = self._convert_to_literal(value)
                graph.add((artifact_node, property_node, literal_value))

                extracted_features[(artifact_uri, property_uri)] = value

        logger.info(
            f"Built context graph with {len(graph)} triples from "
            f"{len(extracted_features)} features"
        )
        return graph, extracted_features

    def build_from_flat_dict(
        self, context_snapshot: Dict[str, Any]
    ) -> Tuple[Graph, Dict[Tuple[str, str], Any]]:
        """Build RDF graph from flat context snapshot.

        Args:
            context_snapshot: Flat dictionary with artifact.property keys
                             Format: {"artifact_uri::property_uri": value, ...}

        Returns:
            Tuple of (RDF Graph, extracted features dict)

        Example:
            >>> snapshot = {
            ...     "http://example.org/artifacts/sensor1::http://example.org/LightSensor#hasLuminosityLevel": 15000
            ... }
            >>> graph, extracted = builder.build_from_flat_dict(snapshot)
        """
        context_features: Dict[str, Dict[str, Any]] = {}

        for key, value in context_snapshot.items():
            if "::" in key:
                artifact_uri, property_uri = key.split("::", 1)
            else:
                logger.warning(
                    f"Skipping key without '::' separator: {key}"
                )
                continue

            if artifact_uri not in context_features:
                context_features[artifact_uri] = {}

            context_features[artifact_uri][property_uri] = value

        return self.build_from_kv(context_features)

    def normalize_context(
        self, context_input: Any
    ) -> Tuple[Graph, Dict[Tuple[str, str], Any]]:
        """Normalize context input and convert to RDF graph.

        This method accepts various context input formats and normalizes
        them into a canonical RDF graph.

        Args:
            context_input: Context in various formats (dict, Graph, etc.)

        Returns:
            Tuple of (RDF Graph, extracted features dict)

        Raises:
            ValueError: If context format is not supported
        """
        if isinstance(context_input, Graph):
            logger.debug("Context input is already an RDF Graph")
            return context_input, self._extract_features_from_graph(
                context_input
            )

        if isinstance(context_input, dict):
            if any("::" in key for key in context_input.keys()):
                logger.debug("Parsing flat dict context format")
                return self.build_from_flat_dict(context_input)
            else:
                logger.debug("Parsing nested dict context format")
                return self.build_from_kv(context_input)

        raise ValueError(
            f"Unsupported context input type: {type(context_input)}"
        )

    def _convert_to_literal(self, value: Any) -> Literal:
        """Convert Python value to RDF Literal with appropriate datatype.

        Args:
            value: Python value to convert

        Returns:
            RDF Literal with proper XSD datatype
        """
        if isinstance(value, bool):
            return Literal(value, datatype=XSD.boolean)
        elif isinstance(value, int):
            return Literal(value, datatype=XSD.integer)
        elif isinstance(value, float):
            return Literal(value, datatype=XSD.double)
        elif isinstance(value, str):
            return Literal(value, datatype=XSD.string)
        else:
            logger.warning(
                f"Unknown value type {type(value)}, defaulting to string"
            )
            return Literal(str(value), datatype=XSD.string)

    def _extract_features_from_graph(
        self, graph: Graph
    ) -> Dict[Tuple[str, str], Any]:
        """Extract (artifact, property) -> value features from graph.

        Args:
            graph: RDF Graph

        Returns:
            Dictionary of extracted features
        """
        features: Dict[Tuple[str, str], Any] = {}

        for s, p, o in graph:
            if isinstance(s, URIRef) and isinstance(p, URIRef):
                artifact_uri = str(s)
                property_uri = str(p)

                if isinstance(o, Literal):
                    value = o.toPython()
                else:
                    value = str(o)

                features[(artifact_uri, property_uri)] = value

        logger.debug(f"Extracted {len(features)} features from graph")
        return features

    def add_type_information(
        self,
        graph: Graph,
        artifact_types: Dict[str, str],
    ) -> Graph:
        """Add rdf:type information to artifacts in the graph.

        Args:
            graph: RDF Graph to augment
            artifact_types: Mapping of artifact URIs to their type URIs

        Returns:
            Augmented RDF Graph

        Example:
            >>> types = {
            ...     "http://example.org/artifacts/sensor1": "http://example.org/LightSensor"
            ... }
            >>> graph = builder.add_type_information(graph, types)
        """
        for artifact_uri, type_uri in artifact_types.items():
            artifact_node = URIRef(artifact_uri)
            type_node = URIRef(type_uri)
            graph.add((artifact_node, RDF.type, type_node))

        logger.debug(
            f"Added type information for {len(artifact_types)} artifacts"
        )
        return graph
