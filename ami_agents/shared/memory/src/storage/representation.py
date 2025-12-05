"""Representation Service for normalizing signifier data.

This module handles conversion between RDF and JSON representations,
and prepares signifiers for storage.
"""

import json
import logging
import re
from typing import Dict, Optional

from rdflib import Graph, Literal, Namespace, RDF, URIRef
from rdflib.namespace import XSD

from src.models.signifier import (
    IntentContext,
    IntentionDescription,
    Provenance,
    Signifier,
    SignifierStatus,
    StructuredCondition,
    ValueCondition,
)

logger = logging.getLogger(__name__)

CASHMERE = Namespace("https://aimas.cs.pub.ro/ont/cashmere#")
SH = Namespace("http://www.w3.org/ns/shacl#")


class RepresentationService:
    """Service for normalizing and converting signifier representations.

    This service handles:
    - Parsing RDF signifiers to internal models
    - Generating RDF from internal models
    - Extracting structured conditions from SHACL shapes
    - Normalizing natural language and structured fields
    """

    @staticmethod
    def _preprocess_rdf(rdf_data: str) -> str:
        """Preprocess RDF data to handle non-standard syntax.

        Args:
            rdf_data: Raw RDF data

        Returns:
            Preprocessed RDF data
        """
        prefixes = [
            "@prefix cashmere: <https://aimas.cs.pub.ro/ont/cashmere#> .",
            "@prefix sh: <http://www.w3.org/ns/shacl#> .",
            "@prefix hmas: <https://aimas.cs.pub.ro/ont/cashmere#> .",
            "@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .",
            "@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .",
            "",
        ]

        lines = []
        for line in rdf_data.split("\n"):
            if "//" in line:
                comment_pos = line.find("//")
                line = line[:comment_pos].rstrip()

            if line.strip():
                lines.append(line)

        processed = "\n".join(lines)

        pattern = r'cashmere:hasStructuredDescription\s+"(.*?)"(?:\^\^xsd:string)?'

        def replace_multiline_string(match):
            content = match.group(1).strip()
            content_fixed = re.sub(r'<(http[^>]+)>', r"'\1'", content)
            return f'cashmere:hasStructuredDescription """{content_fixed}"""^^xsd:string'

        processed = re.sub(pattern, replace_multiline_string, processed, flags=re.DOTALL)

        return "\n".join(prefixes) + "\n" + processed

    @staticmethod
    def parse_rdf_signifier(rdf_data: str, format: str = "turtle") -> Signifier:
        """Parse RDF signifier into internal Signifier model.

        Args:
            rdf_data: RDF data as string
            format: RDF serialization format

        Returns:
            Parsed Signifier instance

        Raises:
            ValueError: If RDF parsing fails or required fields are missing
        """
        try:
            if "@prefix" not in rdf_data:
                rdf_data = RepresentationService._preprocess_rdf(rdf_data)

            graph = Graph()
            graph.parse(data=rdf_data, format=format)

            signifier_nodes = list(graph.subjects(RDF.type, CASHMERE.Signifier))
            if not signifier_nodes:
                raise ValueError("No Signifier found in RDF data")

            signifier_node = signifier_nodes[0]
            signifier_id = str(signifier_node).split("#")[-1]

            affordance_uri = graph.value(signifier_node, CASHMERE.signifies)
            if not affordance_uri:
                raise ValueError("Missing cashmere:signifies property")

            intent_node = graph.value(
                signifier_node, CASHMERE.hasIntentionDescription
            )
            if not intent_node:
                raise ValueError("Missing cashmere:hasIntentionDescription")

            intent_nl = graph.value(intent_node, CASHMERE.hasStructuredDescription)
            if not intent_nl:
                raise ValueError("Missing intent description")

            intent_dict = json.loads(str(intent_nl))
            intent = IntentionDescription(
                nl_text=intent_dict.get("intent", ""),
                structured=intent_dict,
            )

            context_node = graph.value(signifier_node, CASHMERE.recommendsContext)
            context = IntentContext()

            if context_node:
                context_nl = graph.value(
                    context_node, CASHMERE.hasStructuredDescription
                )
                if context_nl:
                    context.nl_description = str(context_nl)
                    try:
                        context_dict = json.loads(str(context_nl))
                        conditions = context_dict.get("conditions", [])
                        context.structured_conditions = [
                            RepresentationService._parse_structured_condition(c)
                            for c in conditions
                        ]
                    except json.JSONDecodeError as e:
                        logger.warning(f"Failed to parse context description: {e}")

                shacl_shapes = list(
                    graph.objects(context_node, CASHMERE.hasShaclCondition)
                )
                if shacl_shapes:
                    context.shacl_shapes = RepresentationService._extract_shacl_shapes(
                        graph, shacl_shapes
                    )

            provenance = Provenance(
                created_by="system",
                source="rdf_import",
            )

            signifier = Signifier(
                signifier_id=signifier_id,
                version=1,
                status=SignifierStatus.ACTIVE,
                intent=intent,
                context=context,
                affordance_uri=str(affordance_uri),
                provenance=provenance,
            )

            logger.info(f"Parsed RDF signifier: {signifier_id}")
            return signifier

        except Exception as e:
            logger.error(f"Failed to parse RDF signifier: {e}")
            raise ValueError(f"Invalid RDF signifier: {e}")

    @staticmethod
    def _parse_structured_condition(condition_dict: Dict) -> StructuredCondition:
        """Parse structured condition from dictionary.

        Args:
            condition_dict: Condition dictionary

        Returns:
            StructuredCondition instance
        """
        value_conditions = []
        for vc in condition_dict.get("valueConditions", []):
            value_conditions.append(
                ValueCondition(
                    operator=vc.get("operator", "equals"),
                    value=vc.get("value"),
                    datatype=vc.get("datatype"),
                )
            )

        return StructuredCondition(
            artifact=str(condition_dict.get("artifact", "")),
            property_affordance=str(condition_dict.get("propertyAffordance", "")),
            value_conditions=value_conditions,
        )

    @staticmethod
    def _extract_shacl_shapes(graph: Graph, shape_nodes: list) -> str:
        """Extract SHACL shapes as Turtle string.

        Args:
            graph: RDF Graph
            shape_nodes: List of SHACL NodeShape nodes

        Returns:
            SHACL shapes serialized as Turtle
        """
        shapes_graph = Graph()
        shapes_graph.bind("sh", SH)
        shapes_graph.bind("xsd", XSD)

        for shape_node in shape_nodes:
            for s, p, o in graph.triples((shape_node, None, None)):
                shapes_graph.add((s, p, o))

            property_shapes = graph.objects(shape_node, SH.property)
            for prop_shape in property_shapes:
                for s, p, o in graph.triples((prop_shape, None, None)):
                    shapes_graph.add((s, p, o))

        return shapes_graph.serialize(format="turtle")

    @staticmethod
    def generate_rdf(signifier: Signifier, base_uri: str = "") -> str:
        """Generate RDF representation from Signifier model.

        Args:
            signifier: Signifier instance
            base_uri: Base URI for the signifier

        Returns:
            RDF serialized as Turtle
        """
        graph = Graph()
        graph.bind("cashmere", CASHMERE)
        graph.bind("sh", SH)
        graph.bind("xsd", XSD)

        if not base_uri:
            base_uri = f"http://example.org/signifiers"

        signifier_uri = URIRef(f"{base_uri}#{signifier.signifier_id}")

        graph.add((signifier_uri, RDF.type, CASHMERE.Signifier))

        graph.add(
            (signifier_uri, CASHMERE.signifies, URIRef(signifier.affordance_uri))
        )

        intent_node = URIRef(f"{base_uri}#{signifier.signifier_id}-intent")
        graph.add((signifier_uri, CASHMERE.hasIntentionDescription, intent_node))
        graph.add((intent_node, RDF.type, CASHMERE.IntentionDescription))

        intent_json = json.dumps(
            signifier.intent.structured
            or {"intent": signifier.intent.nl_text}
        )
        graph.add(
            (
                intent_node,
                CASHMERE.hasStructuredDescription,
                Literal(intent_json, datatype=XSD.string),
            )
        )

        if signifier.context:
            context_node = URIRef(f"{base_uri}#{signifier.signifier_id}-context")
            graph.add((signifier_uri, CASHMERE.recommendsContext, context_node))
            graph.add((context_node, RDF.type, CASHMERE.IntentContext))

            if signifier.context.nl_description:
                graph.add(
                    (
                        context_node,
                        CASHMERE.hasStructuredDescription,
                        Literal(
                            signifier.context.nl_description, datatype=XSD.string
                        ),
                    )
                )

            if signifier.context.shacl_shapes:
                shapes_graph = Graph()
                shapes_graph.parse(
                    data=signifier.context.shacl_shapes, format="turtle"
                )
                for s, p, o in shapes_graph:
                    graph.add((s, p, o))
                    if (s, RDF.type, SH.NodeShape) in shapes_graph:
                        graph.add((context_node, CASHMERE.hasShaclCondition, s))

        logger.debug(f"Generated RDF for signifier {signifier.signifier_id}")
        return graph.serialize(format="turtle")

    @staticmethod
    def normalize_signifier(signifier: Signifier) -> Signifier:
        """Normalize signifier data (placeholder for future enhancements).

        Args:
            signifier: Signifier instance

        Returns:
            Normalized signifier
        """
        logger.debug(f"Normalized signifier {signifier.signifier_id}")
        return signifier
