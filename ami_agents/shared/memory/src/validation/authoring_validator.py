"""Authoring validator for signifier structure validation.

This module validates signifier objects at ingest time to ensure
they have required properties and valid SHACL shapes.
"""

import logging
from typing import List, Optional

from rdflib import Graph, SH, URIRef

from src.models.signifier import Signifier

logger = logging.getLogger(__name__)


class AuthoringValidationError(Exception):
    """Exception raised when signifier authoring validation fails."""

    pass


class AuthoringValidator:
    """Validator for signifier authoring constraints.

    This validator checks signifier structure at ingest time to ensure:
    - Required properties are present
    - SHACL shapes are well-formed
    - Property paths use valid IRIs
    """

    def __init__(self, strict_mode: bool = False):
        """Initialize the authoring validator.

        Args:
            strict_mode: If True, enforce all optional checks as well
        """
        self.strict_mode = strict_mode
        logger.info(f"Authoring Validator initialized (strict={strict_mode})")

    def validate_signifier(
        self, signifier: Signifier, enable_shacl_check: bool = True
    ) -> List[str]:
        """Validate signifier structure.

        Args:
            signifier: The signifier to validate
            enable_shacl_check: Whether to validate SHACL shapes

        Returns:
            List of validation error messages (empty if valid)

        Raises:
            AuthoringValidationError: If validation fails in strict mode
        """
        errors: List[str] = []

        errors.extend(self._check_required_fields(signifier))

        if enable_shacl_check and signifier.context.shacl_shapes:
            errors.extend(self._check_shacl_shapes(signifier.context.shacl_shapes))

        if self.strict_mode:
            errors.extend(self._check_optional_fields(signifier))

        if errors:
            error_msg = f"Signifier validation failed: {'; '.join(errors)}"
            logger.warning(error_msg)
            if self.strict_mode:
                raise AuthoringValidationError(error_msg)

        return errors

    def _check_required_fields(self, signifier: Signifier) -> List[str]:
        """Check required signifier fields.

        Args:
            signifier: The signifier to check

        Returns:
            List of error messages
        """
        errors = []

        if not signifier.signifier_id:
            errors.append("Missing signifier_id")

        if not signifier.affordance_uri:
            errors.append("Missing affordance_uri (cashmere:signifies)")

        if not signifier.intent:
            errors.append("Missing intent (cashmere:hasIntentionDescription)")
        elif not signifier.intent.nl_text:
            errors.append("Missing intent.nl_text")

        if not signifier.context:
            errors.append("Missing context (cashmere:recommendsContext)")

        return errors

    def _check_optional_fields(self, signifier: Signifier) -> List[str]:
        """Check optional but recommended fields.

        Args:
            signifier: The signifier to check

        Returns:
            List of warning messages
        """
        warnings = []

        if not signifier.context.shacl_shapes and not signifier.context.structured_conditions:
            warnings.append(
                "No SHACL shapes or structured conditions defined"
            )

        if not signifier.provenance:
            warnings.append("Missing provenance information")

        return warnings

    def _check_shacl_shapes(self, shacl_shapes: str) -> List[str]:
        """Validate SHACL shapes are well-formed.

        Args:
            shacl_shapes: SHACL shapes as Turtle string

        Returns:
            List of error messages
        """
        errors = []

        try:
            shapes_graph = Graph()
            shapes_graph.parse(data=shacl_shapes, format="turtle")

            node_shapes = list(
                shapes_graph.subjects(predicate=None, object=SH.NodeShape)
            )

            if not node_shapes:
                errors.append("No sh:NodeShape found in SHACL shapes")
                return errors

            for shape in node_shapes:
                errors.extend(self._validate_node_shape(shapes_graph, shape))

        except Exception as e:
            errors.append(f"Failed to parse SHACL shapes: {e}")

        return errors

    def _validate_node_shape(
        self, graph: Graph, shape_node: URIRef
    ) -> List[str]:
        """Validate a single NodeShape.

        Args:
            graph: The shapes graph
            shape_node: The NodeShape to validate

        Returns:
            List of error messages
        """
        errors = []

        has_target = False
        for target_pred in [SH.targetNode, SH.targetClass, SH.targetSubjectsOf]:
            if (shape_node, target_pred, None) in graph:
                has_target = True
                break

        if not has_target:
            errors.append(
                f"NodeShape {shape_node} has no target (sh:targetNode, sh:targetClass, etc.)"
            )

        property_shapes = list(graph.objects(shape_node, SH.property))

        for prop_shape in property_shapes:
            path = graph.value(prop_shape, SH.path)
            if not path:
                errors.append(
                    f"Property shape {prop_shape} missing sh:path"
                )
                continue

            if not isinstance(path, URIRef):
                errors.append(
                    f"sh:path must be a valid IRI, got: {path}"
                )

            datatype = graph.value(prop_shape, SH.datatype)
            if datatype and not str(datatype).startswith("http://www.w3.org/2001/XMLSchema#"):
                logger.warning(
                    f"Unusual datatype for {path}: {datatype}"
                )

        return errors

    def validate_and_raise(
        self, signifier: Signifier, enable_shacl_check: bool = True
    ) -> None:
        """Validate signifier and raise exception if invalid.

        Args:
            signifier: The signifier to validate
            enable_shacl_check: Whether to validate SHACL shapes

        Raises:
            AuthoringValidationError: If validation fails
        """
        errors = self.validate_signifier(signifier, enable_shacl_check)
        if errors:
            raise AuthoringValidationError(
                f"Signifier validation failed: {'; '.join(errors)}"
            )
