"""SHACL Validator module for validating context graphs.

This module implements SHACL validation capabilities using pyshacl.
It validates context graphs against SHACL shapes defined in signifiers.
"""

import hashlib
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from pyshacl import validate
from rdflib import Graph

logger = logging.getLogger(__name__)


@dataclass
class ViolationDetail:
    """Details about a single SHACL validation violation.

    Args:
        focus_node: The node that failed validation
        result_path: The property path that was violated
        message: Human-readable violation message
        severity: Severity level (Violation, Warning, Info)
        source_constraint_component: The SHACL constraint that failed
        value: The actual value that caused the violation (if applicable)
    """

    focus_node: str
    result_path: Optional[str]
    message: str
    severity: str = "Violation"
    source_constraint_component: Optional[str] = None
    value: Optional[str] = None

    def to_dict(self) -> Dict:
        """Convert violation to dictionary.

        Returns:
            Dictionary representation of the violation
        """
        return {
            "focus_node": self.focus_node,
            "result_path": self.result_path,
            "message": self.message,
            "severity": self.severity,
            "source_constraint_component": self.source_constraint_component,
            "value": self.value,
        }


@dataclass
class ValidationResult:
    """Result of SHACL validation.

    Args:
        conforms: Whether the data graph conforms to the shapes
        violations: List of violation details
        validation_report_text: Full validation report as text
        validation_report_graph: Validation report as RDF graph
    """

    conforms: bool
    violations: List[ViolationDetail]
    validation_report_text: Optional[str] = None
    validation_report_graph: Optional[Graph] = None

    def to_dict(self) -> Dict:
        """Convert validation result to dictionary.

        Returns:
            Dictionary representation suitable for API responses
        """
        return {
            "conforms": self.conforms,
            "violations": [v.to_dict() for v in self.violations],
            "violation_count": len(self.violations),
        }


class SHACLValidator:
    """SHACL Validator for validating context graphs against shapes.

    This validator uses pyshacl to validate RDF graphs against SHACL shapes.
    It supports caching of validation results and provides detailed violation reports.
    """

    def __init__(self, enable_caching: bool = True):
        """Initialize the SHACL Validator.

        Args:
            enable_caching: Enable validation result caching
        """
        self.enable_caching = enable_caching
        self._cache: Dict[str, ValidationResult] = {}
        logger.info("SHACL Validator initialized")

    def parse_shapes(self, shapes_data: str, format: str = "turtle") -> Graph:
        """Parse SHACL shapes from string into RDF Graph.

        Args:
            shapes_data: SHACL shapes as string
            format: RDF serialization format

        Returns:
            RDF Graph containing the shapes

        Raises:
            ValueError: If shapes parsing fails
        """
        try:
            shapes_graph = Graph()
            shapes_graph.parse(data=shapes_data, format=format)
            logger.debug(
                f"Parsed {len(shapes_graph)} triples from SHACL shapes"
            )
            return shapes_graph
        except Exception as e:
            logger.error(f"Failed to parse SHACL shapes: {e}")
            raise ValueError(f"Invalid SHACL shapes: {e}")

    def validate(
        self,
        data_graph: Graph,
        shapes_graph: Graph,
        use_cache: bool = True,
    ) -> ValidationResult:
        """Validate data graph against SHACL shapes.

        Args:
            data_graph: The RDF graph to validate
            shapes_graph: The SHACL shapes graph
            use_cache: Use cached result if available

        Returns:
            ValidationResult with conforms status and violations

        Raises:
            ValueError: If validation execution fails
        """
        cache_key = self._compute_cache_key(data_graph, shapes_graph)

        if use_cache and self.enable_caching and cache_key in self._cache:
            logger.debug("Returning cached validation result")
            return self._cache[cache_key]

        try:
            logger.debug("Executing SHACL validation")
            conforms, results_graph, results_text = validate(
                data_graph,
                shacl_graph=shapes_graph,
                inference="rdfs",
                abort_on_first=False,
                allow_infos=True,
                allow_warnings=True,
            )

            violations = self._parse_violations(results_graph)

            result = ValidationResult(
                conforms=conforms,
                violations=violations,
                validation_report_text=results_text,
                validation_report_graph=results_graph,
            )

            if self.enable_caching:
                self._cache[cache_key] = result

            logger.info(
                f"Validation complete: conforms={conforms}, "
                f"violations={len(violations)}"
            )
            return result

        except Exception as e:
            logger.error(f"SHACL validation failed: {e}")
            raise ValueError(f"Validation execution failed: {e}")

    def validate_signifier_context(
        self,
        context_graph: Graph,
        shapes_data: str,
        format: str = "turtle",
    ) -> ValidationResult:
        """Validate context graph against signifier's SHACL shapes.

        Args:
            context_graph: The context graph to validate
            shapes_data: SHACL shapes as string
            format: RDF serialization format

        Returns:
            ValidationResult

        Raises:
            ValueError: If validation fails
        """
        shapes_graph = self.parse_shapes(shapes_data, format)
        return self.validate(context_graph, shapes_graph)

    def _parse_violations(self, results_graph: Graph) -> List[ViolationDetail]:
        """Parse violations from validation results graph.

        Args:
            results_graph: RDF graph containing validation results

        Returns:
            List of ViolationDetail objects
        """
        from rdflib import SH

        violations = []

        for result_node in results_graph.subjects(
            SH.resultSeverity, SH.Violation
        ):
            focus_node = results_graph.value(result_node, SH.focusNode)
            result_path = results_graph.value(result_node, SH.resultPath)
            message = results_graph.value(result_node, SH.resultMessage)
            constraint = results_graph.value(
                result_node, SH.sourceConstraintComponent
            )
            value = results_graph.value(result_node, SH.value)

            violation = ViolationDetail(
                focus_node=str(focus_node) if focus_node else "unknown",
                result_path=str(result_path) if result_path else None,
                message=str(message) if message else "Validation failed",
                severity="Violation",
                source_constraint_component=(
                    str(constraint) if constraint else None
                ),
                value=str(value) if value else None,
            )
            violations.append(violation)

        logger.debug(f"Parsed {len(violations)} violations from results")
        return violations

    def _compute_cache_key(
        self, data_graph: Graph, shapes_graph: Graph
    ) -> str:
        """Compute cache key for validation result.

        Args:
            data_graph: The data graph
            shapes_graph: The shapes graph

        Returns:
            Cache key as hex string
        """
        data_hash = hashlib.sha256(
            data_graph.serialize(format="turtle").encode()
        ).hexdigest()
        shapes_hash = hashlib.sha256(
            shapes_graph.serialize(format="turtle").encode()
        ).hexdigest()

        return f"{data_hash}:{shapes_hash}"

    def clear_cache(self) -> None:
        """Clear the validation cache."""
        self._cache.clear()
        logger.info("Validation cache cleared")

    def get_cache_stats(self) -> Dict:
        """Get cache statistics.

        Returns:
            Dictionary with cache size and other stats
        """
        return {
            "enabled": self.enable_caching,
            "size": len(self._cache),
        }
