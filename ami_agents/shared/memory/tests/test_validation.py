"""Tests for Phase 2 validation modules.

This module tests SHACL validation, context graph building,
and authoring validation.
"""

import pytest
from rdflib import Graph

from src.models.signifier import GoalSignifier
from src.validation import (
    AuthoringValidationError,
    AuthoringValidator,
    ContextGraphBuilder,
    SHACLValidator,
)


class TestContextGraphBuilder:
    """Tests for Context Graph Builder."""

    def test_build_from_kv_nested_dict(self):
        """Test building graph from nested dictionary."""
        builder = ContextGraphBuilder()

        context = {
            "http://example.org/artifacts/sensor1": {
                "http://example.org/LightSensor#hasLuminosityLevel": 15000
            }
        }

        graph, features = builder.build_from_kv(context)

        assert len(graph) == 1
        assert len(features) == 1
        assert (
            "http://example.org/artifacts/sensor1",
            "http://example.org/LightSensor#hasLuminosityLevel",
        ) in features

    def test_build_from_flat_dict(self):
        """Test building graph from flat dictionary with :: separator."""
        builder = ContextGraphBuilder()

        context = {
            "http://example.org/artifacts/sensor1::http://example.org/LightSensor#hasLuminosityLevel": 15000
        }

        graph, features = builder.build_from_flat_dict(context)

        assert len(graph) == 1
        assert len(features) == 1

    def test_add_type_information(self):
        """Test adding RDF type information to artifacts."""
        builder = ContextGraphBuilder()

        context = {
            "http://example.org/artifacts/sensor1": {
                "http://example.org/LightSensor#hasLuminosityLevel": 15000
            }
        }

        graph, _ = builder.build_from_kv(context)

        types = {
            "http://example.org/artifacts/sensor1": "http://example.org/LightSensor"
        }

        graph = builder.add_type_information(graph, types)

        assert len(graph) == 2

    def test_normalize_context_graph_input(self):
        """Test normalizing context when input is already a graph."""
        builder = ContextGraphBuilder()
        input_graph = Graph()

        result_graph, features = builder.normalize_context(input_graph)

        assert result_graph == input_graph


class TestSHACLValidator:
    """Tests for SHACL Validator."""

    def test_parse_shapes(self):
        """Test parsing SHACL shapes from Turtle."""
        validator = SHACLValidator(enable_caching=False)

        shapes = """
        @prefix sh: <http://www.w3.org/ns/shacl#> .
        @prefix ex: <http://example.org/> .
        @prefix xsd: <http://www.w3.org/2001/XMLSchema#> .

        ex:TestShape a sh:NodeShape ;
            sh:targetNode ex:artifact1 ;
            sh:property [
                sh:path ex:prop1 ;
                sh:datatype xsd:integer ;
                sh:minInclusive 100 ;
            ] .
        """

        shapes_graph = validator.parse_shapes(shapes)

        assert len(shapes_graph) > 0

    def test_validate_conforming(self):
        """Test validation with conforming data."""
        validator = SHACLValidator(enable_caching=False)

        shapes = """
        @prefix sh: <http://www.w3.org/ns/shacl#> .
        @prefix ex: <http://example.org/> .
        @prefix xsd: <http://www.w3.org/2001/XMLSchema#> .

        ex:TestShape a sh:NodeShape ;
            sh:targetNode ex:artifact1 ;
            sh:property [
                sh:path ex:prop1 ;
                sh:datatype xsd:integer ;
                sh:minInclusive 100
            ] .
        """

        data = """
        @prefix ex: <http://example.org/> .
        @prefix xsd: <http://www.w3.org/2001/XMLSchema#> .

        ex:artifact1 ex:prop1 150 .
        """

        shapes_graph = validator.parse_shapes(shapes)
        data_graph = Graph()
        data_graph.parse(data=data, format="turtle")

        result = validator.validate(data_graph, shapes_graph, use_cache=False)

        assert result.conforms is True
        assert len(result.violations) == 0

    def test_validate_non_conforming(self):
        """Test validation with non-conforming data."""
        validator = SHACLValidator(enable_caching=False)

        shapes = """
        @prefix sh: <http://www.w3.org/ns/shacl#> .
        @prefix ex: <http://example.org/> .
        @prefix xsd: <http://www.w3.org/2001/XMLSchema#> .

        ex:TestShape a sh:NodeShape ;
            sh:targetNode ex:artifact1 ;
            sh:property [
                sh:path ex:prop1 ;
                sh:datatype xsd:integer ;
                sh:minInclusive 100
            ] .
        """

        data = """
        @prefix ex: <http://example.org/> .
        @prefix xsd: <http://www.w3.org/2001/XMLSchema#> .

        ex:artifact1 ex:prop1 50 .
        """

        shapes_graph = validator.parse_shapes(shapes)
        data_graph = Graph()
        data_graph.parse(data=data, format="turtle")

        result = validator.validate(data_graph, shapes_graph, use_cache=False)

        assert result.conforms is False
        assert len(result.violations) > 0

    def test_validation_caching(self):
        """Test validation result caching."""
        validator = SHACLValidator(enable_caching=True)

        shapes = """
        @prefix sh: <http://www.w3.org/ns/shacl#> .
        @prefix ex: <http://example.org/> .
        @prefix xsd: <http://www.w3.org/2001/XMLSchema#> .

        ex:TestShape a sh:NodeShape ;
            sh:targetNode ex:artifact1 ;
            sh:property [
                sh:path ex:prop1 ;
                sh:datatype xsd:integer
            ] .
        """

        data = """
        @prefix ex: <http://example.org/> .

        ex:artifact1 ex:prop1 100 .
        """

        shapes_graph = validator.parse_shapes(shapes)
        data_graph = Graph()
        data_graph.parse(data=data, format="turtle")

        result1 = validator.validate(data_graph, shapes_graph, use_cache=True)
        result2 = validator.validate(data_graph, shapes_graph, use_cache=True)

        assert result1.conforms == result2.conforms
        stats = validator.get_cache_stats()
        assert stats["size"] > 0


class TestAuthoringValidator:
    """The authoring validator, over the goal signifier record."""

    TREE = {"type": "action", "action_url": "http://h/a", "parameters": {"value": 1}}

    def _explicit(self, **fields):
        base = {"signifier_id": "sig-1", "kind": "explicit",
                "key": {"artifact_name": "ac"}, "intent_text": "cool it", "tree": self.TREE}
        return GoalSignifier(**{**base, **fields})

    def test_validate_valid_signifier(self):
        validator = AuthoringValidator(strict_mode=False)
        assert validator.validate_signifier(self._explicit(), enable_shacl_check=False) == []

    def test_validate_missing_fields(self):
        validator = AuthoringValidator(strict_mode=False)
        errors = validator.validate_signifier(self._explicit(tree=None),
                                              enable_shacl_check=False)
        assert any("tree" in err for err in errors)
        ambiguous = GoalSignifier(signifier_id="sig-2", kind="ambiguous", key={"v": 1})
        errors = validator.validate_signifier(ambiguous, enable_shacl_check=False)
        assert any("context" in err for err in errors)
        assert any("used" in err for err in errors)

    def test_validate_invalid_shacl_shapes(self):
        validator = AuthoringValidator(strict_mode=False)
        invalid_shapes = """
        @prefix sh: <http://www.w3.org/ns/shacl#> .
        invalid turtle syntax
        """
        errors = validator.validate_signifier(self._explicit(shacl_shapes=invalid_shapes),
                                              enable_shacl_check=True)
        assert any("parse" in err.lower() for err in errors)

    def test_validate_and_raise_strict_mode(self):
        validator = AuthoringValidator(strict_mode=True)
        with pytest.raises(AuthoringValidationError):
            validator.validate_signifier(self._explicit(tree=None))


class TestIntegration:
    """Integration tests for validation workflow."""

    def test_full_validation_workflow(self):
        """Test complete validation workflow from context to SHACL validation."""
        builder = ContextGraphBuilder()
        validator = SHACLValidator(enable_caching=False)

        context = {
            "http://example.org/precis/workspaces/lab308/artifacts/external_light_sensing308": {
                "http://example.org/LightSensor#hasLuminosityLevel": 15000
            }
        }

        shapes = """
        @prefix sh: <http://www.w3.org/ns/shacl#> .
        @prefix xsd: <http://www.w3.org/2001/XMLSchema#> .

        <#LightShape> a sh:NodeShape ;
            sh:targetNode <http://example.org/precis/workspaces/lab308/artifacts/external_light_sensing308> ;
            sh:property [
                sh:path <http://example.org/LightSensor#hasLuminosityLevel> ;
                sh:datatype xsd:integer ;
                sh:minInclusive 10000 ;
            ] .
        """

        context_graph, _ = builder.build_from_kv(context)

        result = validator.validate_signifier_context(context_graph, shapes)

        assert result.conforms is True

    def test_validation_with_violation(self):
        """Test validation that should produce violations."""
        builder = ContextGraphBuilder()
        validator = SHACLValidator(enable_caching=False)

        context = {
            "http://example.org/precis/workspaces/lab308/artifacts/external_light_sensing308": {
                "http://example.org/LightSensor#hasLuminosityLevel": 5000
            }
        }

        shapes = """
        @prefix sh: <http://www.w3.org/ns/shacl#> .
        @prefix xsd: <http://www.w3.org/2001/XMLSchema#> .

        <#LightShape> a sh:NodeShape ;
            sh:targetNode <http://example.org/precis/workspaces/lab308/artifacts/external_light_sensing308> ;
            sh:property [
                sh:path <http://example.org/LightSensor#hasLuminosityLevel> ;
                sh:datatype xsd:integer ;
                sh:minInclusive 10000 ;
            ] .
        """

        context_graph, _ = builder.build_from_kv(context)

        result = validator.validate_signifier_context(context_graph, shapes)

        assert result.conforms is False
        assert len(result.violations) > 0
