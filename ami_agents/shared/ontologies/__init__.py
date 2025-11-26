"""
Ontology management for AMI agents.

This module provides utilities for loading and working with OWL ontologies
used in the AmI HMAS system, including HMAS and CASHMERE ontologies.
"""

from .loader import OntologyLoader, get_hmas_ontology, get_cashmere_ontology

__all__ = ["OntologyLoader", "get_hmas_ontology", "get_cashmere_ontology"]
