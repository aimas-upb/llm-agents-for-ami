"""Test Phase 1 - Storage implementation with example signifiers.

This test module validates the Phase 1 storage functionality
using the three example signifiers.
"""

import json
import logging
from pathlib import Path

import pytest

from src.models.signifier import Signifier, SignifierStatus
from src.storage.registry import SignifierRegistry

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@pytest.fixture
def test_storage_dir(tmp_path):
    """Create temporary storage directory.

    Args:
        tmp_path: Pytest temporary directory fixture

    Returns:
        Path to test storage directory
    """
    return str(tmp_path / "test_storage")


@pytest.fixture
def registry(test_storage_dir):
    """Create SignifierRegistry instance.

    Args:
        test_storage_dir: Test storage directory path

    Returns:
        SignifierRegistry instance
    """
    return SignifierRegistry(storage_dir=test_storage_dir)


@pytest.fixture
def signifier_files():
    """Get paths to example signifier files.

    Returns:
        Dictionary mapping signifier names to file paths
    """
    base_path = Path(__file__).parent.parent / "signifiers"
    return {
        "raise_blinds": base_path / "raise-blinds-signifier.ttl",
        "turn_light_on": base_path / "turn-light-on-signifier.ttl",
        "lower_blinds": base_path / "lower-blinds-signifier.ttl",
    }


def test_create_signifier_from_rdf(registry, signifier_files):
    """Test creating signifiers from RDF files.

    Args:
        registry: SignifierRegistry instance
        signifier_files: Dictionary of signifier file paths
    """
    for name, file_path in signifier_files.items():
        logger.info(f"Testing {name} signifier from {file_path}")

        with open(file_path, "r", encoding="utf-8") as f:
            rdf_data = f.read()

        signifier = registry.create_from_rdf(rdf_data, format="turtle")

        assert signifier is not None
        assert signifier.signifier_id is not None
        assert signifier.version == 1
        assert signifier.status == SignifierStatus.ACTIVE
        assert signifier.intent is not None
        assert signifier.intent.nl_text is not None
        assert signifier.affordance_uri is not None

        logger.info(f"Created signifier: {signifier.signifier_id}")
        logger.info(f"  Intent: {signifier.intent.nl_text}")
        logger.info(f"  Affordance: {signifier.affordance_uri}")
        logger.info(f"  Conditions: {len(signifier.context.structured_conditions)}")


def test_retrieve_signifier(registry, signifier_files):
    """Test retrieving signifiers after creation.

    Args:
        registry: SignifierRegistry instance
        signifier_files: Dictionary of signifier file paths
    """
    created_ids = []

    for name, file_path in signifier_files.items():
        with open(file_path, "r", encoding="utf-8") as f:
            rdf_data = f.read()
        signifier = registry.create_from_rdf(rdf_data, format="turtle")
        created_ids.append(signifier.signifier_id)

    for signifier_id in created_ids:
        retrieved = registry.get(signifier_id)
        assert retrieved is not None
        assert retrieved.signifier_id == signifier_id
        logger.info(f"Retrieved signifier: {signifier_id}")


def test_list_signifiers(registry, signifier_files):
    """Test listing signifiers.

    Args:
        registry: SignifierRegistry instance
        signifier_files: Dictionary of signifier file paths
    """
    for name, file_path in signifier_files.items():
        with open(file_path, "r", encoding="utf-8") as f:
            rdf_data = f.read()
        registry.create_from_rdf(rdf_data, format="turtle")

    all_signifiers = registry.list_signifiers()
    assert len(all_signifiers) == 3
    logger.info(f"Listed {len(all_signifiers)} signifiers")

    active_signifiers = registry.list_signifiers(status=SignifierStatus.ACTIVE)
    assert len(active_signifiers) == 3


def test_update_signifier_status(registry, signifier_files):
    """Test updating signifier status.

    Args:
        registry: SignifierRegistry instance
        signifier_files: Dictionary of signifier file paths
    """
    file_path = signifier_files["raise_blinds"]
    with open(file_path, "r", encoding="utf-8") as f:
        rdf_data = f.read()
    signifier = registry.create_from_rdf(rdf_data, format="turtle")

    updated = registry.update_status(
        signifier.signifier_id, SignifierStatus.DEPRECATED
    )
    assert updated is not None
    assert updated.status == SignifierStatus.DEPRECATED
    logger.info(f"Updated status to DEPRECATED for {signifier.signifier_id}")


def test_property_index(registry, signifier_files):
    """Test property index functionality.

    Args:
        registry: SignifierRegistry instance
        signifier_files: Dictionary of signifier file paths
    """
    for name, file_path in signifier_files.items():
        with open(file_path, "r", encoding="utf-8") as f:
            rdf_data = f.read()
        registry.create_from_rdf(rdf_data, format="turtle")

    results = registry.find_by_property(
        artifact_uri="http://example.org/precis/workspaces/lab308/artifacts/external_light_sensing308",
        property_uri="http://example.org/LightSensor#hasLuminosityLevel",
    )

    assert len(results) >= 2
    logger.info(
        f"Found {len(results)} signifiers for external light sensor property"
    )


def test_rdf_round_trip(registry, signifier_files):
    """Test RDF storage and retrieval.

    Args:
        registry: SignifierRegistry instance
        signifier_files: Dictionary of signifier file paths
    """
    file_path = signifier_files["raise_blinds"]
    with open(file_path, "r", encoding="utf-8") as f:
        original_rdf = f.read()

    signifier = registry.create_from_rdf(original_rdf, format="turtle")

    retrieved_rdf = registry.get_rdf_representation(signifier.signifier_id)
    assert retrieved_rdf is not None
    logger.info(f"Retrieved RDF for {signifier.signifier_id}")


def test_delete_signifier(registry, signifier_files):
    """Test deleting signifiers.

    Args:
        registry: SignifierRegistry instance
        signifier_files: Dictionary of signifier file paths
    """
    file_path = signifier_files["raise_blinds"]
    with open(file_path, "r", encoding="utf-8") as f:
        rdf_data = f.read()
    signifier = registry.create_from_rdf(rdf_data, format="turtle")

    success = registry.delete(signifier.signifier_id)
    assert success is True

    retrieved = registry.get(signifier.signifier_id)
    assert retrieved is None
    logger.info(f"Deleted signifier: {signifier.signifier_id}")


def test_versioning(registry, signifier_files):
    """Test signifier versioning.

    Args:
        registry: SignifierRegistry instance
        signifier_files: Dictionary of signifier file paths
    """
    file_path = signifier_files["raise_blinds"]
    with open(file_path, "r", encoding="utf-8") as f:
        rdf_data = f.read()
    signifier = registry.create_from_rdf(rdf_data, format="turtle")

    assert signifier.version == 1

    signifier.intent.nl_text = "updated intent"
    updated = registry.update(signifier, create_new_version=True)

    assert updated.version == 2
    logger.info(
        f"Created new version {updated.version} for {updated.signifier_id}"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
