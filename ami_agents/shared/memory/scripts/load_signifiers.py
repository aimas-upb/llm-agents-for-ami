"""Load the fixed example signifiers into the system.

This script loads the three corrected signifier files from the signifiers/
directory into the RD4 storage system.
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import get_settings, setup_logging
from src.storage.registry import SignifierRegistry

setup_logging(get_settings())
logger = logging.getLogger(__name__)


def load_fixed_signifiers():
    """Load the three fixed signifier files."""
    signifiers_dir = Path(__file__).parent.parent / "signifiers"

    registry = SignifierRegistry()

    signifier_files = [
        ("raise-blinds-signifier.ttl", "Raise Blinds"),
        ("turn-light-on-signifier.ttl", "Turn Light On"),
        ("lower-blinds-signifier.ttl", "Lower Blinds"),
    ]

    logger.info(f"Loading {len(signifier_files)} signifier files\n")

    loaded_count = 0
    for filename, name in signifier_files:
        file_path = signifiers_dir / filename

        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            continue

        logger.info(f"Loading: {name} ({filename})")

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                rdf_data = f.read()

            signifier = registry.create_from_rdf(rdf_data, format="turtle")

            logger.info(f"  Created: {signifier.signifier_id}")
            logger.info(f"  Intent: {signifier.intent.nl_text}")
            logger.info(f"  Affordance: {signifier.affordance_uri}")
            logger.info(
                f"  Conditions: {len(signifier.context.structured_conditions)}"
            )
            logger.info(f"  Version: {signifier.version}")
            logger.info("")

            loaded_count += 1

        except ValueError as e:
            if "already exists" in str(e):
                logger.warning(f"  Already exists, skipping")
            else:
                logger.error(f"  Failed to load: {e}")
        except Exception as e:
            logger.error(f"  Failed to load signifier: {e}")

    all_signifiers = registry.list_signifiers()
    logger.info(f"Total signifiers in system: {len(all_signifiers)}")
    logger.info(f"Successfully loaded: {loaded_count} signifiers")

    # Display summary
    logger.info("\n=== Signifier Summary ===")
    for signifier in all_signifiers:
        logger.info(f"  - {signifier.signifier_id}: {signifier.intent.nl_text}")

    # Test property index
    logger.info("\n=== Property Index Test ===")
    results = registry.find_by_property(
        artifact_uri="http://example.org/precis/workspaces/lab308/artifacts/external_light_sensing308",
        property_uri="http://example.org/LightSensor#hasLuminosityLevel",
    )
    logger.info(
        f"Signifiers with external light sensor property: {len(results)}"
    )
    for sig in results:
        logger.info(f"  - {sig.signifier_id}")


if __name__ == "__main__":
    load_fixed_signifiers()
