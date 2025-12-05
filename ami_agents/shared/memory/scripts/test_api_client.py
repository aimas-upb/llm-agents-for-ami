"""Client test script for the simplified FastAPI interface.

This script tests all the API routes:
1. GET /signifiers - List all signifiers
2. POST /signifiers - Create signifier from RDF
3. DELETE /signifiers - Delete all signifiers
4. GET /signifiers/match - Match query with intent and context
"""

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class APITestClient:
    """Client for testing the signifier API."""

    def __init__(self, base_url: str = "http://localhost:8000"):
        """Initialize the test client.

        Args:
            base_url: Base URL of the API server
        """
        self.base_url = base_url
        self.session = requests.Session()

    def print_section(self, title: str) -> None:
        """Print a section header.

        Args:
            title: Section title
        """
        print("\n" + "=" * 80)
        print(f" {title}")
        print("=" * 80)

    def test_health_check(self) -> bool:
        """Test the health check endpoint.

        Returns:
            True if API is healthy
        """
        self.print_section("TEST 0: Health Check")

        try:
            response = self.session.get(f"{self.base_url}/health")
            response.raise_for_status()

            data = response.json()
            logger.info(f"Health check: {data}")
            print(f"Status: {data.get('status')}")
            print(f"Version: {data.get('version')}")

            return data.get("status") == "healthy"

        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False

    def test_delete_all_signifiers(self) -> bool:
        """Test DELETE /signifiers - Delete all signifiers.

        Returns:
            True if test passed
        """
        self.print_section("TEST 1: DELETE /signifiers - Clear Memory")

        try:
            response = self.session.delete(f"{self.base_url}/signifiers")
            response.raise_for_status()

            data = response.json()
            logger.info(f"Delete all response: {data}")
            print(f"Success: {data.get('success')}")
            print(f"Message: {data.get('message')}")
            print(f"Deleted count: {data.get('deleted_count')}")

            return data.get("success") is True

        except Exception as e:
            logger.error(f"Delete all test failed: {e}")
            return False

    def test_list_signifiers_empty(self) -> bool:
        """Test GET /signifiers when storage is empty.

        Returns:
            True if test passed
        """
        self.print_section("TEST 2: GET /signifiers - List Empty Memory")

        try:
            response = self.session.get(f"{self.base_url}/signifiers")
            response.raise_for_status()

            data = response.json()
            logger.info(f"List signifiers (empty): {data}")
            print(f"Total: {data.get('total')}")
            print(f"Signifiers: {len(data.get('signifiers', []))}")

            return data.get("total") == 0

        except Exception as e:
            logger.error(f"List empty signifiers test failed: {e}")
            return False

    def test_create_signifier(self, ttl_file: Path) -> bool:
        """Test POST /signifiers - Create signifier from RDF.

        Args:
            ttl_file: Path to TTL file

        Returns:
            True if test passed
        """
        self.print_section(f"TEST 3: POST /signifiers - Create from {ttl_file.name}")

        try:
            with open(ttl_file, "r", encoding="utf-8") as f:
                rdf_data = f.read()

            payload = {"rdf_data": rdf_data}

            response = self.session.post(
                f"{self.base_url}/signifiers",
                json=payload
            )
            response.raise_for_status()

            data = response.json()
            logger.info(f"Create signifier response: {data}")
            print(f"Signifier ID: {data.get('signifier_id')}")
            print(f"Message: {data.get('message')}")

            return "signifier_id" in data

        except Exception as e:
            logger.error(f"Create signifier test failed: {e}")
            if hasattr(e, 'response'):
                logger.error(f"Response: {e.response.text}")
            return False

    def test_list_signifiers(self, expected_count: int = None) -> bool:
        """Test GET /signifiers - List all signifiers.

        Args:
            expected_count: Expected number of signifiers

        Returns:
            True if test passed
        """
        self.print_section(f"TEST 4: GET /signifiers - List All (expect {expected_count})")

        try:
            response = self.session.get(f"{self.base_url}/signifiers")
            response.raise_for_status()

            data = response.json()
            signifiers = data.get("signifiers", [])

            logger.info(f"List signifiers: found {len(signifiers)}")
            print(f"Total: {data.get('total')}")
            print(f"\nSignifiers:")

            for sig in signifiers:
                print(f"  - ID: {sig.get('signifier_id')}")
                print(f"    Version: {sig.get('version')}")
                print(f"    Status: {sig.get('status')}")
                print(f"    Intent: {sig.get('intent')}")
                print(f"    Affordance URI: {sig.get('affordance_uri')}")
                print()

            if expected_count is not None:
                return len(signifiers) == expected_count

            return True

        except Exception as e:
            logger.error(f"List signifiers test failed: {e}")
            return False

    def test_match_signifiers(self, query: Dict[str, Any]) -> bool:
        """Test GET /signifiers/match - Match query.

        Args:
            query: Query with intent and context

        Returns:
            True if test passed
        """
        query_id = query.get("query_id", "unknown")
        intent = query.get("intent", "")
        context = query.get("context", {})

        self.print_section(f"TEST 5: GET /signifiers/match - Query {query_id}")

        print(f"Description: {query.get('description', 'N/A')}")
        print(f"Intent: {intent}")
        print(f"Context: {json.dumps(context, indent=2)}")

        try:
            params = {
                "intent": intent,
            }

            if context:
                params["context"] = json.dumps(context)

            response = self.session.get(
                f"{self.base_url}/signifiers/match",
                params=params
            )
            response.raise_for_status()

            data = response.json()
            matches = data.get("matches", [])
            final_matches = data.get("final_matches", [])

            logger.info(f"Match results: {len(matches)} intent matches, {len(final_matches)} final matches")

            print(f"\nTotal signifiers: {data.get('total_signifiers')}")
            print(f"Intent matches: {len(matches)}")
            print(f"Final matches (passed SHACL): {len(final_matches)}")

            print("\nDetailed matches:")
            for match in matches:
                print(f"  - Signifier: {match.get('signifier_id')}")
                print(f"    Intent similarity: {match.get('intent_similarity'):.4f}")
                print(f"    SHACL conforms: {match.get('shacl_conforms')}")
                if match.get('shacl_violations'):
                    print(f"    Violations:")
                    for violation in match.get('shacl_violations'):
                        print(f"      - {violation}")
                print()

            print(f"\nFinal matching signifiers: {final_matches}")

            return True

        except Exception as e:
            logger.error(f"Match test failed: {e}")
            if hasattr(e, 'response'):
                logger.error(f"Response: {e.response.text}")
            return False

    def run_all_tests(self, scenario_path: Path) -> None:
        """Run all API tests using a scenario.

        Args:
            scenario_path: Path to test scenario folder
        """
        print("\n" + "=" * 80)
        print(" API CLIENT TEST SUITE")
        print(f" Base URL: {self.base_url}")
        print(f" Scenario: {scenario_path}")
        print("=" * 80)

        signifiers_dir = scenario_path / "signifiers"
        queries_file = scenario_path / "queries.json"

        if not signifiers_dir.exists():
            logger.error(f"Signifiers directory not found: {signifiers_dir}")
            sys.exit(1)

        if not queries_file.exists():
            logger.error(f"Queries file not found: {queries_file}")
            sys.exit(1)

        results = {}

        results["health_check"] = self.test_health_check()
        if not results["health_check"]:
            logger.error("Health check failed. Is the API server running?")
            sys.exit(1)

        results["delete_all"] = self.test_delete_all_signifiers()

        results["list_empty"] = self.test_list_signifiers_empty()

        signifier_files = sorted(signifiers_dir.glob("*.ttl"))
        logger.info(f"Found {len(signifier_files)} signifier files")

        create_results = []
        for ttl_file in signifier_files:
            result = self.test_create_signifier(ttl_file)
            create_results.append(result)

        results["create_signifiers"] = all(create_results)

        results["list_all"] = self.test_list_signifiers(expected_count=len(signifier_files))

        with open(queries_file, "r", encoding="utf-8") as f:
            queries = json.load(f)

        match_results = []
        for query_key, query_data in queries.items():
            result = self.test_match_signifiers(query_data)
            match_results.append(result)

        results["match_queries"] = all(match_results)

        self.print_section("TEST SUMMARY")
        print(f"Health Check: {'PASS' if results['health_check'] else 'FAIL'}")
        print(f"Delete All Signifiers: {'PASS' if results['delete_all'] else 'FAIL'}")
        print(f"List Empty Signifiers: {'PASS' if results['list_empty'] else 'FAIL'}")
        print(f"Create Signifiers ({len(signifier_files)}): {'PASS' if results['create_signifiers'] else 'FAIL'}")
        print(f"List All Signifiers: {'PASS' if results['list_all'] else 'FAIL'}")
        print(f"Match Queries ({len(queries)}): {'PASS' if results['match_queries'] else 'FAIL'}")

        all_passed = all(results.values())
        print("\n" + "=" * 80)
        if all_passed:
            print(" ALL TESTS PASSED")
        else:
            print(" SOME TESTS FAILED")
        print("=" * 80 + "\n")

        sys.exit(0 if all_passed else 1)


def main():
    """Main entry point for the test script."""
    if len(sys.argv) < 2:
        print("Usage: python test_api_client.py <scenario_folder> [api_url]")
        print("Example: python test_api_client.py test_scenario/3 http://localhost:8000")
        sys.exit(1)

    scenario_path = Path(sys.argv[1])
    api_url = sys.argv[2] if len(sys.argv) > 2 else "http://localhost:8000"

    if not scenario_path.exists():
        logger.error(f"Scenario path does not exist: {scenario_path}")
        sys.exit(1)

    client = APITestClient(base_url=api_url)
    client.run_all_tests(scenario_path)


if __name__ == "__main__":
    main()
