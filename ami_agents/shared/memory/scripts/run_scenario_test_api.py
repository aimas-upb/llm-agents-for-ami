"""Test scenario runner script using FastAPI interface.

This script runs a complete test scenario by:
1. Starting the API server
2. Clearing the memory store via API
3. Loading signifiers from the scenario folder via API
4. Running queries from queries.json via API
5. Saving results to JSON file
"""

import json
import logging
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import requests

logging.basicConfig(
    level=logging.CRITICAL,
    format="%(message)s"
)

logger = logging.getLogger(__name__)


class APIScenarioTestRunner:
    """Runner for executing test scenarios using API interface."""

    def __init__(self, scenario_path: str, api_url: str = "http://localhost:8000"):
        """Initialize the API scenario test runner.

        Args:
            scenario_path: Path to scenario folder (e.g., test_scenario/1)
            api_url: Base URL of the API server
        """
        self.scenario_path = Path(scenario_path)
        self.api_url = api_url
        self.results_dir = self.scenario_path / "test_log"

        if not self.scenario_path.exists():
            raise ValueError(f"Scenario path does not exist: {scenario_path}")

        self.signifiers_dir = self.scenario_path / "signifiers"
        self.queries_file = self.scenario_path / "queries.json"

        if not self.signifiers_dir.exists():
            raise ValueError(f"Signifiers directory not found: {self.signifiers_dir}")
        if not self.queries_file.exists():
            raise ValueError(f"Queries file not found: {self.queries_file}")

        self.results_dir.mkdir(exist_ok=True)

        self.session = requests.Session()
        self.api_process = None

    def print_header(self, text: str) -> None:
        """Print a section header.

        Args:
            text: Header text
        """
        print(f"\n{'=' * 80}")
        print(f"{text}")
        print('=' * 80)

    def start_api_server(self) -> bool:
        """Start the FastAPI server in the background.

        Returns:
            True if server started successfully
        """
        self.print_header("STEP 0: Starting API Server")

        try:
            response = self.session.get(f"{self.api_url}/health", timeout=2)
            if response.status_code == 200:
                print("API server already running")
                return True
        except requests.exceptions.RequestException:
            pass

        print("Starting API server...")

        venv_python = Path("venv/Scripts/python.exe")
        if not venv_python.exists():
            venv_python = Path("venv/bin/python")

        self.api_process = subprocess.Popen(
            [str(venv_python.absolute()), "-m", "uvicorn", "src.api.main:app",
             "--host", "0.0.0.0", "--port", "8000"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            cwd=Path.cwd()
        )

        max_wait = 15
        for i in range(max_wait):
            time.sleep(1)
            try:
                response = self.session.get(f"{self.api_url}/health", timeout=2)
                if response.status_code == 200:
                    print(f"API server started successfully (waited {i+1}s)")
                    return True
            except requests.exceptions.RequestException:
                continue

        print("ERROR: API server failed to start")
        return False

    def stop_api_server(self) -> None:
        """Stop the FastAPI server if it was started by this script."""
        if self.api_process:
            self.api_process.terminate()
            self.api_process.wait(timeout=5)
            print("API server stopped")

    def clear_storage(self) -> None:
        """Clear the storage via API."""
        self.print_header("STEP 1: Clearing Storage")

        try:
            response = self.session.delete(f"{self.api_url}/signifiers")
            response.raise_for_status()

            data = response.json()
            deleted_count = data.get("deleted_count", 0)
            print(f"Deleted {deleted_count} signifiers")
            print("Storage cleared")

        except Exception as e:
            print(f"ERROR: Failed to clear storage: {e}")
            raise

    def load_signifiers(self) -> List[str]:
        """Load all signifiers from the scenario signifiers folder via API.

        Returns:
            List of loaded signifier IDs
        """
        self.print_header("STEP 2: Loading Signifiers")

        signifier_files = sorted(self.signifiers_dir.glob("*.ttl"))
        print(f"Found {len(signifier_files)} signifier files\n")

        loaded_ids = []

        for idx, file_path in enumerate(signifier_files, 1):
            print(f"Loading {idx}/{len(signifier_files)}: {file_path.name}")

            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    rdf_data = f.read()

                response = self.session.post(
                    f"{self.api_url}/signifiers",
                    json={"rdf_data": rdf_data}
                )
                response.raise_for_status()

                data = response.json()
                signifier_id = data.get("signifier_id")
                print(f"  ID: {signifier_id}")

                response = self.session.get(f"{self.api_url}/signifiers")
                response.raise_for_status()
                all_signifiers = response.json().get("signifiers", [])

                matching_signifier = next(
                    (s for s in all_signifiers if s["signifier_id"] == signifier_id),
                    None
                )

                if matching_signifier:
                    print(f"  Intent: {matching_signifier.get('intent', 'N/A')}")

                loaded_ids.append(signifier_id)

            except Exception as e:
                print(f"  ERROR: {e}")

        print(f"\nLoaded {len(loaded_ids)} signifiers")
        return loaded_ids

    def run_queries(self) -> Dict[str, Any]:
        """Run all queries from queries.json via API.

        Returns:
            Dictionary with query results
        """
        self.print_header("STEP 3: Running Queries")

        with open(self.queries_file, "r", encoding="utf-8") as f:
            queries = json.load(f)

        print(f"Loaded {len(queries)} queries\n")

        results = {}

        for query_key, query_data in queries.items():
            result = self._run_single_query(query_key, query_data)
            results[query_key] = result

        return results

    def _run_single_query(self, query_key: str, query_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run a single query via API and return results.

        Args:
            query_key: Query identifier
            query_data: Query configuration

        Returns:
            Dictionary with query execution results
        """
        query_id = query_data.get("query_id", query_key)
        description = query_data.get("description", "")
        intent = query_data.get("intent", "")
        context = query_data.get("context", {})

        print(f"\n{'-' * 80}")
        print(f"Query {query_key}: {query_id}")
        print(f"Description: {description}")
        print(f"Intent: {intent}")
        print(f"{'-' * 80}")

        result = {
            "query_id": query_id,
            "description": description,
            "intent": intent,
            "context": context,
            "matches": []
        }

        try:
            params = {
                "intent": intent,
            }

            if context:
                params["context"] = json.dumps(context)

            response = self.session.get(
                f"{self.api_url}/signifiers/match",
                params=params
            )
            response.raise_for_status()

            data = response.json()
            matches = data.get("matches", [])
            final_matches = data.get("final_matches", [])

            print(f"Total signifiers in registry: {data.get('total_signifiers', 0)}")
            print(f"\nPhase 1: Intent Matching")
            print(f"Found {len(matches)} intent matches")

            formatted_matches = []
            for match in matches:
                signifier_id = match.get("signifier_id")
                intent_similarity = match.get("intent_similarity")
                shacl_conforms = match.get("shacl_conforms")
                shacl_violations = match.get("shacl_violations", [])

                print(f"\n  Signifier: {signifier_id}")
                print(f"    Intent similarity: {intent_similarity:.4f}")

                if shacl_violations:
                    print(f"    SHACL validation: FAIL")
                    for violation in shacl_violations:
                        print(f"      - {violation}")
                else:
                    if shacl_conforms:
                        print(f"    SHACL validation: PASS")
                    else:
                        print(f"    SHACL validation: N/A (no constraints)")

                formatted_match = {
                    "signifier_id": signifier_id,
                    "intent_similarity": intent_similarity,
                    "shacl_conforms": shacl_conforms,
                    "shacl_violations": shacl_violations
                }
                formatted_matches.append(formatted_match)

            print(f"\nFinal Matches: {len(final_matches)}")
            for signifier_id in final_matches:
                matching_entry = next(
                    (m for m in formatted_matches if m["signifier_id"] == signifier_id),
                    None
                )
                if matching_entry:
                    print(f"  - {signifier_id} (similarity: {matching_entry['intent_similarity']:.4f})")

            result["matches"] = formatted_matches
            result["final_matches"] = final_matches

        except Exception as e:
            print(f"ERROR: {e}")
            result["error"] = str(e)

        return result

    def save_results(self, results: Dict[str, Any]) -> str:
        """Save test results to JSON file.

        Args:
            results: Complete test execution results

        Returns:
            Path to results file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = self.results_dir / f"results_{timestamp}.json"

        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

        return str(results_file)

    def run(self, start_server: bool = False) -> None:
        """Execute the complete test scenario.

        Args:
            start_server: Whether to start API server (default: False, assumes already running)
        """
        start_time = datetime.now()

        print("\n" + "=" * 80)
        print("API SCENARIO TEST RUNNER")
        print(f"Scenario: {self.scenario_path}")
        print(f"API URL: {self.api_url}")
        print(f"Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)

        try:
            if start_server:
                if not self.start_api_server():
                    sys.exit(1)

            response = self.session.get(f"{self.api_url}/health", timeout=5)
            if response.status_code != 200:
                print("ERROR: API server is not responding")
                sys.exit(1)

            self.clear_storage()
            loaded_ids = self.load_signifiers()
            results = self.run_queries()
            results_file = self.save_results(results)

            end_time = datetime.now()
            duration = end_time - start_time

            self.print_header("TEST COMPLETE")
            print(f"Duration: {duration.total_seconds():.2f} seconds")
            print(f"Signifiers loaded: {len(loaded_ids)}")
            print(f"Queries executed: {len(results)}")
            print(f"Results saved to: {results_file}")
            print()

        except Exception as e:
            print(f"\nERROR: Test scenario failed: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

        finally:
            if start_server:
                self.stop_api_server()


def main():
    """Main entry point for the script."""
    if len(sys.argv) < 2:
        print("Usage: python run_scenario_test_api.py <scenario_folder> [--start-server] [--api-url <url>]")
        print("Example: python run_scenario_test_api.py test_scenario/3")
        print("Example: python run_scenario_test_api.py test_scenario/3 --start-server")
        print("Example: python run_scenario_test_api.py test_scenario/3 --api-url http://localhost:8000")
        sys.exit(1)

    scenario_path = sys.argv[1]
    start_server = "--start-server" in sys.argv
    api_url = "http://localhost:8000"

    if "--api-url" in sys.argv:
        try:
            idx = sys.argv.index("--api-url")
            api_url = sys.argv[idx + 1]
        except (IndexError, ValueError):
            print("ERROR: --api-url requires a URL argument")
            sys.exit(1)

    runner = APIScenarioTestRunner(scenario_path, api_url=api_url)
    runner.run(start_server=start_server)


if __name__ == "__main__":
    main()
