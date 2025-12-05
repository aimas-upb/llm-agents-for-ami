"""Test scenario runner script.

This script runs a complete test scenario by:
1. Clearing the memory store
2. Loading signifiers from the scenario folder
3. Running queries from queries.json
4. Saving results to JSON file
"""

import json
import logging
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import get_settings
from src.matching.registry import IntentMatcherRegistry
from src.storage.registry import SignifierRegistry
from src.validation.context_builder import ContextGraphBuilder
from src.validation.shacl_validator import SHACLValidator

import warnings
warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.CRITICAL,
    format="%(message)s"
)

for logger_name in ["sentence_transformers", "transformers", "torch", "tensorflow"]:
    logging.getLogger(logger_name).setLevel(logging.CRITICAL)


class ScenarioTestRunner:
    """Runner for executing test scenarios."""

    def __init__(self, scenario_path: str):
        """Initialize the scenario test runner.

        Args:
            scenario_path: Path to scenario folder (e.g., test_scenario/1)
        """
        self.scenario_path = Path(scenario_path)
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

        settings = get_settings()
        self.storage_dir = Path(settings.storage_dir)

        self.registry = None
        self.matcher_registry = None
        self.context_builder = None
        self.shacl_validator = None

    def print_header(self, text: str) -> None:
        """Print a section header.

        Args:
            text: Header text
        """
        print(f"\n{'=' * 80}")
        print(f"{text}")
        print('=' * 80)

    def clear_storage(self) -> None:
        """Clear the storage directory to ensure clean state."""
        self.print_header("STEP 1: Clearing Storage")

        if self.storage_dir.exists():
            for subdir in ["rdf", "json", "indexes"]:
                subdir_path = self.storage_dir / subdir
                if subdir_path.exists():
                    file_count = len(list(subdir_path.glob("*")))
                    if file_count > 0:
                        print(f"Removing {file_count} files from {subdir}/")
                        shutil.rmtree(subdir_path)
            print("Storage cleared")
        else:
            print("Storage directory does not exist yet")

    def initialize_components(self) -> None:
        """Initialize the system components."""
        self.print_header("STEP 2: Initializing Components")

        self.registry = SignifierRegistry(
            storage_dir=str(self.storage_dir),
            enable_authoring_validation=False
        )
        self.matcher_registry = IntentMatcherRegistry(default_version="v1")
        self.context_builder = ContextGraphBuilder()
        self.shacl_validator = SHACLValidator(enable_caching=False)

        print("All components initialized")

    def load_signifiers(self) -> List[str]:
        """Load all signifiers from the scenario signifiers folder.

        Returns:
            List of loaded signifier IDs
        """
        self.print_header("STEP 3: Loading Signifiers")

        signifier_files = sorted(self.signifiers_dir.glob("*.ttl"))
        print(f"Found {len(signifier_files)} signifier files\n")

        loaded_ids = []

        for idx, file_path in enumerate(signifier_files, 1):
            print(f"Loading {idx}/{len(signifier_files)}: {file_path.name}")

            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    rdf_data = f.read()

                signifier = self.registry.create_from_rdf(rdf_data, format="turtle")
                print(f"  ID: {signifier.signifier_id}")
                print(f"  Intent: {signifier.intent.nl_text}")
                loaded_ids.append(signifier.signifier_id)

            except Exception as e:
                print(f"  ERROR: {e}")

        print(f"\nLoaded {len(loaded_ids)} signifiers")
        return loaded_ids

    def run_queries(self) -> Dict[str, Any]:
        """Run all queries from queries.json.

        Returns:
            Dictionary with query results
        """
        self.print_header("STEP 4: Running Queries")

        with open(self.queries_file, "r", encoding="utf-8") as f:
            queries = json.load(f)

        print(f"Loaded {len(queries)} queries\n")

        results = {}

        for query_key, query_data in queries.items():
            result = self._run_single_query(query_key, query_data)
            results[query_key] = result

        return results

    def _run_single_query(self, query_key: str, query_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run a single query and return results.

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
            all_signifiers = self.registry.list_signifiers(limit=10000)
            signifier_dicts = [s.model_dump() for s in all_signifiers]
            print(f"Total signifiers in registry: {len(signifier_dicts)}")

            print(f"\nPhase 1: Intent Matching")
            match_results = self.matcher_registry.match(
                intent_query=intent,
                signifiers=signifier_dicts,
                k=10,
                version="v1"
            )

            context_graph, _ = self.context_builder.normalize_context(context)

            matches = []
            for match in match_results:
                signifier = self.registry.get(match.signifier_id)
                if not signifier:
                    continue

                print(f"\n  Signifier: {match.signifier_id}")
                print(f"    Intent similarity: {match.similarity:.4f}")

                shacl_result = {"conforms": True, "violations": []}

                if signifier.context.shacl_shapes:
                    validation = self.shacl_validator.validate_signifier_context(
                        context_graph,
                        signifier.context.shacl_shapes,
                        format="turtle"
                    )
                    shacl_result = {
                        "conforms": validation.conforms,
                        "violations": [v.message for v in validation.violations]
                    }
                    print(f"    SHACL validation: {'PASS' if validation.conforms else 'FAIL'}")
                    if validation.violations:
                        for v in validation.violations:
                            print(f"      - {v.message}")
                else:
                    print(f"    SHACL validation: N/A (no constraints)")

                match_info = {
                    "signifier_id": match.signifier_id,
                    "intent_similarity": round(match.similarity, 4),
                    "shacl_conforms": shacl_result["conforms"],
                    "shacl_violations": shacl_result["violations"]
                }
                matches.append(match_info)

            final_matches = [m for m in matches if m["shacl_conforms"]]

            print(f"\nFinal Matches: {len(final_matches)}")
            for m in final_matches:
                print(f"  - {m['signifier_id']} (similarity: {m['intent_similarity']:.4f})")

            result["matches"] = matches
            result["final_matches"] = [m["signifier_id"] for m in final_matches]

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

    def run(self) -> None:
        """Execute the complete test scenario."""
        start_time = datetime.now()

        print("\n" + "=" * 80)
        print("SCENARIO TEST RUNNER")
        print(f"Scenario: {self.scenario_path}")
        print(f"Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)

        try:
            self.clear_storage()
            self.initialize_components()
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


def main():
    """Main entry point for the script."""
    if len(sys.argv) != 2:
        print("Usage: python run_scenario_test.py <scenario_folder>")
        print("Example: python run_scenario_test.py test_scenario/1")
        sys.exit(1)

    scenario_path = sys.argv[1]
    runner = ScenarioTestRunner(scenario_path)
    runner.run()


if __name__ == "__main__":
    main()
