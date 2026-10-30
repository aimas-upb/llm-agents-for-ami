"""Export lab308e's live assertion results into the shared benchmark score format."""
import argparse
import json
from pathlib import Path

from tests.hasp.hasp_lab308e_benchmark import write_json


def evaluate(record):
    result = record.get("evaluation")
    if not isinstance(result, dict) or result.get("score") not in {-1, 0, 1}:
        result = {"score": -1, "error_type": "Missing live HASP assertions", "check": "lab308e"}
    checks = []

    def collect(assertion):
        if "results" in assertion:
            # Preserve any_of as one logical check; flattening would incorrectly
            # fail an alternative that the fixture does not require.
            if assertion.get("type") == "all_of":
                for child in assertion["results"]:
                    collect(child)
                return
        checks.append({"goal": 0, "description": assertion.get("path") or assertion.get("type"),
                       "class": "ok" if assertion.get("passed") else "wrong_value"})

    for assertion in result.get("assertions", []):
        collect(assertion)
    good = sum(check["class"] == "ok" for check in checks)
    return {**result, "episode": record["episode"], "query_type": record["query_type"],
            "case": record["case"], "outcome": record.get("outcome"),
            "breakdown": {"checks": checks, "total": len(checks), "ok": good,
                          "success": good / len(checks) if checks else None}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", required=True)
    parser.add_argument("--skip-scored", action="store_true")
    args = parser.parse_args(argv)
    directory = Path(args.records)
    paths = [directory] if directory.is_file() else sorted(directory.glob("*.json"))
    scores = []
    for path in paths:
        if path.name.endswith(".eval.json"):
            continue
        record = json.loads(path.read_text())
        if "episode" not in record:
            continue
        target = path.with_name(path.stem + ".eval.json")
        result = json.loads(target.read_text()) if args.skip_scored and target.exists() else evaluate(record)
        if not (args.skip_scored and target.exists()):
            write_json(target, result)
        scores.append(result["score"])
        print(f"{record['episode']}: score={result['score']}")
    return 0 if scores and all(score == 1 for score in scores) else 1


if __name__ == "__main__":
    raise SystemExit(main())
