# Test Scenario Runner

Simple script for testing signifier matching scenarios.

## Usage

```bash
python scripts/run_scenario_test.py test_scenario/1
```

## Directory Structure

```
test_scenario/
  1/
    signifiers/           - TTL signifier files to load
    queries.json          - Test queries with intent and context
    test_log/            - Generated results (JSON files)
```

## Query Format

Each query in `queries.json`:

```json
{
  "1": {
    "query_id": "query-1-heat",
    "description": "Room is cold (18°C), user wants to increase temperature",
    "intent": "increase the temperature in the room",
    "context": {
      "http://example.org/artifacts/temperature_sensor": {
        "http://example.org/TemperatureSensor#hasTemperatureLevel": 18
      }
    }
  }
}
```

## What the Script Does

### Step 1: Clear Storage
Removes all existing signifiers to ensure clean state

### Step 2: Initialize Components
Creates SignifierRegistry, IntentMatcher, ContextBuilder, and SHACLValidator

### Step 3: Load Signifiers
Loads all .ttl files from the signifiers/ folder

### Step 4: Run Queries
For each query:
1. **Intent Matching** - Matches intent against all signifiers using embeddings
2. **SHACL Validation** - Validates context against each signifier's constraints
3. **Final Matches** - Returns only signifiers that pass both intent and context

## Output

### Console Output
Clean, readable output showing:
- Each signifier's intent similarity score
- SHACL validation results (PASS/FAIL)
- Violation messages when SHACL fails
- Final list of matching signifiers

### JSON File
Saved to `test_log/results_YYYYMMDD_HHMMSS.json`

Contains for each query:
- Query details (ID, description, intent, context)
- All matches with similarity scores
- SHACL validation results
- Final matching signifier IDs

Example JSON structure:

```json
{
  "1": {
    "query_id": "query-1-heat",
    "description": "Room is cold (18°C), user wants to increase temperature",
    "intent": "increase the temperature in the room",
    "context": { ... },
    "matches": [
      {
        "signifier_id": "heat-room-signifier",
        "intent_similarity": 0.8433,
        "shacl_conforms": true,
        "shacl_violations": []
      },
      {
        "signifier_id": "cool-room-signifier",
        "intent_similarity": 0.8253,
        "shacl_conforms": false,
        "shacl_violations": [
          "Value is not > Literal(\"25\", datatype=xsd:integer)"
        ]
      }
    ],
    "final_matches": ["heat-room-signifier"]
  }
}
```

## Creating New Test Scenarios

1. Create directory: `test_scenario/N/`
2. Add `signifiers/` subfolder with .ttl files
3. Create `queries.json` with test queries
4. Run: `python scripts/run_scenario_test.py test_scenario/N`
