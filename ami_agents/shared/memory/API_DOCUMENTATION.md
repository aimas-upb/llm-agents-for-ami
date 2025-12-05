# RD4 Signifier System API Documentation

## Overview

The RD4 Signifier System provides an OpenAI-compatible REST API for managing signifiers and matching them against user intents and contexts. The system performs two-phase matching:

1. Intent matching using semantic similarity
2. SHACL validation against context constraints

Base URL: `http://localhost:8000`

## Quick Start

### Starting the Server

Start the API server using uvicorn:

```bash
python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

Or start it programmatically:

```python
import uvicorn

uvicorn.run(
    "src.api.main:app",
    host="0.0.0.0",
    port=8000,
    reload=True
)
```

### Verify Server is Running

```bash
curl http://localhost:8000/health
```

Response:
```json
{
    "status": "healthy",
    "version": "1.0.0"
}
```

## Endpoints

### Root Endpoint

Get API information.

```http
GET /
```

**Response:**
```json
{
    "name": "RD4 Signifier System",
    "version": "1.0.0",
    "phase": "Phase 3 - Intent Matcher",
    "status": "operational"
}
```

### Health Check

Check if the API is healthy and running.

```http
GET /health
```

**Response:**
```json
{
    "status": "healthy",
    "version": "1.0.0"
}
```

### List All Signifiers

Get a list of all signifiers stored in memory.

```http
GET /signifiers
```

**Response:**
```json
{
    "signifiers": [
        {
            "signifier_id": "heat-room-generic-signifier",
            "version": 1,
            "status": "active",
            "intent": "increase temperature in a room",
            "affordance_uri": "http://example.org/affordances/heating-system"
        }
    ],
    "total": 1
}
```

**Example using Python:**

```python
import requests

response = requests.get("http://localhost:8000/signifiers")
data = response.json()

print(f"Total signifiers: {data['total']}")
for signifier in data['signifiers']:
    print(f"- {signifier['signifier_id']}: {signifier['intent']}")
```

**Example using curl:**

```bash
curl http://localhost:8000/signifiers
```

### Create Signifier from RDF

Create a new signifier from RDF data in Turtle format.

```http
POST /signifiers
Content-Type: application/json
```

**Request Body:**
```json
{
    "rdf_data": "@prefix cashmere: <https://aimas.cs.pub.ro/ont/cashmere#> .\n..."
}
```

**Response (201 Created):**
```json
{
    "signifier_id": "heat-room-generic-signifier",
    "message": "Signifier heat-room-generic-signifier created successfully"
}
```

**Example using Python:**

```python
import requests

# Read RDF data from file
with open("signifiers/heat-room-signifier.ttl", "r", encoding="utf-8") as f:
    rdf_data = f.read()

# Create signifier
response = requests.post(
    "http://localhost:8000/signifiers",
    json={"rdf_data": rdf_data}
)

if response.status_code == 201:
    data = response.json()
    print(f"Created: {data['signifier_id']}")
else:
    print(f"Error: {response.status_code} - {response.text}")
```

**Example using curl:**

```bash
curl -X POST http://localhost:8000/signifiers \
  -H "Content-Type: application/json" \
  -d "{\"rdf_data\":\"$(cat signifiers/heat-room-signifier.ttl)\"}"
```

**RDF Format Example:**

The RDF data must be in Turtle format following the CASHMERE ontology:

```turtle
@prefix cashmere: <https://aimas.cs.pub.ro/ont/cashmere#> .
@prefix sh: <http://www.w3.org/ns/shacl#> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .

<#heat-room-generic-signifier> a cashmere:Signifier ;
    cashmere:signifies <http://example.org/affordances/heating-system> ;
    cashmere:hasIntentionDescription [
        a cashmere:IntentionDescription ;
        cashmere:hasStructuredDescription """
{
    "intent": "increase temperature in a room"
}
"""^^xsd:string ;
    ] ;
    cashmere:recommendsContext [
        a cashmere:IntentContext ;
        cashmere:hasStructuredDescription """
{
    "conditions": [
        {
            "artifact": "http://example.org/artifacts/temperature_sensor",
            "propertyAffordance": "http://example.org/TemperatureSensor#hasTemperatureLevel",
            "valueConditions": [
                {
                    "operator": "lessThan",
                    "value": 20
                }
            ]
        }
    ]
}
"""^^xsd:string ;
        cashmere:hasShaclCondition [
            a sh:NodeShape ;
            sh:targetNode <http://example.org/artifacts/temperature_sensor> ;
            sh:property [
                sh:path <http://example.org/TemperatureSensor#hasTemperatureLevel> ;
                sh:datatype xsd:integer ;
                sh:maxExclusive 20 ;
            ] ;
        ] ;
    ] .
```

### Delete All Signifiers

Clear all signifiers from memory (delete storage).

```http
DELETE /signifiers
```

**Response:**
```json
{
    "success": true,
    "message": "Successfully deleted all signifiers from memory",
    "deleted_count": 5
}
```

**Example using Python:**

```python
import requests

response = requests.delete("http://localhost:8000/signifiers")
data = response.json()

if data['success']:
    print(f"Deleted {data['deleted_count']} signifiers")
```

**Example using curl:**

```bash
curl -X DELETE http://localhost:8000/signifiers
```

### Match Signifiers

Match signifiers based on an intent query and optional context constraints.

```http
GET /signifiers/match?intent={intent}&context={context_json}
```

**Query Parameters:**

- `intent` (required): Natural language intent query
- `context` (optional): JSON string of context data with artifact URIs and property values

**Context Format:**

The context parameter should be a JSON string with the following structure:

```json
{
    "http://example.org/artifacts/temperature_sensor": {
        "http://example.org/TemperatureSensor#hasTemperatureLevel": 18
    },
    "http://example.org/artifacts/person_counter": {
        "http://example.org/PersonCounter#hasPersonCount": 5
    }
}
```

**Response:**
```json
{
    "matches": [
        {
            "signifier_id": "heat-room-generic-signifier",
            "intent_similarity": 0.8542,
            "shacl_conforms": true,
            "shacl_violations": []
        },
        {
            "signifier_id": "heat-room-crowded-signifier",
            "intent_similarity": 0.8234,
            "shacl_conforms": false,
            "shacl_violations": [
                "Value 18 does not conform to constraint maxExclusive 15"
            ]
        }
    ],
    "final_matches": [
        "heat-room-generic-signifier"
    ],
    "total_signifiers": 10
}
```

**Response Fields:**

- `matches`: All signifiers that matched the intent query (including those that failed SHACL validation)
  - `signifier_id`: Unique identifier of the signifier
  - `intent_similarity`: Semantic similarity score between 0.0 and 1.0
  - `shacl_conforms`: Whether the context passed SHACL validation
  - `shacl_violations`: List of validation error messages (empty if passed)
- `final_matches`: List of signifier IDs that passed both intent matching and SHACL validation
- `total_signifiers`: Total number of signifiers in the registry

**Example using Python:**

```python
import requests
import json

# Simple intent matching without context
response = requests.get(
    "http://localhost:8000/signifiers/match",
    params={"intent": "make the room warmer"}
)

data = response.json()
print(f"Total signifiers: {data['total_signifiers']}")
print(f"Intent matches: {len(data['matches'])}")
print(f"Final matches: {len(data['final_matches'])}")

for match in data['matches']:
    status = "PASS" if match['shacl_conforms'] else "FAIL"
    print(f"\n{match['signifier_id']}")
    print(f"  Similarity: {match['intent_similarity']:.4f}")
    print(f"  SHACL: {status}")
    if match['shacl_violations']:
        for violation in match['shacl_violations']:
            print(f"    - {violation}")
```

**Example with Context:**

```python
import requests
import json

# Intent matching with context constraints
context = {
    "http://example.org/artifacts/temperature_sensor": {
        "http://example.org/TemperatureSensor#hasTemperatureLevel": 18
    },
    "http://example.org/artifacts/person_counter": {
        "http://example.org/PersonCounter#hasPersonCount": 5
    }
}

response = requests.get(
    "http://localhost:8000/signifiers/match",
    params={
        "intent": "make the room more comfortable",
        "context": json.dumps(context)
    }
)

data = response.json()
print(f"Final matches (passed SHACL): {data['final_matches']}")
```

**Example using curl:**

```bash
# Simple intent matching
curl "http://localhost:8000/signifiers/match?intent=make+the+room+warmer"

# With context
curl -G "http://localhost:8000/signifiers/match" \
  --data-urlencode "intent=make the room more comfortable" \
  --data-urlencode 'context={"http://example.org/artifacts/temperature_sensor":{"http://example.org/TemperatureSensor#hasTemperatureLevel":18}}'
```

## Complete Usage Example

Here is a complete example showing how to use the API from start to finish:

```python
import json
import requests
from pathlib import Path

# Configuration
API_URL = "http://localhost:8000"
SIGNIFIERS_DIR = Path("test_scenario/4/signifiers")

# Create a session for connection pooling
session = requests.Session()

# Step 1: Check server health
response = session.get(f"{API_URL}/health")
print(f"Server health: {response.json()['status']}")

# Step 2: Clear existing signifiers
response = session.delete(f"{API_URL}/signifiers")
print(f"Cleared {response.json()['deleted_count']} signifiers")

# Step 3: Load signifiers from files
signifier_files = sorted(SIGNIFIERS_DIR.glob("*.ttl"))
print(f"\nLoading {len(signifier_files)} signifiers...")

loaded_ids = []
for file_path in signifier_files:
    with open(file_path, "r", encoding="utf-8") as f:
        rdf_data = f.read()

    response = session.post(
        f"{API_URL}/signifiers",
        json={"rdf_data": rdf_data}
    )

    if response.status_code == 201:
        signifier_id = response.json()["signifier_id"]
        loaded_ids.append(signifier_id)
        print(f"  Loaded: {signifier_id}")

print(f"\nTotal loaded: {len(loaded_ids)}")

# Step 4: List all signifiers
response = session.get(f"{API_URL}/signifiers")
signifiers = response.json()["signifiers"]
print(f"\nSignifiers in memory: {len(signifiers)}")
for s in signifiers[:3]:
    print(f"  - {s['signifier_id']}: {s['intent']}")

# Step 5: Match signifiers with intent and context
intent = "make the room more comfortable"
context = {
    "http://example.org/artifacts/temperature_sensor": {
        "http://example.org/TemperatureSensor#hasTemperatureLevel": 18
    },
    "http://example.org/artifacts/person_counter": {
        "http://example.org/PersonCounter#hasPersonCount": 5
    },
    "http://example.org/artifacts/internal_light_sensor": {
        "http://example.org/LightSensor#hasLuminosityLevel": 50
    },
    "http://example.org/artifacts/external_light_sensor": {
        "http://example.org/LightSensor#hasLuminosityLevel": 15000
    }
}

response = session.get(
    f"{API_URL}/signifiers/match",
    params={
        "intent": intent,
        "context": json.dumps(context)
    }
)

data = response.json()
print(f"\n{'='*80}")
print(f"Query: {intent}")
print(f"{'='*80}")
print(f"Total signifiers: {data['total_signifiers']}")
print(f"Intent matches: {len(data['matches'])}")
print(f"Final matches (passed SHACL): {len(data['final_matches'])}")

print(f"\nMatches:")
for match in data['matches']:
    status = "PASS" if match['shacl_conforms'] else "FAIL"
    print(f"\n  {match['signifier_id']}")
    print(f"    Similarity: {match['intent_similarity']:.4f}")
    print(f"    SHACL: {status}")
    if match['shacl_violations']:
        for violation in match['shacl_violations']:
            print(f"      - {violation}")

print(f"\nFinal matches:")
for signifier_id in data['final_matches']:
    matching_entry = next(
        (m for m in data['matches'] if m['signifier_id'] == signifier_id),
        None
    )
    if matching_entry:
        print(f"  - {signifier_id} (similarity: {matching_entry['intent_similarity']:.4f})")
```

## Error Handling

The API returns standard HTTP status codes:

- `200 OK`: Request succeeded
- `201 Created`: Resource created successfully
- `400 Bad Request`: Invalid request data
- `404 Not Found`: Resource not found
- `500 Internal Server Error`: Server error

**Error Response Format:**
```json
{
    "detail": "Error message describing what went wrong"
}
```

**Example Error Handling:**

```python
import requests

try:
    response = requests.post(
        "http://localhost:8000/signifiers",
        json={"rdf_data": "invalid rdf data"}
    )
    response.raise_for_status()

except requests.exceptions.HTTPError as e:
    if e.response.status_code == 400:
        print(f"Invalid RDF data: {e.response.json()['detail']}")
    elif e.response.status_code == 500:
        print(f"Server error: {e.response.json()['detail']}")
    else:
        print(f"HTTP error: {e}")

except requests.exceptions.RequestException as e:
    print(f"Request failed: {e}")
```

## Running Test Scenarios

The repository includes a test runner script that demonstrates complete API usage:

```bash
python scripts/run_scenario_test_api.py test_scenario/4
```

This script:
1. Starts the API server (optional with `--start-server` flag)
2. Clears the storage
3. Loads all signifiers from the scenario folder
4. Runs queries from `queries.json`
5. Saves results to a JSON file

**Usage:**
```bash
# Using existing server
python scripts/run_scenario_test_api.py test_scenario/4

# Start server automatically
python scripts/run_scenario_test_api.py test_scenario/4 --start-server

# Use custom API URL
python scripts/run_scenario_test_api.py test_scenario/4 --api-url http://localhost:8080
```

## API Client Libraries

### Python

```python
import requests
import json

class SignifierAPI:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
        self.session = requests.Session()

    def health(self):
        response = self.session.get(f"{self.base_url}/health")
        return response.json()

    def list_signifiers(self):
        response = self.session.get(f"{self.base_url}/signifiers")
        return response.json()

    def create_signifier(self, rdf_data):
        response = self.session.post(
            f"{self.base_url}/signifiers",
            json={"rdf_data": rdf_data}
        )
        response.raise_for_status()
        return response.json()

    def delete_all_signifiers(self):
        response = self.session.delete(f"{self.base_url}/signifiers")
        return response.json()

    def match(self, intent, context=None):
        params = {"intent": intent}
        if context:
            params["context"] = json.dumps(context)

        response = self.session.get(
            f"{self.base_url}/signifiers/match",
            params=params
        )
        response.raise_for_status()
        return response.json()

# Usage
api = SignifierAPI()
print(api.health())

# Load signifier
with open("signifier.ttl", "r") as f:
    api.create_signifier(f.read())

# Match
results = api.match(
    intent="make the room warmer",
    context={
        "http://example.org/artifacts/temperature_sensor": {
            "http://example.org/TemperatureSensor#hasTemperatureLevel": 18
        }
    }
)
print(f"Matches: {results['final_matches']}")
```

## Configuration

The API can be configured using environment variables or a `.env` file:

```bash
APP_NAME="RD4 Signifier System"
VERSION="1.0.0"
LOG_LEVEL="INFO"
STORAGE_DIR="storage"
ENABLE_AUTHORING_VALIDATION=false
```

## Performance Considerations

1. The API maintains signifiers in memory for fast access
2. Intent matching uses pre-computed embeddings
3. SHACL validation is performed on-demand
4. Connection pooling is recommended for multiple requests
5. Use bulk operations when loading multiple signifiers

## Limitations

1. All signifiers are stored in memory (cleared on restart)
2. No authentication or authorization
3. No pagination for large result sets (use limit parameter)
4. CORS enabled for all origins (configure for production)
5. Context must be provided as JSON string in GET requests

## Support

For issues, questions, or feature requests, please refer to the project documentation or contact the development team.
