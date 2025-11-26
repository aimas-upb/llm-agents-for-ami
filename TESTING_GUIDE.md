# Testing Guide for YggdrasilIntegration

## Fixed Issues

1. The `Affordance` dataclass had a field ordering issue that has been fixed. The `uri` field was moved before optional fields to comply with Python dataclass rules.

2. **Ontology Format Requirements**: The HMAS and CASHMERE ontology files must be in OWL/XML (`.owl`), RDF/XML (`.rdf`), or NTriples (`.nt`) format, as these are the only formats supported by owlready2.

## Important Notes

- **Ontology files** (in `ami_agents/shared/ontologies/`): Must be OWL/XML, RDF/XML, or NTriples
- **Yggdrasil RDF content** (from HTTP server): Can be any RDF format (Turtle, RDF/XML, etc.)
- **Why the difference?**: owlready2 (for ontology vocabulary) has limited format support, but rdflib (for graph parsing) supports all RDF formats

## Running the Tests

### 1. Install Dependencies (if not already done)

```bash
pip install -r requirements.txt
```

### 2. Start Your Yggdrasil Instance

Make sure you have a Yggdrasil instance running at `http://localhost:8080/` that serves RDF content.

### 3. Verify Localhost is Working

Before running the full test suite, verify that your localhost is serving valid HMAS content:

```bash
python3 tests/check_localhost.py
```

Expected output if working:
```
Checking http://localhost:8080/...
------------------------------------------------------------
✓ Successfully parsed RDF from http://localhost:8080/
  Total triples: X

✓ Case A: URL is directly typed as hmas:HypermediaMASPlatform
  Platform URI: http://localhost:8080/
```

OR:

```
✓ URL is typed as hmas:ResourceProfile
✓ Case B: Profile points to hmas:HypermediaMASPlatform
  Profile URI: http://localhost:8080/
  Platform URI: http://localhost:8080/platform
```

### 4. Run the Tests

```bash
# Run all integration tests
pytest tests/environment/integration/test_yggdrasil_integration.py -v -s

# Run a specific test
pytest tests/environment/integration/test_yggdrasil_integration.py::TestYggdrasilIntegration::test_initialize_with_localhost -v -s

# Run with detailed RDF graph output
pytest tests/environment/integration/test_yggdrasil_integration.py::TestYggdrasilIntegration::test_initialize_graph_structure -v -s
```

### 5. Understanding Test Output

The tests will:

1. **test_initialize_with_localhost**: Basic initialization test
   - Verifies connection to localhost:8080
   - Checks that platform URI is discovered
   - Validates RDF graph was parsed

2. **test_initialize_with_direct_platform**: Verifies Case A
   - Checks if URL is directly a HypermediaMASPlatform
   - Validates platform URI matches the URL

3. **test_initialize_graph_structure**: Detailed graph inspection
   - Prints all RDF triples
   - Shows graph statistics
   - Useful for debugging

4. **test_initialize_invalid_url**: Error handling test
   - Verifies graceful failure with invalid URLs
   - Doesn't require localhost to be running

## Sample RDF Content for Testing

### Case A: Direct Platform

Save this to a file served at `http://localhost:8080/`:

```turtle
@prefix hmas: <https://purl.org/hmas/> .
@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .

<http://localhost:8080/> a hmas:HypermediaMASPlatform ;
    rdfs:label "Test Yggdrasil Platform" ;
    hmas:hasName "Test Platform" .
```

### Case B: ResourceProfile

```turtle
@prefix hmas: <https://purl.org/hmas/> .
@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .

<http://localhost:8080/> a hmas:ResourceProfile ;
    rdfs:label "Platform Profile" ;
    hmas:isProfileOf <http://localhost:8080/platform> .

<http://localhost:8080/platform> a hmas:HypermediaMASPlatform ;
    rdfs:label "Test Yggdrasil Platform" ;
    hmas:hasName "Test Platform" .
```

## Troubleshooting

### "Failed to dereference or validate Yggdrasil URL"

This means:
- Localhost:8080 is not running, or
- The RDF content doesn't match the expected HMAS structure, or
- The content type headers are not correct

**Solution**: Run `python3 tests/check_localhost.py` to see exactly what's wrong.

### "Failed to load HMAS ontology"

This can mean:
- The HMAS ontology file is missing from `ami_agents/shared/ontologies/`
- The ontology file is in an unsupported format (e.g., Turtle)

**Solution**:
1. Check that `ami_agents/shared/ontologies/hmas.owl` exists
2. If you have `hmas.ttl`, convert it to OWL/XML format:
   ```bash
   # Using rdflib in Python
   from rdflib import Graph
   g = Graph()
   g.parse("ami_agents/shared/ontologies/hmas.ttl", format="turtle")
   g.serialize(destination="ami_agents/shared/ontologies/hmas.owl", format="xml")
   ```
3. Ensure the file is in OWL/XML (`.owl`), RDF/XML (`.rdf`), or NTriples (`.nt`) format

### Import errors

**Solution**: Make sure you're in the project root directory and have installed all dependencies:
```bash
cd /path/to/llm-agents-for-ami
pip install -r requirements.txt
```

## Next Steps

Once the tests pass, you can:

1. Extend tests to cover more HMAS platform features
2. Test with real Yggdrasil deployments
3. Add tests for workspace and artifact discovery
4. Test error handling with malformed RDF

## CI/CD Integration

To run these tests in CI/CD, ensure your pipeline:
1. Installs dependencies: `pip install -r requirements.txt`
2. Starts a test Yggdrasil instance on localhost:8080
3. Runs tests: `pytest tests/`
4. Stops the test instance

Example GitHub Actions workflow snippet:
```yaml
- name: Run tests
  run: |
    # Start test Yggdrasil instance
    docker run -d -p 8080:8080 yggdrasil-test

    # Wait for it to be ready
    sleep 5

    # Run tests
    pytest tests/environment/integration/

    # Cleanup
    docker stop $(docker ps -q --filter ancestor=yggdrasil-test)
```
