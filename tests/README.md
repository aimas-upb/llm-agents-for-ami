# AMI Agents Tests

This directory contains tests for the AMI Agents project.

## Setup

Install test dependencies:

```bash
pip install -r requirements.txt
```

## Running Tests

### Run all tests

```bash
pytest
```

### Run tests with verbose output

```bash
pytest -v
```

### Run tests with output capture disabled (see print statements)

```bash
pytest -s
```

### Run specific test file

```bash
pytest tests/environment/integration/test_yggdrasil_integration.py
```

### Run specific test

```bash
pytest tests/environment/integration/test_yggdrasil_integration.py::TestYggdrasilIntegration::test_initialize_with_localhost
```

### Run with coverage

```bash
pytest --cov=ami_agents --cov-report=html
```

Coverage report will be generated in `htmlcov/index.html`.

## YggdrasilIntegration Tests

The tests in `tests/environment/integration/test_yggdrasil_integration.py` verify the HMAS platform dereferencing and validation.

### Prerequisites for localhost tests

To run the tests that use `http://localhost:8080/`, you need:

1. A Yggdrasil instance running on `http://localhost:8080/`
2. The instance should serve RDF content (Turtle, RDF/XML, JSON-LD, etc.)
3. The RDF content should represent either:
   - **Case A**: A direct `hmas:HypermediaMASPlatform` instance
   - **Case B**: An `hmas:ResourceProfile` with `hmas:isProfileOf` pointing to a platform

### Sample RDF Content

#### Case A: Direct Platform

```turtle
@prefix hmas: <https://purl.org/hmas/> .
@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .

<http://localhost:8080/> a hmas:HypermediaMASPlatform ;
    hmas:hasName "Test Yggdrasil Platform" .
```

#### Case B: ResourceProfile

```turtle
@prefix hmas: <https://purl.org/hmas/> .
@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .

<http://localhost:8080/> a hmas:ResourceProfile ;
    hmas:isProfileOf <http://localhost:8080/platform> .

<http://localhost:8080/platform> a hmas:HypermediaMASPlatform ;
    hmas:hasName "Test Yggdrasil Platform" .
```

### Setting up a local test server

You can use a simple Python HTTP server to serve test RDF files:

```bash
# Create a test RDF file
cat > test_platform.ttl << 'EOF'
@prefix hmas: <https://purl.org/hmas/> .
@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .

<http://localhost:8080/> a hmas:HypermediaMASPlatform ;
    hmas:hasName "Test Yggdrasil Platform" .
EOF

# Serve it (note: you'll need to configure the server to serve it at the root)
# Or use a proper Yggdrasil instance
```

## Test Structure

```
tests/
├── __init__.py
├── README.md
└── environment/
    ├── __init__.py
    └── integration/
        ├── __init__.py
        └── test_yggdrasil_integration.py
```

## Writing New Tests

When writing new tests:

1. Use `pytest` fixtures for setup and teardown
2. Mark async tests with `@pytest.mark.asyncio`
3. Use descriptive test names starting with `test_`
4. Include docstrings explaining what the test verifies
5. Follow the Arrange-Act-Assert pattern

Example:

```python
@pytest.mark.asyncio
async def test_my_feature():
    """Test that my feature works correctly."""
    # Arrange
    setup_data = create_test_data()

    # Act
    result = await my_async_function(setup_data)

    # Assert
    assert result == expected_value
```

## Continuous Integration

These tests can be integrated into CI/CD pipelines. Example GitHub Actions workflow:

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
        with:
          python-version: '3.10'
      - run: pip install -r requirements.txt
      - run: pytest --cov=ami_agents
```
