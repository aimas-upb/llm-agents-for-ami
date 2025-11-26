# Ontologies Directory

This directory contains OWL ontology files used by the AMI agents system for semantic processing and reasoning.

## Available Ontologies

### 1. hmas.owl
The **Hypermedia MAS (HMAS)** ontology defines the core concepts for hypermedia-based multi-agent systems, including:
- `HypermediaMASPlatform`: The main platform concept
- `ResourceProfile`: Profile descriptions for resources
- `isProfileOf`: Property linking profiles to resources
- Workspace and artifact concepts
- Thing Description integration

### 2. cashmere.owl
The **CASHMERE** ontology provides the semantic model for signifier representation and affordance theory, including:
- Signifier concepts
- Affordance representations
- Agent-environment interaction patterns

## Supported Formats

**IMPORTANT**: owlready2 only supports the following RDF serialization formats:
- **OWL/XML** (`.owl`) - Recommended
- **RDF/XML** (`.rdf`)
- **NTriples** (`.nt`)

**Unsupported formats**: Turtle (`.ttl`), N3 (`.n3`), JSON-LD, and other formats are NOT supported by owlready2.

If you have ontology files in other formats, convert them to OWL/XML or RDF/XML before placing them in this directory.

## Usage

### Loading Ontologies

```python
from ami_agents.shared.ontologies import get_hmas_ontology, get_cashmere_ontology

# Load HMAS ontology
hmas_onto = get_hmas_ontology()

# Load CASHMERE ontology
cashmere_onto = get_cashmere_ontology()

# Access ontology elements
print(f"HMAS ontology IRI: {hmas_onto.base_iri}")
```

### Using the OntologyLoader Directly

```python
from ami_agents.shared.ontologies import OntologyLoader

# Load a custom ontology
onto = OntologyLoader.load_ontology("custom.owl")

# Get the path to an ontology file
path = OntologyLoader.get_ontology_path("hmas.owl")

# Clear the cache to force reload
OntologyLoader.clear_cache()
```

### Working with Loaded Ontologies

```python
from ami_agents.shared.ontologies import get_hmas_ontology

# Load the ontology
hmas = get_hmas_ontology()

# Access classes
platform_class = hmas.HypermediaMASPlatform
profile_class = hmas.ResourceProfile

# Access properties
is_profile_of = hmas.isProfileOf

# Get the IRI of a class
platform_iri = platform_class.iri

# Query instances (if any exist in the ontology)
platforms = list(platform_class.instances())
```

## Adding New Ontologies

To add a new ontology:

1. **Convert to supported format** (if necessary):
   ```bash
   # Example: Convert Turtle to OWL/XML using rapper
   rapper -i turtle -o rdfxml your_ontology.ttl > your_ontology.owl

   # Or using rdflib in Python
   from rdflib import Graph
   g = Graph()
   g.parse("your_ontology.ttl", format="turtle")
   g.serialize(destination="your_ontology.owl", format="xml")
   ```

2. Place the `.owl` file in this directory

3. Add a convenience function in [loader.py](loader.py):

   ```python
   def get_your_ontology(reload: bool = False) -> Optional[Ontology]:
       """Load your custom ontology."""
       for ext in ["your_ontology.owl", "your_ontology.rdf", "your_ontology.nt"]:
           if (ONTOLOGIES_DIR / ext).exists():
               return OntologyLoader.load_ontology(ext, reload=reload)

       logger.error("Your ontology file not found")
       return None
   ```

4. Export the function in [\_\_init\_\_.py](__init__.py):

   ```python
   from .loader import get_your_ontology

   __all__ = [..., "get_your_ontology"]
   ```

## Integration with RDFLib

The ontologies loaded with owlready2 provide vocabulary access, while RDFLib handles graph operations:

```python
from rdflib import Graph, URIRef
from rdflib.namespace import RDF
from ami_agents.shared.ontologies import get_hmas_ontology

# Load ontology with owlready2 to get vocabulary
hmas_onto = get_hmas_ontology()

# Extract vocabulary IRIs
HypermediaMASPlatform_iri = URIRef(hmas_onto.HypermediaMASPlatform.iri)
ResourceProfile_iri = URIRef(hmas_onto.ResourceProfile.iri)
isProfileOf_iri = URIRef(hmas_onto.isProfileOf.iri)

# Use RDFLib for graph operations
graph = Graph()
graph.parse("https://example.org/platform")

# Check for HMAS platform using vocabulary from ontology
platform_uri = URIRef("https://example.org/platform")
if (platform_uri, RDF.type, HypermediaMASPlatform_iri) in graph:
    print("Found HMAS platform")
```

## Converting Ontology Formats

If you have ontology files in unsupported formats (like Turtle), you need to convert them:

### Using rdflib (Python)

```python
from rdflib import Graph

# Load from Turtle
g = Graph()
g.parse("ontology.ttl", format="turtle")

# Save as OWL/XML
g.serialize(destination="ontology.owl", format="xml")
```

### Using rapper (Command Line)

```bash
# Install rapper (part of raptor2-utils package)
sudo apt-get install raptor2-utils  # Ubuntu/Debian
brew install raptor              # macOS

# Convert Turtle to OWL/XML
rapper -i turtle -o rdfxml ontology.ttl > ontology.owl
```

### Using Protégé

1. Open the ontology file in Protégé
2. Go to File → Save As
3. Choose "RDF/XML" as the format
4. Save with `.owl` extension

## Notes

- Ontologies are cached after the first load to improve performance
- Use `reload=True` parameter to force reload an ontology
- The ontology files must be in OWL/XML, RDF/XML, or NTriples format
- owlready2 provides reasoning capabilities and direct Python object access to ontology elements
- For pure RDF graph operations, use RDFLib directly
- Vocabulary access from owlready2 + graph operations from RDFLib = best of both worlds

## Troubleshooting

### "Failed to load ontology" errors

**Cause**: The ontology file is in an unsupported format (e.g., Turtle)

**Solution**: Convert the file to OWL/XML format using one of the methods above

### "Ontology file not found" errors

**Cause**: The file doesn't exist or has a different name

**Solution**: Check that the file exists in this directory and matches the expected name (e.g., `hmas.owl`, `cashmere.owl`)
