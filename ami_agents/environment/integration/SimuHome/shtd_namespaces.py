"""Every vocabulary IRI SHTD serves, defined once for the whole adapter.

SHTD and its verification scripts run from this directory and cannot import the
agents' package, so they keep this one copy of the namespaces instead of one
per module. The copy must equal the agents' single source,
`ami_agents/shared/utils/namespaces.py`: `tests/unit/test_namespaces.py`
compares the two and fails on any difference. Changing a namespace -- for
example when homeont and TD-SOSA move from their `example.org` placeholders to
published IRIs -- is an edit there and here, and the test says if one was missed.

Modules import from here with the package-or-script fallback the adapter
already uses (`from .shtd_namespaces import TD` / `from shtd_namespaces import TD`).
"""

from rdflib import Namespace

# -- W3C Web of Things -------------------------------------------------------
TD = Namespace("https://www.w3.org/2019/wot/td#")
JS = Namespace("https://www.w3.org/2019/wot/json-schema#")
HCTL = Namespace("https://www.w3.org/2019/wot/hypermedia#")
WOTSEC = Namespace("https://www.w3.org/2019/wot/security#")
HTV = Namespace("http://www.w3.org/2011/http#")

# -- Hypermedia MAS ----------------------------------------------------------
HMAS = Namespace("https://purl.org/hmas/")
WEBSUB = Namespace("https://purl.org/hmas/websub/")
JACAMO = Namespace("https://purl.org/hmas/jacamo/")

# -- Sensing, actuation, devices, units ----------------------------------------
SOSA = Namespace("http://www.w3.org/ns/sosa/")
SSN = Namespace("http://www.w3.org/ns/ssn/")
SAREF = Namespace("https://saref.etsi.org/core/")
QUDT = Namespace("http://qudt.org/schema/qudt/")
UNIT = Namespace("http://qudt.org/vocab/unit/")
QUANTITYKIND = Namespace("http://qudt.org/vocab/quantitykind/")
# Thing-level device metadata (manufacturer, product name). The TD spec mints no
# vendor vocabulary of its own and points at schema.org for it.
SCHEMA = Namespace("https://schema.org/")

# -- Our own vocabularies (placeholder IRIs until published) --------------------
# All SHTD vocabulary -- spaces, device families, environmental properties and
# Matter provenance -- lives in one ontology: ami_agents/shared/ontologies/homeont.ttl
HOMEONT = Namespace("http://example.org/homeont/")
TDSOSA = Namespace("https://example.org/hmas/td-sosa-ext#")
