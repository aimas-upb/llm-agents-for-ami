# RD4 Signifier System - Project Roadmap

## Status Convention
- [ ] = Todo (not started)
- [IN PROGRESS] = Currently being worked on (add YYYY-MM-DD when started)
- [DONE] = Completed (add YYYY-MM-DD when finished)

---

## High Priority

### Phase 1 - Storage (MVP)
Foundation for persisting signifiers with dual representation (NL + structured).

- [DONE] Design and implement Signifier Registry (SR) module (2025-11-11)
  - CRUD operations for signifiers
  - Versioning support (signifier_id + version)
  - Status management (active/deprecated)
  - Provenance tracking (created_at, created_by, source)

- [DONE] Implement Memory Store (MS) module (2025-11-11)
  - RDF store with named graphs (graph:signifier/{id}/{version})
  - JSON document store for canonical signifiers
  - Property indexes: (artifact_uri, property_uri) catalog

- [DONE] Build Representation Service (RS) module (2025-11-11)
  - Normalize NL and structured fields
  - Validate signifier structure at ingest
  - Accept SHACL shapes (validation disabled in Phase 1)

- [DONE] Create public Storage APIs (2025-11-11)
  - POST /signifiers (create)
  - PUT /signifiers/{id} (update with versioning)
  - GET /signifiers/{id} (retrieve)
  - GET /signifiers (list with filters: status, affordance_uri, q, limit, offset)

- [DONE] Implement basic property index catalog (2025-11-11)
  - Materialize (artifact, property) mappings at ingest
  - Enable fast candidate prefiltering

- [DONE] Storage acceptance criteria (2025-11-11)
  - Round-trip fidelity for signifier data
  - Listing filters work correctly
  - Index catalog functional
  - Idempotent ingest on repeated payloads

### Phase 2 - SHACL Constraints (Precise Validator)
Add validation capabilities for context snapshots and signifier authoring.

- [DONE] Implement SHACL Validator (SV) module core (2025-11-13)
  - Parse and store SHACL shapes graphs
  - Validate context graphs against signifier shapes
  - Return conforms + violation details

- [DONE] Add authoring SHACL validation (toggle via flag) (2025-11-13)
  - Validate signifier object structure at ingest
  - Check for required properties (signifies, hasIntentionDescription, recommendsContext)
  - Verify SHACL property paths use valid IRIs

- [DONE] Build runtime SHACL validation (2025-11-13)
  - Evaluate constraints: datatype, minInclusive, maxInclusive, minExclusive, maxExclusive
  - Target node selection (sh:targetNode, sh:targetClass)
  - Property shape evaluation

- [DONE] Implement batch validation endpoint (2025-11-13)
  - POST /validate/shacl for multiple candidates
  - Optimize for single context graph vs multiple signifiers

- [DONE] Add Context Graph Builder (CGB) module (2025-11-13)
  - Convert KV maps to canonical RDF
  - Extract context features: (artifact, property) -> value
  - Ensure typed literals (xsd:integer, etc.)

- [DONE] Implement validation result caching (2025-11-13)
  - Cache key: (signifier_id, rdf_hash, options)
  - Track cache hit rates

- [DONE] Create validation APIs (2025-11-13)
  - POST /signifiers/validate-authoring
  - POST /validate/shacl
  - POST /context/normalize

- [DONE] Phase 2 acceptance criteria (2025-11-13)
  - Invalid shapes rejected at ingest with diagnostics
  - Runtime validation returns conforms + top violations
  - Cache hit rate visible in metrics
  - Test with "Raise Blinds" and "Turn Light On" examples

### Project Setup & Configuration
- [DONE] Create project structure (2025-11-11)
- [DONE] Set up tracking files (2025-11-11)
- [DONE] Set up Python virtual environment (3.10+) (2025-11-11)
- [DONE] Create requirements.txt with core dependencies (2025-11-11)
  - RDF library (rdflib)
  - SHACL validation (pyshacl)
  - Vector embedding library (for Phase 3)
  - Web framework (FastAPI recommended)
- [DONE] Define global configuration structure (2025-11-11)
  - Module enable/disable flags
  - Default pipeline configuration
  - Latency budgets
  - Hard gates configuration

---

## Medium Priority

### Phase 3 - Intent Matcher (Replaceable, Versioned)
Pluggable intent matching with multiple algorithm versions.

- [DONE] Design Intent Matcher (IM) interface (2025-11-13)
  - Input: intent_query (nl_text | structured), k
  - Output: [(signifier_id, similarity)]
  - Versionable contract

- [DONE] Implement IM v0 - String Contains (2025-11-13)
  - Search intent.nl_text for query tokens
  - Search structured JSON literal for tokens
  - Simple string matching algorithm

- [DONE] Implement IM v1 - Embedding Similarity (2025-11-13)
  - Generate embeddings for intent.nl_text
  - Vector index for fast similarity search
  - Cosine similarity scoring
  - Store intent_embedding in signifier indexes

- [DONE] Add module version selection (2025-11-13)
  - Configurable default version
  - Per-request version override
  - Module registry with version tracking

- [DONE] Phase 3 acceptance criteria (2025-11-13)
  - A/B test: route 50/50 to v0 and v1
  - Collect similarity distributions and latency
  - Deterministic behavior for same input/version

### Phase 4 - Retrieval Orchestrator (Auto Pipeline)
Execute configurable retrieval pipelines with multiple stages.

- [ ] Implement Retrieval Orchestrator (ORCH) module
  - Pipeline executor honoring module order
  - Per-module latency tracking
  - Signal aggregation across modules

- [ ] Build default retrieval pipeline
  - Default: [IM, SV, RP] (SSE disabled initially)
  - Normalize context -> IM candidates -> SHACL validate -> Rank -> Return

- [ ] Add pipeline configuration support
  - Global default pipeline
  - Per-request pipeline overrides
  - Module enable/disable via feature flags

- [ ] Implement hard gates logic
  - If SHACL shapes exist and hard_gates.shacl=true, reject non-conforming
  - Configurable gate policies

- [ ] Create retrieval API
  - POST /retrieve/match
  - Request: intent_query, context_input, pipeline config
  - Response: [Match Result] with signals and explanations

- [ ] Build explanation system
  - Per-module signal tracking
  - Human-readable explanation bullets
  - Evidence collection (triples, violated constraints)

- [ ] Phase 4 acceptance criteria
  - End-to-end success on three signifier examples
  - Per-module latency reporting
  - Total latency within budget (150ms baseline)
  - Signals and explanations included in results

### Phase 5 - Structured Subsumption Engine (SSE)
Fast numeric pre-filter before expensive SHACL validation.

- [ ] Implement Structured Subsumption Engine (SSE) module
  - Evaluate structured_conditions operators: =, !=, <, <=, >, >=
  - Use extracted context_features KV map
  - Handle missing values per policy

- [ ] Add SSE result structure
  - Output: sse_pass (true/false)
  - Include reasons for failures

- [ ] Integrate SSE into pipeline
  - Position: after IM, before SV
  - Make optional via configuration

- [ ] Phase 5 acceptance criteria
  - Correct pass/fail on boundary cases (10000 vs >=10000)
  - Ablation study: measure SSE impact on candidate reduction
  - Latency under 20ms target

### Phase 6 - Ranking & Policy
Combine multiple signals into final scores with explainability.

- [ ] Implement Ranker & Policy (RP) module
  - Combine signals: similarity, sse_pass, shacl_conforms
  - Support for optional kg_score, llm_score

- [ ] Implement hard gate logic
  - If shapes present and shacl_conforms=false => score=0
  - Configurable gate enforcement

- [ ] Add weighted scoring
  - Configurable weights (similarity:0.4, sse:0.3, kg:0.2, llm:0.1)
  - Weighted fusion calculation

- [ ] Build explainability output
  - Emit weights and contributions per signal
  - Show which gates triggered
  - Include specificity boost logic

- [ ] Implement tie-breaking
  - More constraints => small specificity boost
  - Stable ordering given fixed weights

- [ ] Phase 6 acceptance criteria
  - Stable ordering with fixed weights
  - Explainability includes all signal contributions
  - Ties broken by specificity

---

## Low Priority

### Phase 7 - Telemetry, Evaluation & A/B Testing
Comprehensive metrics, evaluation harness, and experimentation framework.

- [ ] Implement Telemetry & Evaluation (TE) module
  - Per-module latency metrics
  - Candidate counts at each stage
  - Pass/fail rates per module

- [ ] Build evaluation harness
  - Upload labeled datasets (intent, context, expected_ids, notes)
  - Evaluation execution engine
  - Precision@K calculation

- [ ] Create A/B experiment manager
  - Experiment configuration (arms, pipelines, percentages)
  - Traffic splitting logic (hash on user/session)
  - Experiment result aggregation

- [ ] Implement ablation support
  - Configuration to disable specific modules
  - Compare pipeline variants
  - Measure lift/loss of components

- [ ] Build admin & telemetry APIs
  - GET /admin/stats
  - POST /eval/run
  - POST /datasets, GET /datasets/{id}
  - POST /experiments, GET /experiments

- [ ] Create seed evaluation dataset
  - 6-10 scenarios covering example signifiers
  - Include "Raise Blinds", "Turn Light On", etc.

- [ ] Phase 7 acceptance criteria
  - Reproducible eval runs
  - Stored configs and results
  - Dashboard endpoints functional
  - A/B comparison reports

### Phase 8 - Optional Advanced Modules
KG reasoning and LLM-based semantic judging.

- [ ] Implement KG Reasoner (KGR) module (optional)
  - RDFS/OWL-RL reasoning support
  - Entailment for sh:targetClass checks
  - Boost candidates via inferred relationships

- [ ] Implement LLM Judge (LLMJ) module (optional)
  - Last-mile semantic scoring
  - Never override failed SHACL (hard constraint)
  - Budgeted: candidate limit, timeout
  - Redaction policy for sensitive data

- [ ] Add optional module configuration
  - Off by default
  - Per-request enable/disable
  - Latency budget enforcement

- [ ] Phase 8 acceptance criteria
  - Measurable lift on curated test cases
  - No false positives overriding SHACL failures
  - Within latency budgets when enabled

### Documentation
- [ ] Write comprehensive README
  - System overview and architecture
  - Installation instructions
  - Quick start guide

- [ ] Create OpenAPI specification
  - All public endpoints documented
  - Request/response schemas
  - Authentication and authorization

- [ ] Document module manifests
  - name, version, inputs, outputs
  - latency_budgets_ms, flags

- [ ] Write authoring SHACL profile documentation
  - Required properties and structure
  - Validation rules and examples

- [ ] Create experiment config schema documentation
  - A/B test setup guide
  - Ablation study examples

- [ ] Document data contracts
  - Canonical Signifier structure
  - Context Snapshot format
  - Match Result format

### Testing
- [DONE] Unit tests for Storage module (SR, MS, RS) (2025-11-11)
- [DONE] Unit tests for SHACL Validator (SV) (2025-11-13)
- [DONE] Unit tests for Intent Matcher versions (IM v0, v1) (2025-11-13)
- [DONE] Test scenario runner for automated testing (2025-11-25)
  - Simplified script with clean output
  - Support for multiple test scenarios
  - JSON results with match details
- [DONE] Simplified FastAPI interface (2025-11-25)
  - 4 core endpoints: GET/POST/DELETE /signifiers, GET /signifiers/match
  - API client test script
  - API-based scenario runner
- [ ] Unit tests for SSE module
- [ ] Unit tests for Ranker & Policy (RP)
- [ ] Integration tests for retrieval pipeline
- [ ] Performance benchmarks
  - IM: <=30ms, SSE: <=20ms, SV: <=80ms, RP: <=10ms
  - End-to-end: <=150ms (without KG/LLM)
- [ ] Scalability tests (5k signifiers, <5k triples)

### Security & Operations
- [ ] Implement input validation and size caps
- [ ] Add RDF parser limits and timeouts
- [ ] Implement JSON schema validation
- [ ] Add authentication and authorization
  - Roles: writer, reader, admin
- [ ] Implement rate limiting and quotas
- [ ] Add WAL or transactional writes for reliability
- [ ] Create monitoring and alerting setup

---

## Future Ideas & Enhancements
- Advanced embedding models for IM v2 (hybrid approach)
- Multi-language support for NL text
- Real-time signifier updates and subscriptions
- Distributed caching layer
- Graph visualization for signifier relationships
- Automated SHACL shape generation from examples
- Machine learning for ranking weight optimization
- Historical context tracking and temporal reasoning

---

## Completed Archive
Items moved here when done for historical reference.

### Project Setup
- [DONE] Create project structure (2025-11-11)
- [DONE] Set up tracking files (2025-11-11)
