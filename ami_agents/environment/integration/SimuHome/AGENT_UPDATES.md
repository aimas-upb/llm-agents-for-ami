# Agent-layer updates to make later

Things found while pointing the agents at SHTD that belong in the agent layer,
not in SHTD. **None of these are done** — recorded here so they are not
rediscovered.

---

## 1. `affordance_id` is overloaded: identity and URL at once

**Status: latent. Nothing is broken today; it constrains what an id may be.**

`affordance_id` is assigned `form.href` at discovery
(`integration_engine.py::extract_*_affordances`), so an affordance's identity IS
its invocation URL. Several places then rely on that coincidence:

| where | what it does |
|---|---|
| `integration_engine.py:1341` | `property_uri = affordance.affordance_id or href` — keys the state dict by id, treating it as the property's URI |
| `bt_planner.py:531,535` | derives a human node name by splitting the id on `/` and taking the last segment |

**Why it should change.** A name should come from `td:name`, and an invocation
target from `hctl:hasTarget`. Both are already carried separately — the
capabilities payload emits `name` and `target` as distinct fields
(`data_formatting.py:227-236`), and the planner's own resolver uses
`aff.get("target")` correctly. So the pieces are in place; these two spots just
take a shortcut.

The name-derivation shortcut is the more visible one: it produces `OnOff` from
`…/actions/onOff` by luck of the URL ending in the affordance name. Under the
flat naming scheme that holds, but it is coincidence, not design — and it only
fires when the LLM omits `name`, so it is a fallback path.

**Constraint this imposes right now.** Any affordance whose id is not a
dereferenceable URL must be kept out of `thing_description.actions` /
`.properties` / `.events`. That is why the infrastructure affordances (see §2)
are flagged and filtered rather than listed. Verified: 0 planner-facing or
property affordances have an id differing from `form.href`, and 0 synthetic ids
are reachable from any ThingDescription list.

**Fix.** Have `_fetch_initial_state` key on `form.href` explicitly, and have the
node-name fallback read `td:name` from the affordance index instead of parsing
the id. Then `affordance_id` is free to be an opaque identifier.

---

## 2. Infrastructure affordances: retained-and-flagged, not dropped

**Status: DONE this session, in `integration_engine.py`. Noted because it
changed shared agent code.**

`EXCLUDED_ACTION_NAMES` (subscribe/unsubscribe, representation CRUD) were
*dropped* at discovery, which made subscription impossible: `subscribe_to_artifact`
searched `affordance_map` for an affordance that had been deleted on the way in.
0 of 22 artifacts could subscribe.

They are now recorded with `metadata["infrastructure"] = True` and filtered out
of `thing_description.actions`, so a planner still never sees them.

A second, latent bug surfaced doing this: every artifact's subscribe AND
unsubscribe affordance share one form href (`/hub/`), and `affordance_map` is
keyed by href — so **one entry survived for the whole environment**. This affects
any WebSub environment, HASP included. Infrastructure affordances now get a
synthetic `urn:hmas:infrastructure:<name>:<artifact-iri>` key. 44 retained (22
artifacts × 2) where 1 was.

---

## 3. `subscribe_to_artifact` matches on RDF type

**Status: DONE this session.**

Was: look for an action named `focus`, then an *event* named `subscribe`, then an
action named `subscribe`. Now:

1. **CArtAgO focus** — `{artifactName, callbackUrl}` (JaCaMo environments)
2. **`websub:subscribeToArtifact`** matched on `semantic_types`, POSTing
   `{hub.mode, hub.topic, hub.callback}` to the affordance's `hctl:hasTarget`,
   with the artifact's own IRI as the topic
3. name-based lookup, for environments predating the typed form

Matching on the RDF type rather than the name is the point: the type is what the
vocabulary guarantees.

Result: **22/22 artifacts subscribe**, and notifications reach the listener queue.

---

## 4. `main.py` ignores `current_environment`

**Status: not done.**

`environment.yaml` has `current_environment: "${CURRENT_ENVIRONMENT:-lab308}"`,
but `main.py:141-144` takes **the first environment in the dict** and never reads
that key:

```python
first_env = list(environments.values())[0]
yggdrasil_url = first_env.get("yggdrasil_url", ...)
```

So `CURRENT_ENVIRONMENT=homebench` has no effect, and pointing the agents at a
different environment means editing `LAB308_YGGDRASIL_URL` — which is misleading.

---

## 5. `IntegrationEngineFactory.create_engine` is a stub

**Status: not done.**

`integration_engine.py:1730` is `pass` with a TODO. `environment.yaml` still says
`environment_type: 'homeassistant'`, which is read by nothing —
`env_explorer_agent.py:86` constructs `YggdrasilIntegration` directly. Either
implement the factory or drop the config key; as it stands the config claims a
choice that does not exist.
