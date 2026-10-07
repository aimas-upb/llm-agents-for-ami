"""The goal context: the environment as a goal structurer needs it, scoped to
one atomic goal.

The goal parser needs to know what can be *done*: which actions each device
offers, what value each one takes, which device property it changes and which
environment variable it moves. That is a small slice of the Thing Descriptions,
and only a slice of the home is usually relevant to one goal, so the view is
built from the discovered graph and cut down before it reaches a prompt.

Two additions are switched on per goal, when its structure has to name them:

    with_properties    what devices report (property affordances), sensors
                       included -- the conditions of a dependency request
    with_environment   each room's environment variables, and which device
                       affordances read them -- ambiguous goals and conditions

Nothing here is a URI. Every element is named by its type and its `td:title`,
which is how the parser refers to it in return; target URLs are resolved later,
against the graph, by whoever builds the plan.

Scoping, per goal (`scope_artifacts`):

    1.   small home (artifacts < max OR actions < max)  -> everything
    2.2  several workspaces hold artifacts -> the rooms named in the goal,
         plus the rooms holding each named device family that none of the
         named rooms has; else everything
    2.1  one workspace holds artifacts -> the device families named in the
         goal, else everything

Rooms and device families are found by plain string matching against the
ontology's `rdfs:label` / `skos:altLabel` -- no LLM, no embedding.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from rdflib import Graph, URIRef
from rdflib.namespace import RDF, RDFS, SKOS

from ....shared.utils.namespaces import (
    HMAS,
    HOMEONT,
    JS,
    QUDT,
    SAREF,
    SSN,
    SOSA,
    TD,
    TDSOSA,
    action_types,
    in_namespace,
    shorten,
)
from .capability_resolution import property_branch

# Fallbacks for the thresholds in `agents.yaml > env_explorer > actuation_context`.
DEFAULT_MAX_ARTIFACTS = 15
DEFAULT_MAX_ACTION_AFFORDANCES = 50


# --------------------------------------------------------------------------
# Graph access
# --------------------------------------------------------------------------

def _title(graph: Graph, node: Any) -> Optional[str]:
    value = graph.value(node, TD.title)
    return str(value) if value is not None else None


def _types_in(graph: Graph, node: Any, namespace) -> List[str]:
    return sorted(shorten(t) for t in graph.objects(node, RDF.type)
                  if in_namespace(t, namespace))


def _label(graph: Graph, cls: Any) -> Optional[str]:
    """A class's English (or untagged) `rdfs:label`."""
    for value in graph.objects(cls, RDFS.label):
        if getattr(value, "language", None) in (None, "en"):
            return str(value)
    return None


def _artifacts(graph: Graph) -> Dict[URIRef, URIRef]:
    """Every artifact, mapped to the workspace that directly contains it."""
    out: Dict[URIRef, URIRef] = {}
    for workspace, artifact in graph.subject_objects(HMAS.contains):
        if (artifact, RDF.type, HMAS.Artifact) in graph:
            out[artifact] = workspace
    return out


def _domain_actions(graph: Graph, artifact: Any) -> List[Any]:
    """Actions typed with what they do. Protocol actions (WebSub, CRUD) carry no
    homeont or SAREF command class, so they never reach the view."""
    actions = [a for a in graph.objects(artifact, TD.hasActionAffordance)
               if action_types(graph.objects(a, RDF.type))]
    return sorted(actions, key=lambda a: _title(graph, a) or "")


def _domain_properties(graph: Graph, artifact: Any) -> List[Any]:
    """Property affordances typed with a homeont class -- what a device reports
    about itself or its surroundings. Untyped ones carry no meaning to match on."""
    props = [p for p in graph.objects(artifact, TD.hasPropertyAffordance)
             if _types_in(graph, p, HOMEONT)]
    return sorted(props, key=lambda p: _title(graph, p) or "")


def _owner(graph: Graph, affordance: Any) -> Optional[Any]:
    """The artifact a property affordance belongs to, if any."""
    for owner in graph.subjects(TD.hasPropertyAffordance, affordance):
        if (owner, RDF.type, HMAS.Artifact) in graph:
            return owner
    return None


# --------------------------------------------------------------------------
# The view
# --------------------------------------------------------------------------

def _schema(graph: Graph, schema: Any) -> Dict[str, Any]:
    """A value schema: its type, unit and description, or its parameters.

    An ObjectSchema is a parameterized call -- the action takes named fields,
    each described the same way, recursively.
    """
    types = _types_in(graph, schema, JS)
    if "js:ObjectSchema" in types:
        parameters = []
        for member in graph.objects(schema, JS.properties):
            name = graph.value(member, JS.propertyName)
            parameters.append({
                "parameter_name": str(name) if name is not None else None,
                "parameter_schema": _schema(graph, member),
            })
        parameters.sort(key=lambda p: str(p["parameter_name"]))
        return {"form": "parameterized call", "parameters": parameters}

    out: Dict[str, Any] = {"form": "direct value",
                           "type": types[0] if types else None}
    unit = graph.value(schema, QUDT.unit)
    if unit is not None:
        out["unit"] = shorten(unit)
    description = graph.value(schema, JS.description)
    if description is not None:
        out["description"] = str(description)
    return out


def _acts_upon(graph: Graph, action: Any) -> List[Dict[str, Any]]:
    out = []
    for target in graph.objects(action, SAREF.actsUpon):
        entry: Dict[str, Any] = {"title": _title(graph, target)}
        classes = _types_in(graph, target, HOMEONT)
        if classes:
            entry["class"] = classes[0]
        comment = graph.value(target, RDFS.comment)
        if comment is not None:
            entry["comment"] = str(comment)
        out.append(entry)
    return sorted(out, key=lambda e: str(e["title"]))


def _environment_effects(graph: Graph, action: Any) -> List[Dict[str, Any]]:
    out = []
    for actuation in graph.objects(action, TDSOSA.hasEffectActuation):
        for prop in graph.objects(actuation, SOSA.actsOnProperty):
            entry: Dict[str, Any] = {"title": _title(graph, prop)}
            classes = _types_in(graph, prop, HOMEONT)
            if classes:
                entry["class"] = classes[0]
            quantity_kind = graph.value(prop, QUDT.hasQuantityKind)
            if quantity_kind is not None:
                entry["quantity_kind"] = shorten(quantity_kind)
            unit = graph.value(prop, QUDT.unit)
            if unit is not None:
                entry["unit"] = shorten(unit)
            environment = graph.value(prop, SSN.isPropertyOf)
            if environment is not None:
                label = graph.value(environment, RDFS.label)
                if label is not None:
                    entry["environment"] = str(label)
            out.append(entry)
    return sorted(out, key=lambda e: str(e["title"]))


def _action(graph: Graph, action: Any) -> Dict[str, Any]:
    entry: Dict[str, Any] = {
        "title": _title(graph, action),
        "command_types": sorted(action_types(graph.objects(action, RDF.type))),
    }
    schema = graph.value(action, TD.hasInputSchema)
    if schema is not None:
        entry["input"] = _schema(graph, schema)
    acts_upon = _acts_upon(graph, action)
    if acts_upon:
        entry["acts_upon"] = acts_upon
    effects = _environment_effects(graph, action)
    if effects:
        entry["environment_effect"] = effects
    return entry


def _property(graph: Graph, affordance: Any,
              written_by: Optional[str] = None) -> Dict[str, Any]:
    """A readable property: what it is, and the shape of the value read.

    An object-valued output lists its fields, which is what a predicate's
    `property_field` names.

    A property an action of the same device writes (`saref:actsUpon`) is shown
    without its values: they are the action's input, already listed, so the
    entry names that action (`written_by`) instead of repeating them.
    """
    classes = [t for t in graph.objects(affordance, RDF.type)
               if in_namespace(t, HOMEONT)]
    cls = sorted(classes, key=str)[0] if classes else None
    entry: Dict[str, Any] = {"title": _title(graph, affordance),
                             "class": shorten(cls) if cls is not None else None}
    # Which tree the class sits in: actuatable / state / capability /
    # observable -- whether reading it is about the device or its surroundings.
    branch = property_branch(graph, cls) if cls is not None else None
    if branch:
        entry["branch"] = branch
    if written_by:
        # The action's input already shows these values; the property only
        # needs to say where.
        entry["written_by"] = written_by
        return entry
    schema = graph.value(affordance, TD.hasOutputSchema)
    if schema is not None:
        entry["output"] = _schema(graph, schema)
        if branch == "capability":
            # What a device *can* be set to ("modes off/low/high/auto are
            # possible") is the planner's feasibility check, not something a
            # request conditions on; its enum gloss is long and stays out. A
            # state or measurement keeps its gloss: a condition's words
            # ("finishes") are matched against it.
            entry["output"].pop("description", None)
    return entry


def _artifact(graph: Graph, artifact: Any,
              with_properties: bool = False) -> Dict[str, Any]:
    homeont = _types_in(graph, artifact, HOMEONT)
    entry: Dict[str, Any] = {
        "title": _title(graph, artifact),
        "hmas_types": _types_in(graph, artifact, HMAS),
        "saref_types": _types_in(graph, artifact, SAREF),
        "homeont_types": homeont,
    }
    if homeont:
        label = _label(graph, URIRef(HOMEONT[homeont[0].split(":", 1)[1]]))
        if label:
            entry["label"] = label
    for key, predicate in (("manufacturer", URIRef("https://schema.org/manufacturer")),
                           ("model", URIRef("https://schema.org/model"))):
        value = graph.value(artifact, predicate)
        if value is not None:
            entry[key] = str(value)
    actions = _domain_actions(graph, artifact)
    entry["actions"] = [_action(graph, a) for a in actions]
    if with_properties:
        # Which action writes each property: the target of its actsUpon is
        # the property affordance itself.
        writers: Dict[Any, str] = {}
        for action in actions:
            for target in graph.objects(action, SAREF.actsUpon):
                writers.setdefault(target, _title(graph, action))
        entry["properties"] = [_property(graph, p, writers.get(p))
                               for p in _domain_properties(graph, artifact)]
    return entry


def _environment(graph: Graph, workspace: Any) -> Optional[Dict[str, Any]]:
    """The environment of a workspace's space, and how each variable is read.

    A variable is keyed by its class. Two graph shapes are covered: SimuHome
    states room-level variables (`environment ssn:hasProperty ?var`); Home
    Assistant has none, and a reading exists only as a device affordance that
    is `ssn:isPropertyOf` the environment. Either way, `observed_by` names the
    device affordances that read the variable -- a room is not a sensor, so
    without one the variable cannot be read in this home.
    """
    for space in graph.objects(workspace, HOMEONT.hasSpace):
        for env in graph.objects(space, HOMEONT.hasEnvironment):
            variables: Dict[str, Dict[str, Any]] = {}

            def variable(cls: str, source: Any, title: Optional[str]) -> Dict[str, Any]:
                entry = variables.setdefault(cls, {"title": title, "class": cls,
                                                   "observed_by": []})
                entry["title"] = entry["title"] or title
                for key, predicate in (("quantity_kind", QUDT.hasQuantityKind),
                                       ("unit", QUDT.unit)):
                    value = graph.value(source, predicate)
                    if value is not None and not entry.get(key):
                        entry[key] = shorten(value)
                return entry

            # Room-level variables (SimuHome), and their observers' affordances.
            # The observer's reading may be typed with a device-specific
            # subclass of the variable's class, hence `rdfs:subClassOf*`.
            for var in graph.objects(env, SSN.hasProperty):
                classes = [t for t in graph.objects(var, RDF.type)
                           if in_namespace(t, HOMEONT)]
                if not classes:
                    continue
                var_class = sorted(classes, key=str)[0]
                entry = variable(shorten(var_class), var, _title(graph, var))
                for artifact in graph.subjects(SOSA.observes, var):
                    for aff in _domain_properties(graph, artifact):
                        if any(var_class in set(graph.transitive_objects(t, RDFS.subClassOf))
                               for t in graph.objects(aff, RDF.type)):
                            entry["observed_by"].append(
                                (_title(graph, artifact), _title(graph, aff)))

            # Device affordances reading the environment directly (both shapes).
            # A reading joins the room variable whose class it specialises;
            # without one (Home Assistant) it stands as a variable of its own.
            for aff in graph.subjects(SSN.isPropertyOf, env):
                owner = _owner(graph, aff)
                classes = _types_in(graph, aff, HOMEONT)
                if owner is None or not classes:
                    continue
                ancestors = {shorten(a) for t in graph.objects(aff, RDF.type)
                             for a in graph.transitive_objects(t, RDFS.subClassOf)}
                key = next((cls for cls in variables if cls in ancestors), classes[0])
                variable(key, aff, None)["observed_by"].append(
                    (_title(graph, owner), _title(graph, aff)))

            for entry in variables.values():
                entry["observed_by"] = [
                    {"artifact": a, "affordance": p}
                    for a, p in sorted(set(entry["observed_by"]), key=str)]
            return {"label": str(graph.value(env, RDFS.label) or ""),
                    "variables": sorted(variables.values(),
                                        key=lambda v: str(v["class"]))}
    return None


def _space_class(graph: Graph, workspace: Any) -> Optional[str]:
    """The room kind of a workspace, via the space it `homeont:hasSpace`."""
    for space in graph.objects(workspace, HOMEONT.hasSpace):
        classes = [t for t in _types_in(graph, space, HOMEONT)
                   if t != "homeont:BuildingSpace"]
        if classes:
            return classes[0]
    return None


def build_actuation_context(graph: Graph,
                            artifacts: Optional[Set[Any]] = None,
                            with_properties: bool = False,
                            with_environment: bool = False) -> Dict[str, Any]:
    """The workspace tree of what can be acted on -- and, on request, read.

    `artifacts` restricts the view to those artifacts (None means all).

    - Plain: only artifacts with a typed action. A sensor has nothing to
      contribute to acting, and is left out.
    - `with_properties`: each artifact also lists its typed property
      affordances, and artifacts that only report (sensors) are kept.
    - `with_environment`: each room also lists its environment variables and
      the affordances that read them.

    A workspace with nothing left beneath it is pruned, except that with the
    environment on and no scoping, a room is kept for its variables alone.
    """
    def relevant(entry: Dict[str, Any]) -> bool:
        return bool(entry["actions"] or (with_properties and entry.get("properties")))

    placement = _artifacts(graph)
    children: Dict[Any, List[Any]] = {}
    contained: Set[Any] = set()
    workspaces = set(graph.subjects(RDF.type, HMAS.Workspace))
    for parent, child in graph.subject_objects(HMAS.contains):
        if parent in workspaces and child in workspaces:
            children.setdefault(parent, []).append(child)
            contained.add(child)

    def node(workspace: Any) -> Optional[Dict[str, Any]]:
        members = []
        for artifact, holder in placement.items():
            if holder != workspace:
                continue
            if artifacts is not None and artifact not in artifacts:
                continue
            entry = _artifact(graph, artifact, with_properties)
            if relevant(entry):
                members.append(entry)
        members.sort(key=lambda a: str(a["title"]))
        subs = [n for n in (node(c) for c in sorted(children.get(workspace, []), key=str))
                if n is not None]
        environment = _environment(graph, workspace) if with_environment else None
        keep_for_environment = bool(environment and environment["variables"]
                                    and artifacts is None)
        if not members and not subs and not keep_for_environment:
            return None
        out: Dict[str, Any] = {"title": _title(graph, workspace)}
        space_class = _space_class(graph, workspace)
        if space_class:
            out["space_class"] = space_class
        if environment and environment["variables"]:
            out["environment"] = environment
        out["artifacts"] = members
        if subs:
            out["workspaces"] = subs
        return out

    roots = sorted(workspaces - contained, key=str)
    return {"workspaces": [n for n in (node(r) for r in roots) if n is not None]}


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------

def _render_schema(schema: Dict[str, Any], indent: str, lines: List[str],
                   direction: str = "input") -> None:
    """An action's input or a property's output.

    An object input is a parameterized call (its fields are parameter names); an
    object output is read whole, and its fields are what a predicate can test.
    """
    if schema.get("form") == "parameterized call":
        shape = "parameterized call" if direction == "input" else "object"
        lines.append(f"{indent}{direction}: {shape}")
        for parameter in schema.get("parameters", []):
            _render_parameter(parameter, indent + "  ", lines)
        return
    lines.append(f"{indent}{direction}: direct value; {_value_facts(schema)}")


def _render_parameter(parameter: Dict[str, Any], indent: str, lines: List[str]) -> None:
    inner = parameter["parameter_schema"]
    if inner.get("form") == "parameterized call":
        lines.append(f"{indent}- {parameter['parameter_name']}: object")
        for sub in inner.get("parameters", []):
            _render_parameter(sub, indent + "  ", lines)
    else:
        lines.append(f"{indent}- {parameter['parameter_name']}: {_value_facts(inner)}")


def _value_facts(schema: Dict[str, Any]) -> str:
    facts = [schema.get("type") or "untyped"]
    if schema.get("unit"):
        facts.append(f"unit: {schema['unit']}")
    if schema.get("description"):
        facts.append(f"description: {schema['description']}")
    return "; ".join(facts)


def _render_property(entry: Dict[str, Any], keys: Iterable[str]) -> str:
    facts = [entry[k] for k in keys if entry.get(k)]
    return f"\"{entry.get('title')}\"" + (f" ({'; '.join(facts)})" if facts else "")


def _render_environment(environment: Dict[str, Any], indent: str,
                        lines: List[str]) -> None:
    lines.append(f"{indent}Environment \"{environment.get('label')}\"")
    for var in environment.get("variables", []):
        facts = [var[k] for k in ("class", "quantity_kind", "unit") if var.get(k)]
        observers = ", ".join(f"{o['artifact']}.{o['affordance']}"
                              for o in var.get("observed_by", [])) or "none"
        title = f"\"{var['title']}\"" if var.get("title") else "(no room-level property)"
        lines.append(f"{indent}  Variable {title}: {'; '.join(facts)}; "
                     f"observed by: {observers}")


def _render_properties(properties: List[Dict[str, Any]], indent: str,
                       lines: List[str]) -> None:
    for prop in properties:
        notes = [prop["branch"]] if prop.get("branch") else []
        if prop.get("written_by"):
            notes.append(f"values as Action \"{prop['written_by']}\"")
        detail = f" ({'; '.join(notes)})" if notes else ""
        lines.append(f"{indent}Property \"{prop.get('title')}\": "
                     f"{prop.get('class')}{detail}")
        if prop.get("output"):
            _render_schema(prop["output"], indent + "  ", lines, direction="output")


def render_actuation_context(context: Dict[str, Any]) -> str:
    """Compact indented text for a prompt: one fact per line, no URIs."""
    lines: List[str] = []

    def workspace(ws: Dict[str, Any], indent: str) -> None:
        kind = f" ({ws['space_class']})" if ws.get("space_class") else ""
        lines.append(f"{indent}Workspace \"{ws.get('title')}\"{kind}")
        if ws.get("environment"):
            _render_environment(ws["environment"], indent + "  ", lines)
        for art in ws.get("artifacts", []):
            types = art["homeont_types"] + art["saref_types"] + art["hmas_types"]
            label = f" \"{art['label']}\"" if art.get("label") else ""
            lines.append(f"{indent}  Artifact \"{art.get('title')}\"{label}: "
                         f"{', '.join(types)}")
            made = [f"{k}: {art[k]}" for k in ("manufacturer", "model") if art.get(k)]
            if made:
                lines.append(f"{indent}    {'; '.join(made)}")
            for action in art.get("actions", []):
                lines.append(f"{indent}    Action \"{action.get('title')}\": "
                             f"{', '.join(action['command_types'])}")
                if action.get("input"):
                    _render_schema(action["input"], indent + "      ", lines)
                for target in action.get("acts_upon", []):
                    lines.append(f"{indent}      acts upon: "
                                 f"{_render_property(target, ('class', 'comment'))}")
                for effect in action.get("environment_effect", []):
                    where = (f" of \"{effect['environment']}\""
                             if effect.get("environment") else "")
                    lines.append(
                        f"{indent}      environment effect: "
                        f"{_render_property(effect, ('class', 'quantity_kind', 'unit'))}"
                        f"{where}")
            _render_properties(art.get("properties", []), indent + "    ", lines)
        for sub in ws.get("workspaces", []):
            workspace(sub, indent + "  ")

    for ws in context.get("workspaces", []):
        workspace(ws, "")
    return "\n".join(lines) if lines else "No actuatable devices found."


# --------------------------------------------------------------------------
# Label matching
# --------------------------------------------------------------------------

def _tokens(text: str) -> List[str]:
    """Lowercased word tokens, a trailing plural `s` folded ("blinds" -> "blind")."""
    out = []
    for token in re.split(r"[^a-z0-9]+", text.lower()):
        if not token:
            continue
        if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
            token = token[:-1]
        out.append(token)
    return out


def match_classes(text: str, labels_by_class: Dict[str, List[str]]) -> List[str]:
    """The classes whose labels the text names, longest match first.

    Each label is matched as a whole phrase on word boundaries. Overlapping
    matches keep the longest span, so "master bedroom" names MasterBedroom and
    not also Bedroom. Several classes matching the same span ("light": On/Off
    Light and Dimmable Light) are all kept.
    """
    matched: List[str] = []
    for group in match_class_groups(text, labels_by_class):
        matched.extend(cls for cls in group if cls not in matched)
    return matched


@dataclass(frozen=True)
class _Span:
    """A run of words in the text: positions `start` up to, not including, `end`."""

    start: int
    end: int

    @property
    def length(self) -> int:
        return self.end - self.start

    def overlaps(self, other: "_Span") -> bool:
        return self.start < other.end and other.start < self.end


def _occurrences(words: List[str], phrase: List[str]) -> List[_Span]:
    """Every place `phrase` appears in `words`, as whole words."""
    size = len(phrase)
    return [_Span(start, start + size)
            for start in range(len(words) - size + 1)
            if words[start:start + size] == phrase]


def match_class_groups(text: str,
                       labels_by_class: Dict[str, List[str]]) -> List[List[str]]:
    """Like `match_classes`, but one group per mention in the text.

    The classes one phrase matches belong together: "the kitchen light" names
    one light, which may be an On/Off or a Dimmable Light -- not one of each.
    Groups come back in the order their mentions appear in the text.
    """
    words = _tokens(text)

    # 1. Every place any label of any class occurs in the text.
    hits: List[Tuple[_Span, str]] = []
    for cls, labels in labels_by_class.items():
        for label in labels:
            phrase = _tokens(label)
            if phrase:
                hits.extend((span, cls) for span in _occurrences(words, phrase))

    # 2. Longest hits first, so that when two overlap -- "master bedroom" and
    #    "bedroom" -- the longer one claims the words and the shorter is dropped.
    hits.sort(key=lambda hit: (-hit[0].length, hit[0].start))

    # 3. One group per claimed span. A hit on a span already claimed joins that
    #    span's group ("light" for both On/Off and Dimmable Light); a hit that
    #    only partly overlaps a claimed span is discarded.
    groups: Dict[_Span, List[str]] = {}
    for span, cls in hits:
        if span not in groups:
            if any(span.overlaps(claimed) for claimed in groups):
                continue
            groups[span] = []
        if cls not in groups[span]:
            groups[span].append(cls)

    return [groups[span] for span in sorted(groups, key=lambda s: s.start)]


def _english(graph: Graph, node: Any, predicate: Any) -> List[str]:
    return [str(v) for v in graph.objects(node, predicate)
            if getattr(v, "language", None) in (None, "en")]


def _labels_of(graph: Graph, classes: Iterable[Any],
               head_noun: bool = False) -> Dict[str, List[str]]:
    """`rdfs:label` and `skos:altLabel` phrases per class.

    With `head_noun`, a multi-word `rdfs:label` also contributes its last word
    ("Laundry Washer" -> "washer"): the head noun names the thing, while a
    modifier ("air" in Air Purifier) would also name its neighbours. Alt labels
    never do -- "washing machine" must not make "machine" a device name.
    """
    out: Dict[str, List[str]] = {}
    for cls in classes:
        labels = _english(graph, cls, RDFS.label)
        if head_noun:
            labels += [words[-1] for words in (label.split() for label in labels)
                       if len(words) > 1]
        labels += _english(graph, cls, SKOS.altLabel)
        if labels:
            out[shorten(cls)] = labels
    return out


def _homeont_subclasses(graph: Graph, root: Any) -> List[Any]:
    return [c for c in graph.transitive_subjects(RDFS.subClassOf, root)
            if c != root and in_namespace(c, HOMEONT)]


def space_labels(graph: Graph) -> Dict[str, List[str]]:
    """Labels of every room kind (`homeont:BuildingSpace` subclass)."""
    return _labels_of(graph, _homeont_subclasses(graph, HOMEONT.BuildingSpace))


def device_family_labels(graph: Graph) -> Dict[str, List[str]]:
    """Labels of every homeont device family (below `saref:Device`), head
    nouns included."""
    return _labels_of(graph, _homeont_subclasses(graph, SAREF.Device),
                      head_noun=True)


# --------------------------------------------------------------------------
# Scoping
# --------------------------------------------------------------------------

def _is_a(graph: Graph, node: Any, curies: Iterable[str]) -> bool:
    wanted = {URIRef(HOMEONT[c.split(":", 1)[1]]) for c in curies}
    for cls in graph.objects(node, RDF.type):
        if wanted & set(graph.transitive_objects(cls, RDFS.subClassOf)):
            return True
    return False


def scope_artifacts(graph: Graph, goal_text: Optional[str],
                    max_artifacts: int = DEFAULT_MAX_ARTIFACTS,
                    max_actions: int = DEFAULT_MAX_ACTION_AFFORDANCES,
                    with_properties: bool = False,
                    ) -> Tuple[Optional[Set[Any]], Dict[str, Any]]:
    """Which artifacts the view for this goal should hold.

    Returns `(artifacts, scope)`. `artifacts` is None for the whole
    environment; `scope` records the rule that decided, for the log.

    A match that selects nothing relevant -- a room this home does not have, or
    only sensors in a view without properties -- counts as no match, so it
    falls through rather than producing an empty view. With properties, a
    sensor is relevant: a goal's condition may name it.
    """
    def _actuatable(candidates: Iterable[Any]) -> Set[Any]:
        return {a for a in candidates
                if _domain_actions(graph, a)
                or (with_properties and _domain_properties(graph, a))}

    placement = _artifacts(graph)
    action_count = sum(len(_domain_actions(graph, a)) for a in placement)
    scope: Dict[str, Any] = {"artifacts": len(placement),
                             "action_affordances": action_count}

    if (not goal_text or len(placement) < max_artifacts
            or action_count < max_actions):
        scope["rule"] = "1-whole"
        return None, scope

    holders = set(placement.values())
    mentions = match_class_groups(goal_text, device_family_labels(graph))
    families = [f for group in mentions for f in group]
    families = sorted(set(families), key=families.index)
    # The devices each mention can refer to, across all the families it matched.
    devices_of = [_actuatable(a for a in placement if _is_a(graph, a, group))
                  for group in mentions]

    if len(holders) == 1:
        devices = set().union(*devices_of)
        if devices:
            scope.update(rule="2.1-device", matched_devices=families)
            return devices, scope
        scope["rule"] = "2-no-match-whole"
        return None, scope

    # Several workspaces: the rooms the goal names, plus the rooms holding the
    # devices it names. A goal can reference more than one place ("keep the
    # living room at 22 ... then set the heat pump"), and a room match alone
    # would drop a device that stands elsewhere.
    #
    # A device mention that a named room can satisfy is taken from there only:
    # in "the kitchen light" the room qualifies the light, and lights stand in
    # every room. A mention no named room can satisfy brings in every room
    # that has such a device. The check is per mention, not per family: "light"
    # matches several families, and the kitchen needs to have only one of them.
    rooms = match_classes(goal_text, space_labels(graph))
    named = {ws for ws in holders
             if any(_is_a(graph, space, rooms)
                    for space in graph.objects(ws, HOMEONT.hasSpace))} if rooms else set()
    workspaces = set(named)
    for devices in devices_of:
        located = {placement[a] for a in devices}
        if not located & named:
            workspaces |= located
    chosen = _actuatable(a for a, ws in placement.items() if ws in workspaces)
    if chosen:
        scope.update(rule="2.2-union", matched_rooms=rooms, matched_devices=families,
                     workspaces=sorted(_title(graph, ws) or str(ws)
                                       for ws in workspaces))
        return chosen, scope

    scope["rule"] = "2-no-match-whole"
    return None, scope
