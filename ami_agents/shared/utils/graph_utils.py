from rdflib import Graph, URIRef
from rdflib.namespace import RDF
from rdflib.term import BNode

def extract_subgraph(graph: Graph, root_node: URIRef) -> Graph:
    """Extracts the subgraph reachable from the given root node, including blank nodes."""
    subgraph = Graph()
    visited = set()
    to_visit = [root_node]

    while to_visit:
        current_node = to_visit.pop()
        if current_node in visited:
            continue
        visited.add(current_node)

        for predicate, obj in graph.predicate_objects(current_node):
            subgraph.add((current_node, predicate, obj))
            if isinstance(obj, URIRef) and obj not in visited:
                to_visit.append(obj)
            elif isinstance(obj, BNode) and obj not in visited:
                to_visit.append(obj)
                
    return subgraph