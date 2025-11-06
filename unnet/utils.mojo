"""Utility functions for visualization and graph traversal."""

from .grad import Node


fn walk(node: Node) -> (List[Node], List[Tuple[Node, Node]]):
    """Walk the computation graph and collect nodes and edges."""
    # TODO: Implement graph traversal
    var nodes = List[Node]()
    var edges = List[Tuple[Node, Node]]()
    return (nodes, edges)


fn visualize(node: Node) -> String:
    """Generate a visualization of the computation graph."""
    # TODO: Implement graph visualization (maybe export to DOT format)
    return "graph { }"
