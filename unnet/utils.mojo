"""Utility functions for visualization and graph traversal."""

from .grad import Edge, Node, Op, get_global_registry_copy
from python import Python, PythonObject

# Helper functions for graph traversal and visualization


fn get_node_id(node: Node) -> String:
    """Get a unique string identifier for any node variant."""
    return String(node.uuid)


fn get_node_data(node: Node) -> Tuple[String, Float64, Float64]:
    """Extract name, value, and grad from a node.

    Args:
        node: The node to extract data from.

    Returns:
        A tuple of (name, value, grad).
    """
    return (node.name, node.value, node.grad)


fn walk(root: Node) raises -> Tuple[List[Node], List[Edge]]:
    """Walk the computation graph and collect nodes and edges.

    Uses the global registry to look up parent nodes by UUID.

    Args:
        root: The root node to start traversal from.

    Returns:
        A tuple of (nodes, edges) where nodes is a list of Node objects
        and edges is a list of (parent, child) tuples representing connections.
    """
    var nodes = List[Node]()
    var edges = List[Edge]()

    # Get the global registry to look up parents
    var registry = get_global_registry_copy()

    # Stack for traversal using UUIDs
    var stack = List[UUID]()
    stack.append(root.uuid)

    # Track visited UUIDs
    var visited = List[UUID]()

    while len(stack) > 0:
        var current_uuid = stack.pop()

        # Skip if already visited
        if current_uuid in visited:
            continue

        # Look up the node in the registry
        if current_uuid not in registry:
            continue

        var current = registry[current_uuid]
        visited.append(current_uuid)
        nodes.append(current)

        # Process parents using their UUIDs
        if current.has_parent1:
            var parent1_uuid = current.parent1_uuid
            if parent1_uuid in registry:
                var parent1 = registry[parent1_uuid]
                edges.append((parent1, current))
                stack.append(parent1_uuid)

        if current.has_parent2:
            var parent2_uuid = current.parent2_uuid
            if parent2_uuid in registry:
                var parent2 = registry[parent2_uuid]
                edges.append((parent2, current))
                stack.append(parent2_uuid)

    return nodes^, edges^
