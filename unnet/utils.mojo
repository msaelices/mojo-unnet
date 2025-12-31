"""Utility functions for visualization and graph traversal."""

import os
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


fn walk(root: Node) -> Tuple[List[Node], List[Edge]]:
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

        # Look up the node in the registry (returns Optional)
        var current_opt = registry.get(current_uuid)
        if current_opt == None:
            continue

        var current = current_opt.value()
        visited.append(current_uuid)
        nodes.append(current)

        # Process parents using their UUIDs
        if current.has_parent1:
            var parent1_uuid = current.parent1_uuid
            var parent1_opt = registry.get(parent1_uuid)
            if parent1_opt != None:
                var parent1 = parent1_opt.value()
                edges.append((parent1, current))
                stack.append(parent1_uuid)

        if current.has_parent2:
            var parent2_uuid = current.parent2_uuid
            var parent2_opt = registry.get(parent2_uuid)
            if parent2_opt != None:
                var parent2 = parent2_opt.value()
                edges.append((parent2, current))
                stack.append(parent2_uuid)

    return nodes^, edges^


fn draw(var graph: Node) raises -> PythonObject:
    """Create a graphviz visualization of the computation graph.

    Args:
        graph: The root node of the computation graph to visualize.

    Returns:
        A graphviz Digraph object that can be rendered or displayed.

    Note:
        Returns an empty PythonObject if graphviz is not installed.
        Install with: pip install graphviz
    """
    # Import graphviz module
    var graphviz_module = Python.import_module("graphviz")

    var Digraph = graphviz_module.Digraph
    var plot = Digraph()

    # Get the global registry to look up nodes
    var registry = get_global_registry_copy()

    # Track visited UUIDs
    var visited = List[UUID]()
    var stack = List[UUID]()
    stack.append(graph.uuid)

    while len(stack) > 0:
        var current_uuid = stack.pop()

        # Skip if already visited
        if current_uuid in visited:
            continue

        # Look up the node in the registry
        var current_opt = registry.get(current_uuid)
        if current_opt == None:
            continue

        var current = current_opt.value()
        visited.append(current_uuid)

        var node_id = get_node_id(current)
        var node_data = get_node_data(current)
        var name = node_data[0]
        var value = node_data[1]
        var grad = node_data[2]
        var op = current.op

        # Draw this node
        var label = String(name, " | v: ", value, " | g: ", grad)
        plot.node(node_id, label, "record")

        # If node has an operation, add operation node and connect it
        if op:
            var op_node_id = node_id + "_op"
            plot.node(op_node_id, String(op), "circle")
            plot.edge(op_node_id, node_id)

        # Process parents and create edges using UUIDs
        if current.has_parent1:
            var parent1_uuid = current.parent1_uuid
            var parent1_opt = registry.get(parent1_uuid)
            if parent1_opt != None:
                var parent1_id = String(parent1_uuid)
                if op:
                    plot.edge(parent1_id, node_id + "_op")
                else:
                    plot.edge(parent1_id, node_id)
                stack.append(parent1_uuid)

        if current.has_parent2:
            var parent2_uuid = current.parent2_uuid
            var parent2_opt = registry.get(parent2_uuid)
            if parent2_opt != None:
                var parent2_id = String(parent2_uuid)
                if op:
                    plot.edge(parent2_id, node_id + "_op")
                else:
                    plot.edge(parent2_id, node_id)
                stack.append(parent2_uuid)

    return plot
