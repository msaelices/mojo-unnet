"""Utility functions for visualization and graph traversal."""

from .grad import Edge, Node, Op
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


fn walk[op: Op = Op.NONE](root: Node) -> Tuple[List[Node], List[Edge]]:
    """Walk the computation graph and collect nodes and edges.

    Parameters:
        op: Operation type of the root node.

    Args:
        root: The root node to start traversal from.

    Returns:
        A tuple of (nodes, edges) where nodes is a list of raw node data (name, value, grad)
        and edges is a list of (parent_id, child_id) tuples representing connections.
    """
    var nodes = List[Node]()
    var edges = List[Edge]()
    var visited = List[Node]()

    # Internal stack for traversal
    var stack = List[Node]()
    stack.append(root)

    while len(stack) > 0:
        var current = stack.pop()

        # Skip if already visited
        if current in visited:
            continue

        visited.append(current)
        nodes.append(current)

        # Process parents
        ref parent1, parent2 = current.get_parent[0](), current.get_parent[1]()
        if parent1:
            edges.append((parent1.value(), current))
            stack.append(parent1.value())
        if parent2:
            edges.append((parent2.value(), current))
            stack.append(parent2.value())

    return nodes^, edges^


fn draw(var graph: Node) raises -> PythonObject:
    """Create a graphviz visualization of the computation graph.

    Args:
        graph: The root node of the computation graph to visualize.

    Returns:
        A graphviz Digraph object that can be rendered or displayed.
    """
    var Digraph = Python.import_module("graphviz").Digraph
    var plot = Digraph(format="svg", graph_attr={"rankdir": "LR"})

    # Internal traversal using Node objects
    var node_map = Dict[String, Tuple[String, Float64, Float64, String]]()
    var visited = List[Node]()
    var stack = List[Node]()
    stack.append(graph^)

    while len(stack) > 0:
        var current = stack.pop()
        var node_id = get_node_id(current)

        if current in visited:
            continue

        var node_data = get_node_data(current)
        var name = node_data[0]
        var value = node_data[1]
        var grad = node_data[2]
        var op_str = String(current.op)
        node_map[node_id] = (name, value, grad, op_str)

        # Draw this node
        var label = String(name, " | v: ", value, " | g: ", grad)
        plot.node(name=node_id, label=label, shape="record")

        # If node has an operation, add operation node and connect it
        if len(op_str) > 0:
            var op_node_id = node_id + "_op"
            plot.node(name=op_node_id, label=op_str, shape="circle")
            plot.edge(op_node_id, node_id)

        # Process parents and create edges
        ref parent1 = current.get_parent[0]()
        ref parent2 = current.get_parent[1]()
        if parent1:
            var parent1_node = parent1.value()
            var parent1_id = get_node_id(parent1_node)
            if len(op_str) > 0:
                plot.edge(parent1_id, node_id + "_op")
            else:
                plot.edge(parent1_id, node_id)
            stack.append(parent1_node)
        if parent2:
            var parent2_node = parent2.value()
            var parent2_id = get_node_id(parent2_node)
            if len(op_str) > 0:
                plot.edge(parent2_id, node_id + "_op")
            else:
                plot.edge(parent2_id, node_id)
            stack.append(parent2_node)

        visited.append(current^)

    return plot
