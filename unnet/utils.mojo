"""Utility functions for visualization and graph traversal."""

from .grad import (
    Node,
    Op,
    AnyNode,
    LeafNode,
    AddNode,
    SubNode,
    MulNode,
    PowNode,
    TanhNode,
    RawNode,
)
from python import Python, PythonObject

# Helper functions for graph traversal and visualization


fn get_node_id(node: AnyNode) -> String:
    """Get a unique string identifier for any node variant."""
    if node.isa[LeafNode]():
        var n = node[LeafNode]
        return "leaf_" + n.name + "_" + String(n.value)
    elif node.isa[AddNode]():
        var n = node[AddNode]
        return "add_" + n.name + "_" + String(n.value)
    elif node.isa[SubNode]():
        var n = node[SubNode]
        return "sub_" + n.name + "_" + String(n.value)
    elif node.isa[MulNode]():
        var n = node[MulNode]
        return "mul_" + n.name + "_" + String(n.value)
    elif node.isa[PowNode]():
        var n = node[PowNode]
        return "pow_" + n.name + "_" + String(n.value)
    else:  # TanhNode
        var n = node[TanhNode]
        return "tanh_" + n.name + "_" + String(n.value)


fn get_op_string(node: AnyNode) -> String:
    """Get the operation string for any node variant."""
    if node.isa[LeafNode]():
        return ""
    elif node.isa[AddNode]():
        return "+"
    elif node.isa[SubNode]():
        return "-"
    elif node.isa[MulNode]():
        return "*"
    elif node.isa[PowNode]():
        return "^"
    else:  # TanhNode
        return "tanh"


fn get_parents(node: AnyNode) -> List[AnyNode]:
    """Get the parent nodes from any node variant."""
    if node.isa[LeafNode]():
        return node[LeafNode].parents.copy()
    elif node.isa[AddNode]():
        return node[AddNode].parents.copy()
    elif node.isa[SubNode]():
        return node[SubNode].parents.copy()
    elif node.isa[MulNode]():
        return node[MulNode].parents.copy()
    elif node.isa[PowNode]():
        return node[PowNode].parents.copy()
    else:  # TanhNode
        return node[TanhNode].parents.copy()


fn is_in_list(node_id: String, visited: List[String]) -> Bool:
    """Check if a node ID is in the visited list."""
    for i in range(len(visited)):
        if visited[i] == node_id:
            return True
    return False


fn get_node_data(node: AnyNode) -> Tuple[String, Float64, Float64]:
    """Extract name, value, and grad from any node variant.

    Args:
        node: The node to extract data from.

    Returns:
        A tuple of (name, value, grad).
    """
    if node.isa[LeafNode]():
        var n = node[LeafNode]
        return (n.name, n.value, n.grad)
    elif node.isa[AddNode]():
        var n = node[AddNode]
        return (n.name, n.value, n.grad)
    elif node.isa[SubNode]():
        var n = node[SubNode]
        return (n.name, n.value, n.grad)
    elif node.isa[MulNode]():
        var n = node[MulNode]
        return (n.name, n.value, n.grad)
    elif node.isa[PowNode]():
        var n = node[PowNode]
        return (n.name, n.value, n.grad)
    else:  # TanhNode
        var n = node[TanhNode]
        return (n.name, n.value, n.grad)


fn walk[
    op: Op = Op.NONE
](root: Node[op]) -> Tuple[List[RawNode], List[Tuple[String, String]]]:
    """Walk the computation graph and collect nodes and edges.

    Parameters:
        op: Operation type of the root node.

    Args:
        root: The root node to start traversal from.

    Returns:
        A tuple of (nodes, edges) where nodes is a list of raw node data (name, value, grad)
        and edges is a list of (parent_id, child_id) tuples representing connections.
    """
    var nodes = List[RawNode]()
    var edges = List[Tuple[String, String]]()
    var visited = List[String]()

    # Internal stack for traversal
    var stack = List[AnyNode]()
    stack.append(root)

    while len(stack) > 0:
        var current = stack.pop()
        var node_id = get_node_id(current)

        # Skip if already visited
        if is_in_list(node_id, visited):
            continue

        visited.append(node_id)
        var node_data = get_node_data(current)
        nodes.append(node_data)

        # Process parents
        var parents = get_parents(current)
        for i in range(len(parents)):
            var parent = parents[i]
            var parent_id = get_node_id(parent)
            edges.append((parent_id, node_id))
            stack.append(parent)

    return nodes^, edges^


fn draw[op: Op = Op.NONE](graph: Node[op]) raises -> PythonObject:
    """Create a graphviz visualization of the computation graph.

    Parameters:
        op: Operation type of the root node.

    Args:
        graph: The root node of the computation graph to visualize.

    Returns:
        A graphviz Digraph object that can be rendered or displayed.
    """
    var Digraph = Python.import_module("graphviz").Digraph
    var plot = Digraph(format="svg", graph_attr={"rankdir": "LR"})

    # Internal traversal using AnyNode
    var node_map = Dict[String, Tuple[String, Float64, Float64, String]]()
    var visited = List[String]()
    var stack = List[AnyNode]()
    stack.append(graph)

    while len(stack) > 0:
        var current = stack.pop()
        var node_id = get_node_id(current)

        if is_in_list(node_id, visited):
            continue

        visited.append(node_id)
        var node_data = get_node_data(current)
        var name = node_data[0]
        var value = node_data[1]
        var grad = node_data[2]
        var op_str = get_op_string(current)
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
        var parents = get_parents(current)
        for i in range(len(parents)):
            var parent = parents[i]
            var parent_id = get_node_id(parent)

            # Connect parent to this node's operation (if it has one)
            if len(op_str) > 0:
                plot.edge(parent_id, node_id + "_op")
            else:
                plot.edge(parent_id, node_id)

            stack.append(parent)

    return plot
