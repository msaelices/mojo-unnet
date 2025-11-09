"""Utility functions for visualization and graph traversal."""

from .grad import Node, Op
from python import Python, PythonObject

# Helper functions for graph traversal and visualization


fn get_node_id(node: Node.AnyNode) -> String:
    """Get a unique string identifier for any node variant."""
    if node.isa[Node.LeafNode]():
        var n = node[Node.LeafNode]
        return "leaf_" + n.name + "_" + String(n.value)
    elif node.isa[Node.AddNode]():
        var n = node[Node.AddNode]
        return "add_" + n.name + "_" + String(n.value)
    elif node.isa[Node.SubNode]():
        var n = node[Node.SubNode]
        return "sub_" + n.name + "_" + String(n.value)
    elif node.isa[Node.MulNode]():
        var n = node[Node.MulNode]
        return "mul_" + n.name + "_" + String(n.value)
    elif node.isa[Node.PowNode]():
        var n = node[Node.PowNode]
        return "pow_" + n.name + "_" + String(n.value)
    else:  # TanhNode
        var n = node[Node.TanhNode]
        return "tanh_" + n.name + "_" + String(n.value)


fn get_op_string(node: Node.AnyNode) -> String:
    """Get the operation string for any node variant."""
    if node.isa[Node.LeafNode]():
        return ""
    elif node.isa[Node.AddNode]():
        return "+"
    elif node.isa[Node.SubNode]():
        return "-"
    elif node.isa[Node.MulNode]():
        return "*"
    elif node.isa[Node.PowNode]():
        return "^"
    else:  # TanhNode
        return "tanh"


fn get_parent[i: Int](node: Node.AnyNode) -> Optional[Node.AnyNode]:
    """Get the parent nodes from any node variant."""
    if node.isa[Node.LeafNode]():
        return node[Node.LeafNode].get_parent[i]()
    elif node.isa[Node.AddNode]():
        return node[Node.AddNode].get_parent[i]()
    elif node.isa[Node.SubNode]():
        return node[Node.SubNode].get_parent[i]()
    elif node.isa[Node.MulNode]():
        return node[Node.MulNode].get_parent[i]()
    elif node.isa[Node.PowNode]():
        return node[Node.PowNode].get_parent[i]()
    else:  # TanhNode
        return node[Node.TanhNode].get_parent[i]()


fn is_in_list(node_id: String, visited: List[String]) -> Bool:
    """Check if a node ID is in the visited list."""
    for i in range(len(visited)):
        if visited[i] == node_id:
            return True
    return False


fn get_node_data(node: Node.AnyNode) -> Node.RawNode:
    """Extract name, value, and grad from any node variant.

    Args:
        node: The node to extract data from.

    Returns:
        A tuple of (name, value, grad).
    """
    if node.isa[Node.LeafNode]():
        var n = node[Node.LeafNode]
        return (n.name, n.value, n.grad)
    elif node.isa[Node.AddNode]():
        var n = node[Node.AddNode]
        return (n.name, n.value, n.grad)
    elif node.isa[Node.SubNode]():
        var n = node[Node.SubNode]
        return (n.name, n.value, n.grad)
    elif node.isa[Node.MulNode]():
        var n = node[Node.MulNode]
        return (n.name, n.value, n.grad)
    elif node.isa[Node.PowNode]():
        var n = node[Node.PowNode]
        return (n.name, n.value, n.grad)
    else:  # TanhNode
        var n = node[Node.TanhNode]
        return (n.name, n.value, n.grad)


fn walk[
    op: Op = Op.NONE
](root: Node[op]) -> Tuple[List[Node.RawNode], List[Tuple[String, String]]]:
    """Walk the computation graph and collect nodes and edges.

    Parameters:
        op: Operation type of the root node.

    Args:
        root: The root node to start traversal from.

    Returns:
        A tuple of (nodes, edges) where nodes is a list of raw node data (name, value, grad)
        and edges is a list of (parent_id, child_id) tuples representing connections.
    """
    var nodes = List[Node.RawNode]()
    var edges = List[Tuple[String, String]]()
    var visited = List[String]()

    # Internal stack for traversal
    var stack = List[Node.AnyNode]()
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
        ref parent1, parent2 = get_parent[0](current), get_parent[1](current)
        if parent1:
            var parent1_id = get_node_id(parent1)
            edges.append((parent1_id, node_id))
            stack.append(parent1)
        if parent2:
            var parent2_id = get_node_id(parent2)
            edges.append((parent2_id, node_id))
            stack.append(parent2)

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

    # Internal traversal using Node.AnyNode
    var node_map = Dict[String, Tuple[String, Float64, Float64, String]]()
    var visited = List[String]()
    var stack = List[Node.AnyNode]()
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
        ref parent1, parent2 = get_parent[0](current), get_parent[1](current)
        if parent1:
            var parent1_id = get_node_id(parent1)
            if len(op_str) > 0:
                plot.edge(parent1_id, node_id + "_op")
            else:
                plot.edge(parent1_id, node_id)
            stack.append(parent1)
        if parent2:
            var parent2_id = get_node_id(parent2)
            if len(op_str) > 0:
                plot.edge(parent2_id, node_id + "_op")
            else:
                plot.edge(parent2_id, node_id)
            stack.append(parent2)

    return plot
