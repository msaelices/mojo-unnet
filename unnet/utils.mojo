"""Utility functions for visualization and graph traversal."""

from .grad import Edge, Node, Op
from python import Python, PythonObject

# Helper functions for graph traversal and visualization

# Note: The walk and draw functions are temporarily disabled due to type inference
# issues with UUID in List/Dict. They will be fixed in a future update.


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
