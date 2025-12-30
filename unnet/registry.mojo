"""Global node registry for automatic differentiation.

This module provides a global registry that tracks all Node instances
created during computation graph construction. The registry is automatically
populated when nodes are created and can be used for backpropagation.
"""

from sys.ffi import _Global
from memory import UnsafePointer
from unnet.uuid import UUID
from unnet.grad import Node


struct _NodeRegistry(Copyable):
    """Global registry of all nodes in the computation graph.

    This struct maintains a dictionary mapping UUIDs to Nodes, allowing
    O(1) lookup during backpropagation without requiring manual registration.
    """

    var _registry: Dict[UUID, Node]

    fn __init__(out self):
        """Initialize an empty registry."""
        self._registry = Dict[UUID, Node]()

    fn register(mut self, node: Node):
        """Register a node in the global registry.

        Args:
            node: The node to register.
        """
        self._registry[node.uuid] = node

    fn get_registry_copy(self) raises -> Dict[UUID, Node]:
        """Get a copy of the registry.

        Returns:
            A copy of the registry dictionary.
        """
        var result = Dict[UUID, Node]()
        var uuids = List[UUID]()
        for uuid in self._registry.keys():
            uuids.append(uuid)
        for uuid in uuids:
            result[uuid] = self._registry[uuid]
        return result^

    fn set_grads(mut self, grads: Dict[UUID, Node]) raises:
        """Update gradients in the registry from a copy.

        Args:
            grads: A dictionary containing the updated gradients.
        """
        var uuids = List[UUID]()
        for uuid in grads.keys():
            uuids.append(uuid)
        for uuid in uuids:
            if uuid in self._registry:
                self._registry[uuid].grad = grads[uuid].grad

    fn clear(mut self):
        """Clear all nodes from the registry."""
        self._registry = Dict[UUID, Node]()


fn _init_node_registry() -> _NodeRegistry:
    """Initialize the global node registry.

    Returns:
        A new _NodeRegistry instance.
    """
    return _NodeRegistry()


# Global node registry instance
comptime _global_registry = _Global["node_registry", _init_node_registry]


fn _get_global_registry_ptr() raises -> (
    UnsafePointer[_NodeRegistry, MutOrigin.external]
):
    """Get the global registry pointer.

    Returns:
        A pointer to the global registry.
    """
    return _global_registry.get_or_create_ptr()


fn get_global_registry_copy() raises -> Dict[UUID, Node]:
    """Get a copy of the global node registry.

    Returns:
        A copy of the global registry dictionary.
    """
    var ptr = _get_global_registry_ptr()
    return ptr[].get_registry_copy()


fn update_global_grads(grads: Dict[UUID, Node]) raises:
    """Update gradients in the global registry.

    Args:
        grads: A dictionary containing the updated gradients.
    """
    var ptr = _get_global_registry_ptr()
    ptr[].set_grads(grads)


fn clear_global_registry() raises:
    """Clear all nodes from the global registry."""
    var ptr = _get_global_registry_ptr()
    ptr[].clear()


fn register_node(node: Node) raises:
    """Register a node in the global registry.

    This function is called automatically when nodes are created.

    Args:
        node: The node to register.
    """
    var ptr = _get_global_registry_ptr()
    ptr[].register(node)
