"""Computational graph and automatic differentiation."""

# from builtin._location import __call_location
import math
from memory import UnsafePointer
from sys.ffi import _Global
from unnet.uuid import generate_uuid, UUID


struct Op(Equatable, Stringable, ImplicitlyCopyable & Movable):
    comptime NONE: Int = 0
    comptime ADD: Int = 1
    comptime SUB: Int = 2
    comptime MUL: Int = 3
    comptime POW: Int = 4
    comptime TANH: Int = 5

    var v: Int

    @implicit
    fn __init__(out self, v: Int):
        self.v = v

    fn __eq__(self, other: Self) -> Bool:
        return self.v == other.v

    fn __str__(self) -> String:
        if self.v == Op.NONE:
            return ""
        if self.v == Op.ADD:
            return "+"
        if self.v == Op.SUB:
            return "-"
        if self.v == Op.MUL:
            return "*"
        if self.v == Op.POW:
            return "^"
        if self.v == Op.TANH:
            return "tanh"
        return "UnknownOp"


struct Node(ImplicitlyCopyable & Movable, Equatable, Writable):
    """Representation of an expression node capable of performing math operations and calculating backpropagation.
    """

    var uuid: UUID
    var value: Float64
    var op: Op
    var grad: Float64
    var name: String
    # Store parent UUIDs to avoid recursive type
    var parent1_uuid: UUID
    var parent2_uuid: UUID
    var has_parent1: Bool
    var has_parent2: Bool

    fn __init__(
        out self,
        value: Float64,
        name: String = "N/A",
    ) raises:
        """Initialize a node with a value and optional name."""
        self.uuid = generate_uuid()
        self.value = value
        self.name = name
        self.grad = 0.0
        self.op = Op.NONE
        self.parent1_uuid = UUID()
        self.parent2_uuid = UUID()
        self.has_parent1 = False
        self.has_parent2 = False
        _register_node(self)

    fn __init__(
        out self,
        value: Float64,
        op: Op,
        var parent1: Node,
        var parent2: Optional[Node] = None,
        name: String = "N/A",
    ):
        """Initialize a node with a value and optional name."""
        self.uuid = generate_uuid()
        self.value = value
        self.op = op
        self.name = name
        # Store parent UUIDs
        self.parent1_uuid = parent1.uuid
        self.has_parent1 = True
        if parent2:
            self.parent2_uuid = parent2.value().uuid
            self.has_parent2 = True
        else:
            self.parent2_uuid = UUID()
            self.has_parent2 = False
        self.grad = 0.0
        # Note: don't register here - will be registered by the operator methods
        # after construction to avoid double registration

    @always_inline
    fn __copyinit__(out self, other: Self):
        """Copy initializer for Node."""
        self.uuid = other.uuid
        self.value = other.value
        self.op = other.op
        self.grad = other.grad
        self.name = other.name
        self.parent1_uuid = other.parent1_uuid
        self.parent2_uuid = other.parent2_uuid
        self.has_parent1 = other.has_parent1
        self.has_parent2 = other.has_parent2

    fn __eq__(self, other: Self) -> Bool:
        return self.uuid == other.uuid

    fn __add__(self, ref other: Node) raises -> Node:
        """Add two nodes."""
        var result = Node(
            op=Op.ADD,
            value=self.value + other.value,
            parent1=self,
            parent2=other,
        )
        _register_node(result)
        return result

    fn __sub__(self, var other: Node) raises -> Node:
        """Subtract two nodes."""
        var result = Node(
            op=Op.SUB,
            value=self.value - other.value,
            parent1=self,
            parent2=other^,
        )
        _register_node(result)
        return result

    fn __mul__(self, var other: Node) raises -> Node:
        """Multiply two nodes."""
        var result = Node(
            value=self.value * other.value,
            op=Op.MUL,
            parent1=self,
            parent2=other^,
        )
        _register_node(result)
        return result

    fn __pow__(self, exponent: Float64) raises -> Node:
        """Raise node to a power."""
        var result = Node(
            value=self.value**exponent,
            op=Op.POW,
            parent1=self,
        )
        _register_node(result)
        return result

    fn tanh(self) raises -> Node:
        """Apply hyperbolic tangent activation."""
        var result_val = math.tanh(self.value)
        var result = Node(
            value=result_val,
            op=Op.TANH,
            parent1=self,
        )
        _register_node(result)
        return result

    fn backward(mut self) raises:
        """Compute gradients via backpropagation using the global registry.

        Uses get_global_registry_ptr() for direct access without copying.
        """
        # Get pointer to registry and work with it directly
        var registry_ptr = get_global_registry_ptr()

        # Reset all gradients - iterate over keys and index
        var uuids = List[UUID]()
        for uuid in registry_ptr[]._registry.keys():
            uuids.append(uuid)
        for uuid in uuids:
            registry_ptr[]._registry[uuid].grad = 0.0

        # Collect nodes in topological order (inputs first, then outputs)
        var topo_order = List[UUID]()
        var visited = List[UUID]()

        # Build topological order iteratively
        var added = True
        while added:
            added = False
            for uuid in uuids:
                if uuid in visited:
                    continue

                var node = registry_ptr[]._registry[uuid]

                # Check if all parents are already in topo_order
                var parents_ready = True
                if node.has_parent1:
                    if node.parent1_uuid not in topo_order:
                        parents_ready = False
                if parents_ready and node.has_parent2:
                    if node.parent2_uuid not in topo_order:
                        parents_ready = False

                if parents_ready:
                    visited.append(uuid)
                    topo_order.append(uuid)
                    added = True

        # Process in reverse order (outputs to inputs)
        # First, set the gradient for self in the registry
        if self.uuid in registry_ptr[]._registry:
            registry_ptr[]._registry[self.uuid].grad = 1.0

        print("Processing nodes for backpropagation:")

        for i in range(len(topo_order) - 1, -1, -1):
            var uuid = topo_order[i]
            var node = registry_ptr[]._registry[uuid]

            if node.op == Op.NONE:
                continue

            print("  Node:", node.name, "grad:", node.grad)

            # Calculate gradients based on operation
            if node.op == Op.ADD:
                if node.has_parent1:
                    registry_ptr[]._registry[
                        node.parent1_uuid
                    ].grad += node.grad
                    print(
                        "    Updated",
                        registry_ptr[]._registry[node.parent1_uuid].name,
                        "grad =",
                        registry_ptr[]._registry[node.parent1_uuid].grad,
                    )
                if node.has_parent2:
                    registry_ptr[]._registry[
                        node.parent2_uuid
                    ].grad += node.grad
                    print(
                        "    Updated",
                        registry_ptr[]._registry[node.parent2_uuid].name,
                        "grad =",
                        registry_ptr[]._registry[node.parent2_uuid].grad,
                    )

            elif node.op == Op.SUB:
                if node.has_parent1:
                    registry_ptr[]._registry[
                        node.parent1_uuid
                    ].grad += node.grad
                    print(
                        "    Updated",
                        registry_ptr[]._registry[node.parent1_uuid].name,
                        "grad =",
                        registry_ptr[]._registry[node.parent1_uuid].grad,
                    )
                if node.has_parent2:
                    registry_ptr[]._registry[
                        node.parent2_uuid
                    ].grad -= node.grad
                    print(
                        "    Updated",
                        registry_ptr[]._registry[node.parent2_uuid].name,
                        "grad =",
                        registry_ptr[]._registry[node.parent2_uuid].grad,
                    )

            elif node.op == Op.MUL:
                if node.has_parent1 and node.has_parent2:
                    registry_ptr[]._registry[node.parent1_uuid].grad += (
                        registry_ptr[]._registry[node.parent2_uuid].value
                        * node.grad
                    )
                    registry_ptr[]._registry[node.parent2_uuid].grad += (
                        registry_ptr[]._registry[node.parent1_uuid].value
                        * node.grad
                    )
                    print(
                        "    Updated",
                        registry_ptr[]._registry[node.parent1_uuid].name,
                        "grad =",
                        registry_ptr[]._registry[node.parent1_uuid].grad,
                    )
                    print(
                        "    Updated",
                        registry_ptr[]._registry[node.parent2_uuid].name,
                        "grad =",
                        registry_ptr[]._registry[node.parent2_uuid].grad,
                    )

            elif node.op == Op.TANH:
                if node.has_parent1:
                    registry_ptr[]._registry[node.parent1_uuid].grad += (
                        1 - node.value**2
                    ) * node.grad
                    print(
                        "    Updated",
                        registry_ptr[]._registry[node.parent1_uuid].name,
                        "grad =",
                        registry_ptr[]._registry[node.parent1_uuid].grad,
                    )

            elif node.op == Op.POW:
                if node.has_parent1 and node.has_parent2:
                    registry_ptr[]._registry[node.parent1_uuid].grad += (
                        registry_ptr[]._registry[node.parent2_uuid].value
                        * registry_ptr[]._registry[node.parent1_uuid].value
                        ** (
                            registry_ptr[]._registry[node.parent2_uuid].value
                            - 1
                        )
                        * node.grad
                    )
                    print(
                        "    Updated",
                        registry_ptr[]._registry[node.parent1_uuid].name,
                        "grad =",
                        registry_ptr[]._registry[node.parent1_uuid].grad,
                    )

        # Also update self's grad to match the registry
        self.grad = registry_ptr[]._registry[self.uuid].grad

    fn write_to(self, mut writer: Some[Writer]):
        writer.write("[", self.name, "|", self.value, "|", self.grad, "]")


# ============== Global Node Registry ==============
# Internal registry to track all nodes for backward propagation
# Consolidated here to avoid circular dependency with separate registry module


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
    """Get the global registry pointer (internal).

    Returns:
        A pointer to the global registry.
    """
    return _global_registry.get_or_create_ptr()


fn _register_node(node: Node) raises:
    """Register a node in the global registry (internal).

    This function is called automatically when nodes are created.

    Args:
        node: The node to register.
    """
    var ptr = _get_global_registry_ptr()
    ptr[].register(node)


fn get_global_registry_ptr() raises -> (
    UnsafePointer[_NodeRegistry, MutOrigin.external]
):
    """Get the global registry pointer.

    Returns:
        A mutable pointer to the global registry, allowing direct
        access without copying. Use registry_ptr[] to dereference.
        Access the internal dict via registry_ptr[]._registry.
    """
    return _get_global_registry_ptr()


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


# ============== End Global Node Registry ==============


# Edge type for graph visualization (used by utils.walk when re-enabled)
comptime Edge = Tuple[Node, Node]
