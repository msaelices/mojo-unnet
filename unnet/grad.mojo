"""Computational graph and automatic differentiation."""

# from builtin._location import __call_location
import math
import os
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
    ):
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
        # var call_location = __call_location()
        # print("Copying Node:", self.uuid, self.name, "in ", call_location)

    # @always_inline
    # fn __del__(deinit self):
    #     """Destructor for Node."""
    #     # No special cleanup needed as UnsafePointer does not own the memory
    #     var call_location = __call_location()
    #     print("Deleting Node:", self.uuid, self.name, "in ", call_location)

    fn __eq__(self, other: Self) -> Bool:
        return self.uuid == other.uuid

    fn __add__(self, ref other: Node) -> Node:
        """Add two nodes."""
        var result = Node(
            op=Op.ADD,
            value=self.value + other.value,
            parent1=self,
            parent2=other,
        )
        _register_node(result)
        return result^

    fn __sub__(self, var other: Node) -> Node:
        """Subtract two nodes."""
        var result = Node(
            op=Op.SUB,
            value=self.value - other.value,
            parent1=self,
            parent2=other^,
        )
        _register_node(result)
        return result^

    fn __mul__(self, var other: Node) -> Node:
        """Multiply two nodes."""
        var result = Node(
            value=self.value * other.value,
            op=Op.MUL,
            parent1=self,
            parent2=other^,
        )
        _register_node(result)
        return result^

    fn __pow__(self, exponent: Float64) -> Node:
        """Raise node to a power."""
        var result = Node(
            value=self.value**exponent,
            op=Op.POW,
            parent1=self,
        )
        _register_node(result)
        return result^

    fn tanh(self) -> Node:
        """Apply hyperbolic tangent activation."""
        var result_val = math.tanh(self.value)
        var result = Node(
            value=result_val,
            op=Op.TANH,
            parent1=self,
        )
        _register_node(result)
        return result^

    fn backward(mut self):
        """Compute gradients via backpropagation using the global registry.

        Uses get_global_registry_copy() for working with a local copy.
        """
        # Get a copy of the registry to work with
        var registry = get_global_registry_copy()

        # Reset all gradients
        var uuids = List[UUID]()
        for uuid in registry.keys():
            uuids.append(uuid)
        for uuid in uuids:
            var node_opt = registry.get(uuid)
            if node_opt != None:
                var mut_node = node_opt.value()
                mut_node.grad = 0.0
                registry[uuid] = mut_node

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

                var node_opt = registry.get(uuid)
                if node_opt == None:
                    continue
                var node = node_opt.value()

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
        var self_opt = registry.get(self.uuid)
        if self_opt != None:
            var mut_self = self_opt.value()
            mut_self.grad = 1.0
            registry[self.uuid] = mut_self

        print("Processing nodes for backpropagation:")

        for i in range(len(topo_order) - 1, -1, -1):
            var uuid = topo_order[i]
            var node_opt = registry.get(uuid)
            if node_opt == None:
                continue
            var node = node_opt.value()

            if node.op == Op.NONE:
                continue

            print("  Node:", node.name, "grad:", node.grad)

            # Calculate gradients based on operation
            if node.op == Op.ADD:
                if node.has_parent1:
                    var p1_opt = registry.get(node.parent1_uuid)
                    if p1_opt != None:
                        var p1 = p1_opt.value()
                        p1.grad += node.grad
                        registry[node.parent1_uuid] = p1
                        print("    Updated", p1.name, "grad =", p1.grad)
                if node.has_parent2:
                    var p2_opt = registry.get(node.parent2_uuid)
                    if p2_opt != None:
                        var p2 = p2_opt.value()
                        p2.grad += node.grad
                        registry[node.parent2_uuid] = p2
                        print("    Updated", p2.name, "grad =", p2.grad)

            elif node.op == Op.SUB:
                if node.has_parent1:
                    var p1_opt = registry.get(node.parent1_uuid)
                    if p1_opt != None:
                        var p1 = p1_opt.value()
                        p1.grad += node.grad
                        registry[node.parent1_uuid] = p1
                        print("    Updated", p1.name, "grad =", p1.grad)
                if node.has_parent2:
                    var p2_opt = registry.get(node.parent2_uuid)
                    if p2_opt != None:
                        var p2 = p2_opt.value()
                        p2.grad -= node.grad
                        registry[node.parent2_uuid] = p2
                        print("    Updated", p2.name, "grad =", p2.grad)

            elif node.op == Op.MUL:
                if node.has_parent1 and node.has_parent2:
                    var p1_opt = registry.get(node.parent1_uuid)
                    var p2_opt = registry.get(node.parent2_uuid)
                    if p1_opt != None and p2_opt != None:
                        var p1 = p1_opt.value()
                        var p2 = p2_opt.value()
                        var p2_val = p2.value
                        var p1_val = p1.value
                        p1.grad += p2_val * node.grad
                        p2.grad += p1_val * node.grad
                        registry[node.parent1_uuid] = p1
                        registry[node.parent2_uuid] = p2
                        print("    Updated", p1.name, "grad =", p1.grad)
                        print("    Updated", p2.name, "grad =", p2.grad)

            elif node.op == Op.TANH:
                if node.has_parent1:
                    var p1_opt = registry.get(node.parent1_uuid)
                    if p1_opt != None:
                        var p1 = p1_opt.value()
                        p1.grad += (1 - node.value**2) * node.grad
                        registry[node.parent1_uuid] = p1
                        print("    Updated", p1.name, "grad =", p1.grad)

            elif node.op == Op.POW:
                if node.has_parent1 and node.has_parent2:
                    var p1_opt = registry.get(node.parent1_uuid)
                    var p2_opt = registry.get(node.parent2_uuid)
                    if p1_opt != None and p2_opt != None:
                        var p1 = p1_opt.value()
                        var p2 = p2_opt.value()
                        p1.grad += (
                            p2.value * p1.value ** (p2.value - 1) * node.grad
                        )
                        registry[node.parent1_uuid] = p1
                        print("    Updated", p1.name, "grad =", p1.grad)

        # Also update self's grad to match the registry
        var self_grad_opt = registry.get(self.uuid)
        if self_grad_opt != None:
            self.grad = self_grad_opt.value().grad

        # Update the global registry with the computed gradients
        update_global_grads(registry)

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


fn _get_global_registry_ptr() -> (
    UnsafePointer[_NodeRegistry, MutOrigin.external]
):
    """Get the global registry pointer (internal).

    Returns:
        A pointer to the global registry.
    """
    try:
        return _global_registry.get_or_create_ptr()
    except:
        os.abort()


fn _register_node(node: Node):
    """Register a node in the global registry (internal).

    This function is called automatically when nodes are created.

    Args:
        node: The node to register.
    """
    var ptr = _get_global_registry_ptr()
    ptr[].register(node)


fn get_global_registry_ptr() -> (
    UnsafePointer[_NodeRegistry, MutOrigin.external]
):
    """Get the global registry pointer.

    Returns:
        A mutable pointer to the global registry, allowing direct
        access without copying. Use registry_ptr[] to dereference.
        Access the internal dict via registry_ptr[]._registry.
    """
    return _get_global_registry_ptr()


fn get_global_registry_copy() -> Dict[UUID, Node]:
    """Get a copy of the global node registry.

    Returns:
        A copy of the global registry dictionary.
    """
    try:
        var ptr = _get_global_registry_ptr()
        return ptr[].get_registry_copy()
    except:
        os.abort()


fn update_global_grads(grads: Dict[UUID, Node]):
    """Update gradients in the global registry.

    Args:
        grads: A dictionary containing the updated gradients.
    """
    try:
        var ptr = _get_global_registry_ptr()
        ptr[].set_grads(grads)
    except:
        os.abort()


fn clear_global_registry():
    """Clear all nodes from the global registry."""
    try:
        var ptr = _get_global_registry_ptr()
        ptr[].clear()
    except:
        os.abort()


# ============== End Global Node Registry ==============


# Edge type for graph visualization (used by utils.walk when re-enabled)
comptime Edge = Tuple[Node, Node]
