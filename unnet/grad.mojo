"""Computational graph and automatic differentiation."""

# from builtin._location import __call_location
import math
import os
from collections.dict import DictKeyError, _DictKeyIter
from memory import UnsafePointer
from sys.ffi import _Global
from unnet.uuid import generate_uuid, UUID


struct Op(Equatable, ImplicitlyCopyable, Movable, Representable, Stringable):
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

    fn __bool__(self) -> Bool:
        """Return True if this is not a NONE operation."""
        return self.v != Op.NONE

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

    fn __repr__(self) -> String:
        return String("Op(", self.__str__(), ")")


struct Node(Equatable, ImplicitlyCopyable, Movable, Representable, Writable):
    """Representation of an expression node capable of performing math operations and calculating backpropagation.
    """

    var uuid: UUID
    var op: Op
    var name: String
    # Store parent UUIDs to avoid recursive type
    var parent1_uuid: Optional[UUID]
    var parent2_uuid: Optional[UUID]

    @implicit
    fn __init__(
        out self,
        value: Float64,
    ):
        """Initialize a node with a value and optional name."""
        self.uuid = generate_uuid()
        self.name = "N/A"
        self.op = Op.NONE
        self.parent1_uuid = None
        self.parent2_uuid = None
        _register_node(self.uuid, value)

    fn __init__(
        out self,
        value: Float64,
        name: String = "N/A",
    ):
        """Initialize a node with a value and optional name."""
        self.uuid = generate_uuid()
        self.name = name
        self.op = Op.NONE
        self.parent1_uuid = None
        self.parent2_uuid = None
        _register_node(self.uuid, value)

    @staticmethod
    fn _create(
        value: Float64,
        op: Op,
        parent1_uuid: UUID,
        parent2_uuid: Optional[UUID] = None,
        name: String = "N/A",
    ) -> Node:
        """Create a node for an operation result (internal use only).

        This creates a new node with a generated UUID for operation results.
        The node will be registered separately by the caller.

        Args:
            value: The computed value of the operation result.
            op: The operation type.
            parent1_uuid: First parent UUID.
            parent2_uuid: Optional second parent UUID.
            name: Optional node name.

        Returns:
            A new Node object with a generated UUID.

        Note:
            This is for internal use only by dunder operations.
            The caller must register the node separately.
        """
        var node = Node(0.0, "__temp__")
        node.uuid = generate_uuid()
        node.op = op
        node.name = name
        node.parent1_uuid = parent1_uuid
        node.parent2_uuid = parent2_uuid
        return node

    @staticmethod
    fn _reconstruct(
        uuid: UUID,
        op: Op,
        parent1_uuid: Optional[UUID],
        parent2_uuid: Optional[UUID],
        name: String,
    ) -> Node:
        """Reconstruct a Node from registry data (internal use only).

        This creates a Node handle from existing registry data.
        Does NOT register the node as it already exists in the registry.

        Args:
            uuid: The existing UUID from registry.
            op: The operation type from registry.
            parent1_uuid: Optional first parent UUID from registry.
            parent2_uuid: Optional second parent UUID from registry.
            name: The name to assign to the reconstructed node.

        Returns:
            A Node object with the specified UUID and metadata.

        Note:
            This is for internal use only by from_uuid() and graph traversal.
        """
        var node = Node(0.0, "__temp__")
        node.uuid = uuid
        node.op = op
        node.name = name
        node.parent1_uuid = parent1_uuid
        node.parent2_uuid = parent2_uuid
        return node

    @always_inline
    fn __copyinit__(out self, other: Self):
        """Copy initializer for Node."""
        self.uuid = other.uuid
        self.op = other.op
        self.name = other.name
        self.parent1_uuid = other.parent1_uuid
        self.parent2_uuid = other.parent2_uuid
        # var call_location = __call_location()
        # print("Copying Node:", self.uuid, self.name, "in ", call_location)

    # @always_inline
    # fn __del__(deinit self):
    #     """Destructor for Node."""
    #     # No special cleanup needed as UnsafePointer does not own the memory
    #     var call_location = __call_location()
    #     print("Deleting Node:", self.uuid, self.name, "in ", call_location)

    @always_inline
    fn __eq__(self, other: Self) -> Bool:
        return self.uuid == other.uuid

    @staticmethod
    fn from_uuid(uuid: UUID, name: String = "node") -> Node:
        """Create a Node handle from an existing UUID.

        This reconstructs a Node object from the registry using the UUID.
        The Node's op and parent UUIDs are read from the NodeState in registry.

        Args:
            uuid: The UUID of the node to reconstruct.
            name: The name to assign to the reconstructed node.

        Returns:
            A Node object with the given UUID and metadata from registry.
        """
        var registry_ptr = get_global_registry_ptr()
        var entry_opt = registry_ptr[].get(uuid)
        var op = Op(0)  # Op.NONE
        var parent1_uuid: Optional[UUID] = None
        var parent2_uuid: Optional[UUID] = None

        if entry_opt:
            var entry = entry_opt.value()
            op = entry.op
            parent1_uuid = entry.parent1_uuid
            parent2_uuid = entry.parent2_uuid

        # Create a Node with the existing UUID and metadata
        return Node._reconstruct(uuid, op, parent1_uuid, parent2_uuid, name)

    fn get_value(self) -> Float64:
        """Get the value from the registry (authoritative source).

        Returns:
            The value of this node from the global registry.
        """
        var registry_ptr = get_global_registry_ptr()
        ref entry_opt = registry_ptr[].get(self.uuid)
        if entry_opt:
            return entry_opt.value().value
        return 0.0

    fn __repr__(self) -> String:
        return String(
            "Node(uuid=",
            self.uuid,
            ", value=",
            self.get_value(),
            ", grad=",
            self.get_grad(),
            ", op=",
            repr(self.op),
            ", name=",
            self.name,
            ")",
        )

    @always_inline
    fn __add__(self, other: Node) -> Node:
        """Add two nodes."""
        var result_val = self.get_value() + other.get_value()
        var result = Node._create(
            value=result_val,
            op=Op.ADD,
            parent1_uuid=self.uuid,
            parent2_uuid=other.uuid,
        )
        _register_node(result.uuid, result_val, Op.ADD, self.uuid, other.uuid)
        return result^

    @always_inline
    fn __iadd__(mut self, other: Node):
        """In-place addition of two nodes.

        Note: For the computation graph, this creates a new addition node
        and assigns it to self, as we need to track the computation for
        backpropagation.

        Args:
            other: The node to add.
        """
        var result_val = self.get_value() + other.get_value()
        var result = Node._create(
            value=result_val,
            op=Op.ADD,
            parent1_uuid=self.uuid,
            parent2_uuid=other.uuid,
        )
        _register_node(result.uuid, result_val, Op.ADD, self.uuid, other.uuid)
        self = result^

    @always_inline
    fn __sub__(self, var other: Node) -> Node:
        """Subtract two nodes."""
        var result_val = self.get_value() - other.get_value()
        var result = Node._create(
            value=result_val,
            op=Op.SUB,
            parent1_uuid=self.uuid,
            parent2_uuid=other.uuid,
        )
        _register_node(result.uuid, result_val, Op.SUB, self.uuid, other.uuid)
        return result^

    @always_inline
    fn __mul__(self, var other: Node) -> Node:
        """Multiply two nodes."""
        var result_val = self.get_value() * other.get_value()
        var result = Node._create(
            value=result_val,
            op=Op.MUL,
            parent1_uuid=self.uuid,
            parent2_uuid=other.uuid,
        )
        _register_node(result.uuid, result_val, Op.MUL, self.uuid, other.uuid)
        return result^

    @always_inline
    fn __pow__(self, exponent: Float64) -> Node:
        """Raise node to a power."""
        var result_val = self.get_value() ** exponent
        var result = Node._create(
            value=result_val,
            op=Op.POW,
            parent1_uuid=self.uuid,
        )
        _register_node(result.uuid, result_val, Op.POW, self.uuid)
        return result^

    @always_inline
    fn tanh(self) -> Node:
        """Apply hyperbolic tangent activation."""
        var result_val = math.tanh(self.get_value())
        var result = Node._create(
            value=result_val,
            op=Op.TANH,
            parent1_uuid=self.uuid,
        )
        _register_node(result.uuid, result_val, Op.TANH, self.uuid)
        return result^

    @always_inline
    fn get_grad(self) -> Float64:
        """Get the gradient of this node."""
        var registry_ptr = get_global_registry_ptr()
        ref entry_opt = registry_ptr[].get(self.uuid)
        if entry_opt:
            return entry_opt.value().grad
        return 0.0

    fn walk(self) -> List[UUID]:
        """Walk the computation graph from this node to leaf nodes.

        Traverses the computation graph starting from this node and returns
        a list of visited UUIDs, stopping at leaf nodes (nodes with no parents).

        Returns:
            A list of UUIDs for all nodes reachable from this node,
            ordered from root to leaves (this node's UUID is first).
        """
        var visited = List[UUID]()
        var stack = List[UUID]()
        stack.append(self.uuid)

        while len(stack) > 0:
            var uuid = stack.pop()

            # Skip if already visited
            if uuid in visited:
                continue

            # Get the node from registry
            var registry_ptr = get_global_registry_ptr()
            ref entry_opt = registry_ptr[].get(uuid)
            if not entry_opt:
                continue

            visited.append(uuid)

            # Continue traversing to parents (towards leaf nodes)
            var entry = entry_opt.value()
            if entry.parent1_uuid:
                stack.append(entry.parent1_uuid.value())
            if entry.parent2_uuid:
                stack.append(entry.parent2_uuid.value())

        return visited^

    fn walk_topo(self) -> List[UUID]:
        """Walk the computation graph and return nodes in topological order.

        Traverses the computation graph starting from this node and returns
        a list of UUIDs in topological order (inputs/leaves first, then outputs).
        This is the order needed for backpropagation where gradients flow
        from outputs backwards to inputs.

        Returns:
            A list of UUIDs for all nodes reachable from this node,
            ordered from leaves to root (inputs first, then outputs).
        """
        var registry_ptr = get_global_registry_ptr()

        # Get all nodes reachable from self
        var reachable = self.walk()

        # Collect nodes in topological order (inputs first, then outputs)
        var topo_order = List[UUID]()
        var visited = List[UUID]()

        # Build topological order iteratively (Kahn's algorithm variant)
        var added = True
        while added:
            added = False
            for uuid in reachable:
                if uuid in visited:
                    continue

                ref entry_opt = registry_ptr[].get(uuid)
                if not entry_opt:
                    continue
                var entry = entry_opt.value()

                # Check if all parents are already in topo_order
                var parents_ready = True
                if entry.parent1_uuid:
                    if entry.parent1_uuid.value() not in topo_order:
                        parents_ready = False
                if parents_ready and entry.parent2_uuid:
                    if entry.parent2_uuid.value() not in topo_order:
                        parents_ready = False

                if parents_ready:
                    visited.append(uuid)
                    topo_order.append(uuid)
                    added = True

        return topo_order^

    fn zero_grad(mut self):
        """Zero out gradients for all nodes reachable from this node.

        Traverses the computation graph starting from this node and sets
        gradients to 0.0 until reaching leaf nodes (nodes with no parents).
        """
        var visited = self.walk()
        var registry_ptr = get_global_registry_ptr()
        registry_ptr[].zero_grads(visited)

    fn backward(mut self):
        """Compute gradients via backpropagation using the global registry.

        The registry stores NodeState(grad, node) for each UUID.

        Note: This does NOT reset gradients before computation.
        Call zero_grad() explicitly before backward() if needed,
        similar to PyTorch's behavior.
        """
        var registry_ptr = get_global_registry_ptr()

        # Get nodes in topological order (inputs first, then outputs)
        var topo_order = self.walk_topo()

        # Process in reverse order (outputs to inputs)
        # First, set the gradient for self in the registry
        registry_ptr[].set_grad(self.uuid, 1.0)

        for uuid in reversed(topo_order):
            ref entry_opt = registry_ptr[].get(uuid)
            if not entry_opt:
                continue
            var entry = entry_opt.value()
            var node_grad = entry.grad

            if entry.op == Op.NONE:
                continue

            # Calculate gradients based on operation
            if entry.op == Op.ADD:
                if entry.parent1_uuid:
                    registry_ptr[].add_to_grad(
                        entry.parent1_uuid.value(), node_grad
                    )
                if entry.parent2_uuid:
                    registry_ptr[].add_to_grad(
                        entry.parent2_uuid.value(), node_grad
                    )

            elif entry.op == Op.SUB:
                if entry.parent1_uuid:
                    registry_ptr[].add_to_grad(
                        entry.parent1_uuid.value(), node_grad
                    )
                if entry.parent2_uuid:
                    registry_ptr[].add_to_grad(
                        entry.parent2_uuid.value(), -node_grad
                    )

            elif entry.op == Op.MUL:
                if entry.parent1_uuid and entry.parent2_uuid:
                    ref p1_opt = registry_ptr[].get(entry.parent1_uuid.value())
                    ref p2_opt = registry_ptr[].get(entry.parent2_uuid.value())
                    if p1_opt and p2_opt:
                        var p1_entry = p1_opt.value()
                        var p2_entry = p2_opt.value()
                        var p2_val = p2_entry.value
                        var p1_val = p1_entry.value
                        registry_ptr[].add_to_grad(
                            entry.parent1_uuid.value(), p2_val * node_grad
                        )
                        registry_ptr[].add_to_grad(
                            entry.parent2_uuid.value(), p1_val * node_grad
                        )

            elif entry.op == Op.TANH:
                if entry.parent1_uuid:
                    registry_ptr[].add_to_grad(
                        entry.parent1_uuid.value(),
                        (1 - entry.value**2) * node_grad,
                    )

            elif entry.op == Op.POW:
                if entry.parent1_uuid and entry.parent2_uuid:
                    ref p1_opt = registry_ptr[].get(entry.parent1_uuid.value())
                    ref p2_opt = registry_ptr[].get(entry.parent2_uuid.value())
                    if p1_opt and p2_opt:
                        var p1_entry = p1_opt.value()
                        var p2_entry = p2_opt.value()
                        registry_ptr[].add_to_grad(
                            entry.parent1_uuid.value(),
                            (
                                p2_entry.value
                                * p1_entry.value ** (p2_entry.value - 1)
                                * node_grad
                            ),
                        )
                elif entry.parent1_uuid:
                    # Power with constant exponent (x^const where const is not in graph).
                    # For now, handles x^2 which is the common case for squared error loss.
                    # TODO: Store the exponent value in __pow__ to handle arbitrary powers.
                    ref p1_opt = registry_ptr[].get(entry.parent1_uuid.value())
                    if p1_opt:
                        var p1_entry = p1_opt.value()
                        # For x^2: d/dx(x^2) = 2*x
                        var exponent = 2.0
                        registry_ptr[].add_to_grad(
                            entry.parent1_uuid.value(),
                            (exponent * p1_entry.value * node_grad),
                        )

    fn write_to(self, mut writer: Some[Writer]):
        writer.write(
            "[", self.name, "|", self.get_value(), "|", self.get_grad(), "]"
        )


# ============== Global Node Registry ==============
# Internal registry to track all nodes for backward propagation
# Consolidated here to avoid circular dependency with separate registry module


struct NodeState(ImplicitlyCopyable, Movable):
    var grad: Float64
    var value: Float64  # Store value directly (not the full Node)
    var op: Op  # Store operation type for backward pass
    var parent1_uuid: Optional[UUID]  # Store parent UUIDs for backward pass
    var parent2_uuid: Optional[UUID]

    fn __init__(
        out self,
        value: Float64,
        grad: Float64 = 0.0,
        op: Op = Op.NONE,
        parent1_uuid: Optional[UUID] = None,
        parent2_uuid: Optional[UUID] = None,
    ):
        self.value = value
        self.grad = grad
        self.op = op
        self.parent1_uuid = parent1_uuid
        self.parent2_uuid = parent2_uuid


comptime RegType = NodeState


struct NodeRegistry(Copyable):
    """Global registry of all gradients and nodes in the computation graph.

    This struct maintains two dictionaries:
    - _grads: UUID to Float64 (gradients)
    - _nodes: UUID to Node (for traversal)
    """

    var _registry: Dict[UUID, RegType]

    fn __init__(out self):
        """Initialize an empty registry."""
        self._registry = Dict[UUID, RegType]()

    fn __getitem__(ref self, ref key: UUID) raises -> RegType:
        """Get a gradient from the registry by UUID.

        Args:
            key: The UUID of the node to retrieve.

        Returns:
            The gradient value.

        Raises:
            KeyError: If the UUID is not found in the registry.
        """
        return self._registry[key].copy()

    fn get(self, uuid: UUID) -> Optional[RegType]:
        """Get an entry (grad, node_ptr) from the registry.

        Args:
            uuid: The UUID of the node to retrieve.

        Returns:
            An Optional containing the tuple if found, or None if not found.
        """
        return self._registry.get(uuid)

    fn register(
        mut self,
        uuid: UUID,
        value: Float64,
        grad: Float64 = 0.0,
        op: Op = Op.NONE,
        parent1_uuid: Optional[UUID] = None,
        parent2_uuid: Optional[UUID] = None,
    ):
        """Register a node in the global registry.

        Args:
            uuid: The UUID of the node.
            value: The value of the node.
            grad: The initial gradient value for the node.
            op: The operation type of the node.
            parent1_uuid: Optional first parent UUID.
            parent2_uuid: Optional second parent UUID.
        """
        self._registry[uuid] = NodeState(
            value=value,
            grad=grad,
            op=op,
            parent1_uuid=parent1_uuid,
            parent2_uuid=parent2_uuid,
        )

    fn add_to_grad(mut self, uuid: UUID, delta: Float64):
        """Add a value to the gradient for a node.

        Args:
            uuid: The UUID of the node.
            delta: The value to add to the gradient.
        """
        ref entry_opt = self._registry.get(uuid)
        if entry_opt:
            var entry = entry_opt.value()
            entry.grad += delta
            self._registry[uuid] = entry

    fn set_grad(mut self, uuid: UUID, value: Float64):
        """Set the gradient for a node.

        Args:
            uuid: The UUID of the node.
            value: The new gradient value.
        """
        ref entry_opt = self._registry.get(uuid)
        if entry_opt:
            var entry = entry_opt.value()
            entry.grad = value
            self._registry[uuid] = entry

    fn set_value(mut self, uuid: UUID, value: Float64):
        """Set the value for a node in the registry.

        Args:
            uuid: The UUID of the node.
            value: The new value.
        """
        ref entry_opt = self._registry.get(uuid)
        if entry_opt:
            var entry = entry_opt.value()
            entry.value = value
            self._registry[uuid] = entry

    fn keys(self) -> List[UUID]:
        """Get all UUIDs in the registry.

        Returns:
            A list of UUIDs.
        """
        var result = List[UUID]()
        for uuid in self._registry.keys():
            result.append(uuid)
        return result^

    fn clear(mut self):
        """Clear all entries from the registry."""
        self._registry.clear()

    fn zero_grads(mut self, uuids: List[UUID]):
        """Zero out gradients for all nodes in the given list.

        Args:
            uuids: A list of UUIDs whose gradients should be set to 0.0.
        """
        for uuid in uuids:
            self.set_grad(uuid, 0.0)


fn _init_node_registry() -> NodeRegistry:
    """Initialize the global node registry.

    Returns:
        A new NodeRegistry instance.
    """
    return NodeRegistry()


# Global node registry instance
comptime _global_registry = _Global["node_registry", _init_node_registry]


fn _get_global_registry_ptr() -> (
    UnsafePointer[NodeRegistry, MutOrigin.external]
):
    """Get the global registry pointer (internal).

    Returns:
        A pointer to the global registry.
    """
    try:
        return _global_registry.get_or_create_ptr()
    except:
        os.abort("Failed to get global node registry pointer.")


fn _register_node(
    uuid: UUID,
    value: Float64,
    op: Op = Op.NONE,
    parent1_uuid: Optional[UUID] = None,
    parent2_uuid: Optional[UUID] = None,
):
    """Register a node in the global registry (internal).

    This function is called automatically when nodes are created.

    Args:
        uuid: The UUID of the node.
        value: The value of the node.
        op: The operation type of the node.
        parent1_uuid: Optional first parent UUID.
        parent2_uuid: Optional second parent UUID.
    """
    var ptr = _get_global_registry_ptr()
    ptr[].register(uuid, value, 0.0, op, parent1_uuid, parent2_uuid)


fn get_global_registry_ptr() -> UnsafePointer[NodeRegistry, MutOrigin.external]:
    """Get the global registry pointer.

    Returns:
        A mutable pointer to the global registry, allowing direct
        access without copying. Use registry_ptr[] to dereference.
        Access the internal dict via registry_ptr[]._registry.
    """
    return _get_global_registry_ptr()


fn clear_global_registry():
    """Clear all entries from the global registry."""
    var ptr = _get_global_registry_ptr()
    ptr[].clear()


# ============== End Global Node Registry ==============


# Edge type for graph visualization (used by utils.walk when re-enabled)
comptime Edge = Tuple[Node, Node]
