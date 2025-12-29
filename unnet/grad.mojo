"""Computational graph and automatic differentiation."""

# from builtin._location import __call_location
import math
from memory import UnsafePointer
from benchmark import keep
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

    # @always_inline
    # fn __del__(deinit self):
    #     """Destructor for Node."""
    #     # No special cleanup needed as UnsafePointer does not own the memory
    #     var call_location = __call_location()
    #     print("Deleting Node:", self.uuid, self.name, "in ", call_location)

    fn __add__(self, ref other: Node) -> Node:
        """Add two nodes."""
        return Node(
            op=Op.ADD,
            value=self.value + other.value,
            parent1=self,
            parent2=other,
        )

    fn __sub__(self, var other: Node) -> Node:
        """Subtract two nodes."""
        return Node(
            op=Op.SUB,
            value=self.value - other.value,
            parent1=self,
            parent2=other^,
        )

    fn __mul__(self, var other: Node) -> Node:
        """Multiply two nodes."""
        return Node(
            value=self.value * other.value,
            op=Op.MUL,
            parent1=self,
            parent2=other^,
        )

    fn __pow__(self, exponent: Float64) -> Node:
        """Raise node to a power."""
        return Node(
            value=self.value**exponent,
            op=Op.POW,
            parent1=self,
        )

    fn tanh(self) -> Node:
        """Apply hyperbolic tangent activation."""
        var result = math.tanh(self.value)
        return Node(
            value=result,
            op=Op.TANH,
            parent1=self,
        )

    fn backward(mut self, mut registry: List[Node]):
        """Compute gradients via backpropagation using a registry.

        Args:
            registry: A list of all nodes in the computation graph.
        """

        # Helper: find node index by UUID (inline helper)
        fn find_idx(uuid: UUID, registry: List[Node]) -> Int:
            for i in range(len(registry)):
                if registry[i].uuid == uuid:
                    return i
            return -1

        # Reset all gradients and also set self.grad in registry
        for i in range(len(registry)):
            registry[i].grad = 0.0

        # Collect nodes in topological order (inputs first, then outputs)
        var topo_order = List[Int]()
        var visited = List[UUID]()

        # Build topological order iteratively to avoid closure issues
        var added = True
        while added:
            added = False
            for i in range(len(registry)):
                var node = registry[i]
                if node.uuid in visited:
                    continue

                # Check if all parents are already in topo_order
                var parents_ready = True
                if node.has_parent1:
                    var p1_found = False
                    for idx in topo_order:
                        if registry[idx].uuid == node.parent1_uuid:
                            p1_found = True
                            break
                    if not p1_found:
                        parents_ready = False

                if parents_ready and node.has_parent2:
                    var p2_found = False
                    for idx in topo_order:
                        if registry[idx].uuid == node.parent2_uuid:
                            p2_found = True
                            break
                    if not p2_found:
                        parents_ready = False

                if parents_ready:
                    visited.append(node.uuid)
                    topo_order.append(i)
                    added = True

        # Process in reverse order (outputs to inputs)
        # First, set the gradient for self in the registry
        var self_idx = find_idx(self.uuid, registry)
        if self_idx >= 0:
            registry[self_idx].grad = 1.0

        print("Processing nodes for backpropagation:")

        for i in range(len(topo_order) - 1, -1, -1):
            var node_idx = topo_order[i]
            var node = registry[node_idx]

            if node.op == Op.NONE:
                continue

            print("  Node:", node.name, "grad:", node.grad)

            # Find parent indices
            var p1_idx = -1
            var p2_idx = -1
            if node.has_parent1:
                p1_idx = find_idx(node.parent1_uuid, registry)
            if node.has_parent2:
                p2_idx = find_idx(node.parent2_uuid, registry)

            # Calculate gradients based on operation
            if node.op == Op.ADD:
                if p1_idx >= 0:
                    registry[p1_idx].grad += node.grad
                    print(
                        "    Updated",
                        registry[p1_idx].name,
                        "grad =",
                        registry[p1_idx].grad,
                    )
                if p2_idx >= 0:
                    registry[p2_idx].grad += node.grad
                    print(
                        "    Updated",
                        registry[p2_idx].name,
                        "grad =",
                        registry[p2_idx].grad,
                    )

            elif node.op == Op.SUB:
                if p1_idx >= 0:
                    registry[p1_idx].grad += node.grad
                    print(
                        "    Updated",
                        registry[p1_idx].name,
                        "grad =",
                        registry[p1_idx].grad,
                    )
                if p2_idx >= 0:
                    registry[p2_idx].grad -= node.grad
                    print(
                        "    Updated",
                        registry[p2_idx].name,
                        "grad =",
                        registry[p2_idx].grad,
                    )

            elif node.op == Op.MUL:
                if p1_idx >= 0 and p2_idx >= 0:
                    registry[p1_idx].grad += registry[p2_idx].value * node.grad
                    registry[p2_idx].grad += registry[p1_idx].value * node.grad
                    print(
                        "    Updated",
                        registry[p1_idx].name,
                        "grad =",
                        registry[p1_idx].grad,
                    )
                    print(
                        "    Updated",
                        registry[p2_idx].name,
                        "grad =",
                        registry[p2_idx].grad,
                    )

            elif node.op == Op.TANH:
                if p1_idx >= 0:
                    registry[p1_idx].grad += (1 - node.value**2) * node.grad
                    print(
                        "    Updated",
                        registry[p1_idx].name,
                        "grad =",
                        registry[p1_idx].grad,
                    )

            elif node.op == Op.POW:
                if p1_idx >= 0 and p2_idx >= 0:
                    registry[p1_idx].grad += (
                        registry[p2_idx].value
                        * registry[p1_idx].value ** (registry[p2_idx].value - 1)
                        * node.grad
                    )
                    print(
                        "    Updated",
                        registry[p1_idx].name,
                        "grad =",
                        registry[p1_idx].grad,
                    )

        # Also update self's grad to match the registry
        self.grad = registry[self_idx].grad

    fn backward_simple(mut self):
        """Simple backward that just sets self.grad to 1.0."""
        self.grad = 1.0

    fn get_parent[i: Int](self) -> Optional[Node]:
        constrained[i in (0, 1), "The index i must be 0 or 1"]()

        @parameter
        if i == 0:
            return self.parent1
        else:
            return self.parent2

    fn write_to(self, mut writer: Some[Writer]):
        writer.write("[", self.name, "|", self.value, "|", self.grad, "]")


comptime Edge = Tuple[Node, Node]


fn calculate_gradients(
    op: Op,
    result_ptr: UnsafePointer[Node, MutAnyOrigin],
    node_ptr: UnsafePointer[Node, MutAnyOrigin],
    other_ptr: UnsafePointer[Node, MutAnyOrigin],
) -> None:
    """Calculate gradients for a node based on its operation.

    Args:
        op: The operation type of the node.
        result_ptr: Pointer to the result node from the operation.
        node_ptr: Pointer to the first parent node to update gradients for.
        other_ptr: Pointer to the second parent node involved in the operation (may be null for unary ops).
    """
    if op == Op.NONE:
        return
    elif op == Op.ADD:
        # For addition: d/da(a+b) = 1, d/db(a+b) = 1
        node_ptr[].grad += result_ptr[].grad
        other_ptr[].grad += result_ptr[].grad
    elif op == Op.SUB:
        # For subtraction: d/da(a-b) = 1, d/db(a-b) = -1
        node_ptr[].grad += result_ptr[].grad
        other_ptr[].grad -= result_ptr[].grad
    elif op == Op.MUL:
        # For multiplication: d/da(a*b) = b, d/db(a*b) = a
        node_ptr[].grad += other_ptr[].value * result_ptr[].grad
        other_ptr[].grad += node_ptr[].value * result_ptr[].grad
    elif op == Op.POW:
        # For power: d/da(a^b) = b * a^(b-1)
        node_ptr[].grad += (
            other_ptr[].value
            * node_ptr[].value ** (other_ptr[].value - 1)
            * result_ptr[].grad
        )
    elif op == Op.TANH:
        # For tanh: d/dx(tanh(x)) = 1 - tanh(x)^2
        node_ptr[].grad += (1 - result_ptr[].value ** 2) * result_ptr[].grad
