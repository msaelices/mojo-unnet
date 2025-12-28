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
    # We cannot use Optional[Node] due to recursive type definition issues.
    # Current compiler error: struct has recursive reference to itself
    var parent1_ptr: UnsafePointer[Node, origin=MutAnyOrigin]
    var parent2_ptr: UnsafePointer[Node, origin=MutAnyOrigin]

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
        self.parent1_ptr = UnsafePointer[Node, origin=MutAnyOrigin]()
        self.parent2_ptr = UnsafePointer[Node, origin=MutAnyOrigin]()

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
        # Use MutAnyOrigin instead of origin_of(parent1) to avoid capturing parameter lifetime
        self.parent1_ptr = UnsafePointer[Node, origin=MutAnyOrigin](to=parent1)
        if parent2:
            self.parent2_ptr = UnsafePointer[Node, origin=MutAnyOrigin](
                to=parent2.value()
            )
        else:
            self.parent2_ptr = UnsafePointer[Node, origin=MutAnyOrigin]()
        self.grad = 0.0

    @always_inline
    fn __copyinit__(out self, other: Self):
        """Copy initializer for Node."""
        self.uuid = other.uuid
        self.value = other.value
        self.op = other.op
        self.grad = other.grad
        self.name = other.name
        self.parent1_ptr = other.parent1_ptr
        self.parent2_ptr = other.parent2_ptr
        # var call_location = __call_location()
        # print("Copying Node:", self.uuid, self.name, "in ", call_location)

    fn __eq__(self, other: Self) -> Bool:
        return self.uuid == other.uuid

    # @always_inline
    # fn __del__(deinit self):
    #     """Destructor for Node."""
    #     # No special cleanup needed as UnsafePointer does not own the memory
    #     var call_location = __call_location()
    #     print("Deleting Node:", self.uuid, self.name, "in ", call_location)

    fn __add__(self, var other: Node) -> Node:
        """Add two nodes."""
        return Node(
            op=Op.ADD,
            value=self.value + other.value,
            parent1=self,
            parent2=other^,
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

    fn backward(mut self):
        """Compute gradients via backpropagation."""
        # TODO: Implement backward pass through computation graph
        self.grad = 1.0
        print("DEBUG: ", self.parent1_ptr[].value)
        print("DEBUG: ", self.parent2_ptr[].value)
        var stack = List[UnsafePointer[Node, MutAnyOrigin]]()
        var nodes = List[UnsafePointer[Node, MutAnyOrigin]]()
        var visited = List[UnsafePointer[Node, MutAnyOrigin]]()

        stack.append(
            UnsafePointer[Node, MutAnyOrigin](to=self)
        )  # Start from this node
        print("Starting backpropagation from Node:", self.name)

        while len(stack) > 0:
            print("Stack size:", len(stack))
            var current = stack.pop()
            print("Popped Node:", current[].name)
            print("New stack size:", len(stack))

            # Skip if already visited
            if current in visited:
                continue

            print("Visiting Node:", current[].name)

            visited.append(current)
            nodes.append(current)
            print("Added Node to visited and nodes list:", current[].name)

            # Process parents
            var parent1_ptr, parent2_ptr = (
                current[].parent1_ptr,
                current[].parent2_ptr,
            )
            keep(parent1_ptr)
            keep(parent2_ptr)
            print("Parent1 Pointer:", parent1_ptr)
            print("Parent2 Pointer:", parent2_ptr)
            if parent1_ptr:
                print("Adding parent1 to stack:", parent1_ptr[].name)
                stack.append(parent1_ptr)
            if parent2_ptr:
                print("Adding parent2 to stack:", parent2_ptr[].name)
                stack.append(parent2_ptr)
        print("Backpropagation order (from output to inputs):")
        for result_ptr in nodes:
            print(
                "Visiting Node:",
                result_ptr[].name,
                "Value:",
                result_ptr[].value,
            )
            var node, other = result_ptr[].parent1_ptr, result_ptr[].parent2_ptr
            # calculate_gradients(result_ptr[].op, result_ptr[], node[], other)

    fn get_parent[i: Int](self) -> Optional[Node]:
        constrained[i in (0, 1), "The index i must be 0 or 1"]()

        @parameter
        if i == 0:
            if self.parent1_ptr:
                return self.parent1_ptr[]
            else:
                return None
        else:
            if self.parent2_ptr:
                return self.parent2_ptr[]
            else:
                return None

    fn write_to(self, mut writer: Some[Writer]):
        writer.write("[", self.name, "|", self.value, "|", self.grad, "]")


comptime Edge = Tuple[Node, Node]


fn calculate_gradients(
    op: Op,
    mut result: Node,
    mut node: Node,
    other_ptr: UnsafePointer[Node, origin=MutAnyOrigin],
) -> None:
    """Calculate gradients for a node based on its operation.

    Args:
        op: The operation type of the node.
        result: The result node from the operation.
        node: The current node to update gradients for.
        other_ptr: The other node involved in the operation, if any.
    """
    if op == Op.NONE:
        return
    elif op == Op.ADD and other_ptr:
        node.grad += result.grad
        other_ptr[].grad += result.grad
    elif op == Op.SUB and other_ptr:
        node.grad -= result.grad
        other_ptr[].grad -= result.grad
    elif op == Op.MUL and other_ptr:
        node.grad += other_ptr[].value * result.grad
        other_ptr[].grad += node.value * result.grad
    elif op == Op.POW and other_ptr:
        node.grad += (
            other_ptr[].value
            * node.value ** (other_ptr[].value - 1)
            * result.grad
        )
    elif op == Op.TANH:
        node.grad += (1 - result.value**2) * result.grad
