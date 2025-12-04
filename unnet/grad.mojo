"""Computational graph and automatic differentiation."""

# from builtin._location import __call_location
import math
from memory import UnsafePointer
from unnet.uuid import generate_uuid, UUID


struct Op(EqualityComparable, Stringable, ImplicitlyCopyable & Movable):
    alias NONE: Int = 0
    alias ADD: Int = 1
    alias SUB: Int = 2
    alias MUL: Int = 3
    alias POW: Int = 4
    alias TANH: Int = 5

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


struct Node(ImplicitlyCopyable & Movable, EqualityComparable, Writable):
    """Representation of an expression node capable of performing math operations and calculating backpropagation.
    """

    var uuid: UUID
    var value: Float64
    var op: Op
    var grad: Float64
    var name: String
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
        self.parent1_ptr = UnsafePointer(to=parent1)
        if parent2:
            self.parent2_ptr = UnsafePointer(to=parent2.value())
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

    fn __add__(var self, var other: Node) -> Node:
        """Add two nodes."""
        return Node(
            op=Op.ADD,
            value=self.value + other.value,
            parent1=self^,
            parent2=other^,
        )

    fn __sub__(var self, var other: Node) -> Node:
        """Subtract two nodes."""
        return Node(
            op=Op.SUB,
            value=self.value - other.value,
            parent1=self^,
            parent2=other^,
        )

    fn __mul__(var self, var other: Node) -> Node:
        """Multiply two nodes."""
        return Node(
            value=self.value * other.value,
            op=Op.MUL,
            parent1=self^,
            parent2=other^,
        )

    fn __pow__(var self, exponent: Float64) -> Node:
        """Raise node to a power."""
        return Node(
            value=self.value**exponent,
            op=Op.POW,
            parent1=self^,
        )

    fn tanh(var self) -> Node:
        """Apply hyperbolic tangent activation."""
        var result = math.tanh(self.value)
        return Node(
            value=result,
            op=Op.TANH,
            parent1=self^,
        )

    fn backward(mut self):
        """Compute gradients via backpropagation."""
        # TODO: Implement backward pass through computation graph
        self.grad = 1.0

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


alias Edge = Tuple[Node, Node]
