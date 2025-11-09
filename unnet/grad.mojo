"""Computational graph and automatic differentiation."""

import math
from utils import Variant


struct Op(Stringable):
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


struct Node[
    op: Op = Op.NONE,
](ImplicitlyCopyable & Movable, Writable):
    """Representation of an expression node capable of performing math operations and calculating backpropagation.

    Parameters:
        op: Operation type of the node (default: Op.NONE).
    """

    # Type aliases
    alias LeafNode = Node[op = Op.NONE]
    alias AddNode = Node[op = Op.ADD]
    alias SubNode = Node[op = Op.SUB]
    alias MulNode = Node[op = Op.MUL]
    alias PowNode = Node[op = Op.POW]
    alias TanhNode = Node[op = Op.TANH]

    alias AnyNode = Variant[
        Self.LeafNode,
        Self.AddNode,
        Self.SubNode,
        Self.MulNode,
        Self.PowNode,
        Self.TanhNode,
    ]
    alias RawNode = Tuple[String, Float64, Float64]

    var value: Float64
    var grad: Float64
    var name: String
    var parent1: Optional[Self.AnyNode]
    var parent2: Optional[Self.AnyNode]

    fn __init__(
        out self,
        value: Float64,
        name: String = "N/A",
    ):
        """Initialize a node with a value and optional name."""
        self.value = value
        self.grad = 0.0
        self.name = name
        self.parent1 = None
        self.parent2 = None

    fn __init__(
        out self,
        value: Float64,
        parent1: Self.AnyNode,
        name: String = "N/A",
    ):
        """Initialize a node with a value and optional name."""
        self.value = value
        self.grad = 0.0
        self.name = name
        self.parent1 = parent1
        self.parent2 = None

    fn __init__(
        out self,
        value: Float64,
        parent1: Self.AnyNode,
        parent2: Self.AnyNode,
        name: String = "N/A",
    ):
        """Initialize a node with a value and optional name."""
        self.value = value
        self.grad = 0.0
        self.name = name
        self.parent1 = parent1
        self.parent2 = parent2

    fn __copyinit__(out self, other: Self):
        """Copy initializer for Node."""
        self.value = other.value
        self.grad = other.grad
        self.name = other.name
        self.parent1 = other.parent1
        self.parent2 = other.parent2

    fn __add__(self, other: Node) -> Node[op = Op.ADD]:
        """Add two nodes."""
        return Node[op = Op.ADD](
            self.value + other.value,
            parent1=self,
            parent2=other,
        )

    fn __sub__(self, other: Node) -> Node[op = Op.SUB]:
        """Subtract two nodes."""
        return Node[op = Op.SUB](
            self.value - other.value,
            parent1=self,
            parent2=other,
        )

    fn __mul__(self, other: Node) -> Node[op = Op.MUL]:
        """Multiply two nodes."""
        return Node[op = Op.MUL](
            self.value * other.value,
            parent1=self,
            parent2=other,
        )

    fn __pow__(self, exponent: Float64) -> Node[op = Op.POW]:
        """Raise node to a power."""
        return Node[op = Op.POW](self.value**exponent, parent1=self)

    fn tanh(self) -> Node[op = Op.TANH]:
        """Apply hyperbolic tangent activation."""
        var result = math.tanh(self.value)
        return Node[op = Op.TANH](result, parent1=self)

    fn backward(mut self):
        """Compute gradients via backpropagation."""
        # TODO: Implement backward pass through computation graph
        self.grad = 1.0

    fn get_parent[i: Int](self) -> Optional[Self.AnyNode]:
        constrained[i in (0, 1), "The index i must be 0 or 1"]()

        @parameter
        if i == 0:
            return self.parent1
        else:
            return self.parent2

    fn write_to(self, mut writer: Some[Writer]):
        writer.write("[", self.name, "|", self.value, "|", self.grad, "]")
