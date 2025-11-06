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


alias LeafNode = Node[op = Op.NONE]
alias AddNode = Node[op = Op.ADD]
alias SubNode = Node[op = Op.SUB]
alias MulNode = Node[op = Op.MUL]
alias PowNode = Node[op = Op.POW]
alias TanhNode = Node[op = Op.TANH]

alias AnyNode = Variant[
    LeafNode,
    AddNode,
    SubNode,
    MulNode,
    PowNode,
    TanhNode,
]


struct Node[
    op: Op = Op.NONE,
](ImplicitlyCopyable & Movable, Writable):
    """Representation of an expression node capable of performing math operations and calculating backpropagation.

    Parameters:
        op: Operation type of the node (default: Op.NONE).
    """

    var value: Float64
    var grad: Float64
    var name: String
    var parents: List[AnyNode]

    fn __copyinit__(out self, other: Self):
        """Copy initializer for Node."""
        self.value = other.value
        self.grad = other.grad
        self.name = other.name
        self.parents = other.parents.copy()

    fn __init__(
        out self,
        value: Float64,
        name: String = "N/A",
        parents: List[AnyNode] = List[AnyNode](),
    ):
        """Initialize a node with a value and optional name."""
        self.value = value
        self.grad = 0.0
        self.name = name
        self.parents = parents.copy()

    fn __add__(self, other: Node) -> Node[op = Op.ADD]:
        """Add two nodes."""
        return Node[op = Op.ADD](
            self.value + other.value, parents=[self, other]
        )

    fn __sub__(self, other: Node) -> Node[op = Op.SUB]:
        """Subtract two nodes."""
        return Node[op = Op.SUB](
            self.value - other.value, parents=[self, other]
        )

    fn __mul__(self, other: Node) -> Node[op = Op.MUL]:
        """Multiply two nodes."""
        return Node[op = Op.MUL](
            self.value * other.value, parents=[self, other]
        )

    fn __pow__(self, exponent: Float64) -> Node[op = Op.POW]:
        """Raise node to a power."""
        return Node[op = Op.POW](self.value**exponent, parents=[self])

    fn tanh(self) -> Node[op = Op.TANH]:
        """Apply hyperbolic tangent activation."""
        var result = math.tanh(self.value)
        return Node[op = Op.TANH](result, parents=[self])

    fn backward(mut self):
        """Compute gradients via backpropagation."""
        # TODO: Implement backward pass through computation graph
        self.grad = 1.0

    fn write_to(self, mut writer: Some[Writer]):
        writer.write("[", self.name, "|", self.value, "|", self.grad, "]")
