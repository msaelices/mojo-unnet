"""Computational graph and automatic differentiation."""

import math


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


struct Node[
    op: Op = Op.NONE,
]:
    """Representation of an expression node capable of performing math operations and calculating backpropagation.

    Parameters:
        op: Operation type of the node (default: Op.NONE).
    """

    var value: Float64
    var grad: Float64
    var name: String

    fn __init__(out self, value: Float64, name: String = "N/A"):
        """Initialize a node with a value and optional name."""
        self.value = value
        self.grad = 0.0
        self.name = name

    fn __add__(self, other: Node) -> Node[op = Op.ADD]:
        """Add two nodes."""
        return Node[op = Op.ADD](self.value + other.value)

    fn __sub__(self, other: Node) -> Node[op = Op.SUB]:
        """Subtract two nodes."""
        return Node[op = Op.SUB](self.value - other.value)

    fn __mul__(self, other: Node) -> Node[op = Op.MUL]:
        """Multiply two nodes."""
        return Node[op = Op.MUL](self.value * other.value)

    fn __pow__(self, exponent: Float64) -> Node[op = Op.POW]:
        """Raise node to a power."""
        return Node[op = Op.POW](self.value**exponent)

    fn tanh(self) -> Node[op = Op.TANH]:
        """Apply hyperbolic tangent activation."""
        var result = math.tanh(self.value)
        return Node[op = Op.TANH](result)

    fn backward(mut self):
        """Compute gradients via backpropagation."""
        # TODO: Implement backward pass through computation graph
        self.grad = 1.0
