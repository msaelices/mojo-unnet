"""Computational graph and automatic differentiation."""

import math


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

    var value: Float64
    var op: Op
    var grad: Float64
    var name: String
    var parent1: Optional[Self]
    var parent2: Optional[Self]

    fn __init__(
        out self,
        value: Float64,
        name: String = "N/A",
        op: Op = Op.NONE,
        parent1: Optional[Node] = None,
        parent2: Optional[Node] = None,
    ):
        """Initialize a node with a value and optional name."""
        self.value = value
        self.op = op
        self.name = name
        self.parent1 = parent1
        self.parent2 = parent2
        self.grad = 0.0

    fn __copyinit__(out self, other: Self):
        """Copy initializer for Node."""
        self.value = other.value
        self.op = other.op
        self.grad = other.grad
        self.name = other.name
        self.parent1 = other.parent1
        self.parent2 = other.parent2

    fn __eq__(self, other: Self) -> Bool:
        # TODO: Consider comparing parents as well
        return self.value == other.value and self.name == other.name

    fn __add__(self, other: Node) -> Node:
        """Add two nodes."""
        return Node(
            op=Op.ADD,
            value=self.value + other.value,
            parent1=self,
            parent2=other,
        )

    fn __sub__(self, other: Node) -> Node:
        """Subtract two nodes."""
        return Node(
            op=Op.SUB,
            value=self.value - other.value,
            parent1=self,
            parent2=other,
        )

    fn __mul__(self, other: Node) -> Node:
        """Multiply two nodes."""
        return Node(
            value=self.value * other.value,
            op=Op.MUL,
            parent1=self,
            parent2=other,
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

    fn get_parent[i: Int](self) -> Optional[Node]:
        constrained[i in (0, 1), "The index i must be 0 or 1"]()

        @parameter
        if i == 0:
            return self.parent1
        else:
            return self.parent2

    fn write_to(self, mut writer: Some[Writer]):
        writer.write("[", self.name, "|", self.value, "|", self.grad, "]")


alias Edge = Tuple[Node, Node]
