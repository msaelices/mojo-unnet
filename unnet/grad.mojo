"""Computational graph and automatic differentiation."""


struct Node:
    """Representation of an expression node capable of performing math operations and calculating backpropagation.
    """

    var value: Float64
    var grad: Float64
    var name: String
    # TODO: Add op and parents fields for computation graph

    fn __init__(inoutself, value: Float64, name: String = "N/A"):
        """Initialize a node with a value and optional name."""
        self.value = value
        self.grad = 0.0
        self.name = name

    fn __add__(self, other: Node) -> Node:
        """Add two nodes."""
        # TODO: Implement addition with gradient tracking
        return Node(self.value + other.value)

    fn __sub__(self, other: Node) -> Node:
        """Subtract two nodes."""
        # TODO: Implement subtraction with gradient tracking
        return Node(self.value - other.value)

    fn __mul__(self, other: Node) -> Node:
        """Multiply two nodes."""
        # TODO: Implement multiplication with gradient tracking
        return Node(self.value * other.value)

    fn __pow__(self, exponent: Float64) -> Node:
        """Raise node to a power."""
        # TODO: Implement power with gradient tracking
        return Node(self.value**exponent)

    fn tanh(self) -> Node:
        """Apply hyperbolic tangent activation."""
        # TODO: Implement tanh with gradient tracking
        var result = (self.value.exp() - (-self.value).exp()) / (
            self.value.exp() + (-self.value).exp()
        )
        return Node(result)

    fn backward(inoutself):
        """Compute gradients via backpropagation."""
        # TODO: Implement backward pass through computation graph
        self.grad = 1.0
