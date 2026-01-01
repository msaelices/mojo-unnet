"""Neural network components: Neuron, Layer, Network."""

from .grad import Node, clear_global_registry


struct Neuron(Movable):
    """Single neuron with weights and bias."""

    var w1: Node
    var w2: Node
    var b: Node

    fn __init__(out self, num_inputs: Int):
        """Initialize a neuron with small fixed weights.

        Args:
            num_inputs: Number of input connections (must be 2 for now).
        """
        # Simple fixed initialization for 2 inputs
        self.w1 = Node(0.1, "w1")
        self.w2 = Node(0.2, "w2")
        self.b = Node(0.0, "b")

    fn __call__(self, inputs: List[Float64]) -> Node:
        """Forward pass through the neuron.

        Computes: activation(w1*x1 + w2*x2 + b)

        Args:
            inputs: List of 2 input values.

        Returns:
            A Node representing the output of the neuron.
        """
        # Weighted sum: w1*x1 + w2*x2 + b
        var x1 = Node(inputs[0], "x1")
        var x2 = Node(inputs[1], "x2")
        var sum = self.b + self.w1 * x1 + self.w2 * x2

        # Apply tanh activation
        return sum.tanh()

    fn parameters(self) -> List[Node]:
        """Return all trainable parameters (weights and biases).

        Returns:
            A list of Node objects representing the parameters.
        """
        var params = List[Node]()
        params.append(self.w1)
        params.append(self.w2)
        params.append(self.b)
        return params^


struct Layer2(Movable):
    """Layer with exactly 2 neurons."""

    var n1: Neuron
    var n2: Neuron

    fn __init__(out self, num_inputs: Int):
        """Initialize a layer with 2 neurons.

        Args:
            num_inputs: Number of inputs to each neuron (must be 2).
        """
        self.n1 = Neuron(num_inputs)
        self.n2 = Neuron(num_inputs)

    fn __call__(self, inputs: List[Float64]) -> List[Node]:
        """Forward pass through the layer.

        Args:
            inputs: List of 2 input values.

        Returns:
            A list of 2 Node objects representing the outputs.
        """
        var outputs = List[Node]()
        outputs.append(self.n1(inputs))
        outputs.append(self.n2(inputs))
        return outputs^

    fn parameters(self) -> List[Node]:
        """Return all trainable parameters.

        Returns:
            A list of Node objects representing all parameters.
        """
        var params = List[Node]()
        for p in self.n1.parameters():
            params.append(p)
        for p in self.n2.parameters():
            params.append(p)
        return params^


struct NetworkMLP(Movable):
    """A simple MLP: 2 inputs -> 2 hidden -> 1 output."""

    var hidden: Layer2
    var output: Neuron

    fn __init__(out self):
        """Initialize a 2-layer MLP with 2 inputs, 2 hidden, 1 output."""
        self.hidden = Layer2(2)
        self.output = Neuron(2)

    fn __call__(self, inputs: List[Float64]) -> Node:
        """Forward pass through the network.

        Args:
            inputs: List of 2 input values.

        Returns:
            A Node representing the final output.
        """
        # Hidden layer
        var hidden_outputs = self.hidden(inputs)

        # Convert to Float64 for output layer
        var hidden_values = List[Float64]()
        hidden_values.append(hidden_outputs[0].value)
        hidden_values.append(hidden_outputs[1].value)

        # Output layer
        return self.output(hidden_values)

    fn parameters(self) -> List[Node]:
        """Return all trainable parameters.

        Returns:
            A list of Node objects representing all parameters.
        """
        var params = List[Node]()
        for p in self.hidden.parameters():
            params.append(p)
        for p in self.output.parameters():
            params.append(p)
        return params^

    fn zero_grads(mut self):
        """Zero out all gradients in the network."""
        var params = self.parameters()
        for i in range(len(params)):
            params[i].zero_grad()
