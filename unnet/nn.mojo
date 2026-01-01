"""Neural network components: Neuron, Layer, Network."""

from .grad import Node, clear_global_registry


struct Neuron(Movable):
    """Single neuron with weights and bias."""

    var weights: List[Node]
    var bias: Node

    fn __init__(out self, weight_values: List[Float64], bias_value: Float64):
        """Initialize a neuron with given weight values and bias.

        Args:
            weight_values: List of initial weight values.
            bias_value: Initial bias value.
        """
        self.weights = List[Node]()
        for i in range(len(weight_values)):
            self.weights.append(Node(weight_values[i], "w"))
        self.bias = Node(bias_value, "b")

    fn __call__(self, inputs: List[Float64]) -> Node:
        """Forward pass through the neuron.

        Computes: activation(sum(w_i * x_i) + b)

        Args:
            inputs: List of input values.

        Returns:
            A Node representing the output of the neuron.
        """
        # Start with bias
        var sum = self.bias

        # Add weighted inputs: sum += w_i * x_i
        for i in range(len(inputs)):
            var x = Node(inputs[i], "x")
            sum = sum + self.weights[i] * x

        # Apply tanh activation
        return sum.tanh()

    fn parameters(self) -> List[Node]:
        """Return all trainable parameters (weights and biases).

        Returns:
            A list of Node objects representing the parameters.
        """
        var params = List[Node]()
        for w in self.weights:
            params.append(w)
        params.append(self.bias)
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
        # Initialize neuron 1 with weights [0.1, 0.2] and bias 0.0
        var n1_weights = List[Float64]()
        n1_weights.append(0.1)
        n1_weights.append(0.2)
        self.n1 = Neuron(n1_weights, 0.0)

        # Initialize neuron 2 with weights [0.1, 0.2] and bias 0.0
        var n2_weights = List[Float64]()
        n2_weights.append(0.1)
        n2_weights.append(0.2)
        self.n2 = Neuron(n2_weights, 0.0)

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

        # Output neuron with weights [0.1, 0.2] and bias 0.0
        var out_weights = List[Float64]()
        out_weights.append(0.1)
        out_weights.append(0.2)
        self.output = Neuron(out_weights, 0.0)

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
