"""Neural network components: Neuron, Layer, Network."""

from .grad import Node, clear_global_registry


struct Neuron(Copyable):
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


struct Layer(Copyable, Movable):
    """Layer of neurons."""

    var neurons: List[Neuron]

    fn __init__(out self, num_neurons: Int, num_inputs: Int):
        """Initialize a layer with neurons.

        Args:
            num_neurons: Number of neurons in this layer.
            num_inputs: Number of inputs to each neuron.
        """
        self.neurons = List[Neuron]()
        for i in range(num_neurons):
            var weights = List[Float64]()
            for j in range(num_inputs):
                # Create weights: [0.1, 0.2, 0.3, ...] based on position
                weights.append(0.1 * Float64(j + 1))
            self.neurons.append(Neuron(weights, 0.0))

    fn __call__(self, inputs: List[Float64]) -> List[Node]:
        """Forward pass through the layer.

        Args:
            inputs: List of input values.

        Returns:
            A list of Node objects representing the outputs.
        """
        var outputs = List[Node]()
        for neuron in self.neurons:
            outputs.append(neuron(inputs))
        return outputs^

    fn parameters(self) -> List[Node]:
        """Return all trainable parameters.

        Returns:
            A list of Node objects representing all parameters.
        """
        var params = List[Node]()
        for neuron in self.neurons:
            for p in neuron.parameters():
                params.append(p)
        return params^


struct NetworkMLP(Movable):
    """A simple MLP: 2 inputs -> 2 hidden -> 1 output."""

    var layers: List[Layer]

    fn __init__(out self):
        """Initialize a 2-layer MLP with 2 inputs, 2 hidden, 1 output."""
        self.layers = List[Layer]()

        # Hidden layer: 2 neurons, 2 inputs
        self.layers.append(Layer(2, 2))

        # Output layer: 1 neuron, 2 inputs (from hidden layer)
        self.layers.append(Layer(1, 2))

    fn __call__(self, inputs: List[Float64]) -> Node:
        """Forward pass through the network.

        Args:
            inputs: List of 2 input values.

        Returns:
            A Node representing the final output.
        """
        var current_inputs = inputs.copy()

        for layer in self.layers:
            var layer_outputs = layer(current_inputs)

            # Convert Node outputs to Float64 for next layer
            current_inputs = List[Float64]()
            for output in layer_outputs:
                current_inputs.append(output.value)

        # Return the final output as a Node
        return Node(current_inputs[0], "output")

    fn parameters(self) -> List[Node]:
        """Return all trainable parameters.

        Returns:
            A list of Node objects representing all parameters.
        """
        var params = List[Node]()
        for layer in self.layers:
            for p in layer.parameters():
                params.append(p)
        return params^

    fn zero_grads(mut self):
        """Zero out all gradients in the network."""
        var params = self.parameters()
        for i in range(len(params)):
            params[i].zero_grad()
