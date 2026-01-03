"""Neural network components: Neuron, Layer, Network."""

from random import random_float64
from .grad import Node, clear_global_registry


struct Neuron(Copyable):
    """Single neuron with weights and bias."""

    var weights: List[Node]
    var bias: Node

    fn __init__(out self, weights: List[Float64], bias: Float64):
        """Initialize a neuron with given weight values and bias.

        Args:
            weights: List of initial weight values.
            bias: Initial bias value.
        """
        self.weights = List[Node]()
        for i, w in enumerate(weights):
            self.weights.append(Node(w, "w{}".format(i)))
        self.bias = Node(bias, "b")

    @staticmethod
    fn create_random(num_inputs: Int) -> Neuron:
        """Create a neuron with randomly initialized weights and bias.

        Generates random weights and bias values uniformly distributed
        between -1.0 and 1.0.

        Args:
            num_inputs: Number of inputs to the neuron (determines the
                       number of weights to generate).

        Returns:
            A Neuron with randomly initialized parameters.
        """
        var weights = List[Float64]()
        for _ in range(num_inputs):
            weights.append(random_float64(-1.0, 1.0))
        var bias = random_float64(-1.0, 1.0)
        return Neuron(weights, bias)

    fn __call__(self, inputs: List[Node]) -> Node:
        """Forward pass through the neuron.

        Computes: activation(sum(w_i * x_i) + b)

        Args:
            inputs: List of input values.

        Returns:
            A Node representing the output of the neuron.
        """
        # Start with bias
        var sum: Node = self.bias

        # Add weighted inputs: sum += w_i * x_i
        for i, x in enumerate(inputs):
            sum += self.weights[i] * x

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

    fn __init__(out self):
        """Initialize an empty layer."""
        self.neurons = List[Neuron]()

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

    @staticmethod
    fn create_random(num_neurons: Int, num_inputs: Int) -> Layer:
        """Create a layer with randomly initialized neurons.

        Creates a layer where each neuron has randomly initialized weights
        and bias uniformly distributed between -1.0 and 1.0.

        Args:
            num_neurons: Number of neurons in this layer.
            num_inputs: Number of inputs to each neuron.

        Returns:
            A Layer with randomly initialized neurons.
        """
        var layer = Layer()
        layer.neurons = List[Neuron]()
        for _ in range(num_neurons):
            layer.neurons.append(Neuron.create_random(num_inputs))
        return layer^

    fn __call__(self, inputs: List[Node]) -> List[Node]:
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
    """A simple MLP."""

    var input_size: Int
    var hidden_size: Int
    var output_size: Int
    var layers: List[Layer]
    var output_layer: Layer

    fn __init__(out self):
        """Initialize an empty network."""
        self.input_size = 0
        self.hidden_size = 0
        self.output_size = 0
        self.layers = List[Layer]()
        self.output_layer = Layer()

    fn __init__(
        out self,
        num_layers: Int,
        input_size: Int,
        hidden_size: Int,
        output_size: Int = 1,
    ):
        """Initialize a MLP with input_size inputs, hidden_size hidden, output_size outputs.

        Args:
            num_layers: Number of hidden layers.
            input_size: Number of input features.
            hidden_size: Number of neurons in each hidden layer.
            output_size: Number of output neurons (default: 1).
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.layers = List[Layer]()

        for i in range(num_layers):
            if i == 0:
                # First hidden layer: hidden_size neurons, input_size inputs
                self.layers.append(Layer(hidden_size, input_size))
            else:
                # Subsequent hidden layers: hidden_size neurons, hidden_size inputs
                self.layers.append(Layer(hidden_size, hidden_size))
        # Output layer: output_size neurons, hidden_size inputs
        self.output_layer = Layer(output_size, hidden_size)

    @staticmethod
    fn create_random(
        num_layers: Int, input_size: Int, hidden_size: Int, output_size: Int = 1
    ) -> NetworkMLP:
        """Create an MLP with randomly initialized layers.

        Creates a multi-layer perceptron where all layers have randomly
        initialized weights and biases uniformly distributed between -1.0 and 1.0.

        Args:
            num_layers: Number of hidden layers.
            input_size: Number of input features.
            hidden_size: Number of neurons in each hidden layer.
            output_size: Number of output neurons (default: 1).

        Returns:
            A NetworkMLP with randomly initialized parameters.
        """
        var network = NetworkMLP()
        network.input_size = input_size
        network.hidden_size = hidden_size
        network.output_size = output_size
        network.layers = List[Layer]()

        for i in range(num_layers):
            if i == 0:
                # First hidden layer: hidden_size neurons, input_size inputs
                network.layers.append(
                    Layer.create_random(hidden_size, input_size)
                )
            else:
                # Subsequent hidden layers: hidden_size neurons, hidden_size inputs
                network.layers.append(
                    Layer.create_random(hidden_size, hidden_size)
                )
        # Output layer: output_size neurons, hidden_size inputs
        network.output_layer = Layer.create_random(output_size, hidden_size)
        return network^

    fn __call__(self, inputs: List[Node]) -> List[Node]:
        """Forward pass through the network.

        Args:
            inputs: List of num_inputs input values.

        Returns:
            A list of Node objects representing the final outputs.
        """
        var current_inputs = inputs.copy()

        # Pass through each layer. This composes the graph of nodes.
        for layer in self.layers:
            current_inputs = layer(current_inputs)

        # Return the final output layer outputs
        return self.output_layer(current_inputs)

    fn parameters(self) -> List[Node]:
        """Return all trainable parameters.

        Returns:
            A list of Node objects representing all parameters.
        """
        var params = List[Node]()
        for layer in self.layers:
            for p in layer.parameters():
                params.append(p)
        for p in self.output_layer.parameters():
            params.append(p)
        return params^

    fn zero_grads(mut self):
        """Zero out all gradients in the network."""
        var params = self.parameters()
        for i in range(len(params)):
            params[i].zero_grad()

    fn train(
        mut self,
        training_data: List[List[Float64]],
        desired_output: List[List[Float64]],
        steps: Int = 20,
    ) raises -> List[Float64]:
        """Train the network using gradient descent.

        Performs training by iterating through the dataset for the specified
        number of steps, computing loss, backpropagating gradients, and
        updating weights with a decaying learning rate.

        Args:
            training_data: List of input samples (each sample is a list of floats).
            desired_output: List of target output vectors (one per training sample).
            steps: Number of training iterations (default: 20).

        Returns:
            A list of loss values for each training step.

        Raises:
            Error: If training_data and desired_output have different lengths.
        """
        if len(training_data) != len(desired_output):
            raise Error(
                "training_data and desired_output must have the same length"
            )

        var losses = List[Float64]()
        var loss: Node

        for step in range(steps):
            # Forward pass: compute predictions and accumulate loss
            loss = 0.0
            for i in range(len(training_data)):
                # Convert input floats to Nodes
                var input_nodes = List[Node]()
                for val in training_data[i]:
                    input_nodes.append(val)

                # Get all predictions
                var outputs = self(input_nodes)

                # Compute loss for each output and sum
                for j in range(len(outputs)):
                    var prediction = outputs[j]
                    var target: Node = desired_output[i][j]
                    var error = prediction - target
                    var squared_error = error**2.0
                    loss += squared_error

            # Backward pass: zero gradients, then backpropagate
            self.zero_grads()
            loss.backward()

            # Update parameters with decaying learning rate
            var l_rate = 0.2 - 0.1 * Float64(step) / Float64(steps)
            var params = self.parameters()
            var registry_ptr = get_global_registry_ptr()

            for param in params:
                var grad = param.get_grad()
                var new_value = param.value - l_rate * grad
                registry_ptr[].set_value(param.uuid, new_value)

            # Record loss
            losses.append(loss.value)

        return losses^
