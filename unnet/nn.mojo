"""Neural network components: Neuron, Layer, Network."""

from .grad import Node


struct Neuron(Movable):
    """Single neuron with weights and bias."""

    # TODO: Add weights and bias fields

    fn __init__(out self):
        """Initialize a neuron with random weights."""
        # TODO: Implement initialization
        pass

    fn __call__(self, inputs: List[Float64]) -> Node:
        """Forward pass through the neuron."""
        # TODO: Implement forward pass: weighted sum + bias + activation
        return Node(0.0)


struct Layer(Movable):
    """Layer of neurons."""

    # TODO: Add neurons field

    fn __init__(out self, num_neurons: Int, num_inputs: Int):
        """Initialize a layer with random neurons."""
        # TODO: Implement initialization
        pass

    fn __call__(self, inputs: List[Float64]) -> List[Node]:
        """Forward pass through the layer."""
        # TODO: Implement forward pass through all neurons
        var result = List[Node]()
        return result^


struct Network(Movable):
    """Multi-layer neural network."""

    # TODO: Add layers field

    fn __init__(out self):
        """Initialize an empty network."""
        # TODO: Implement initialization
        pass

    fn __call__(self, inputs: List[Float64]) -> Node:
        """Forward pass through the network."""
        # TODO: Implement forward pass through all layers
        return Node(0.0)

    fn train(
        inoutself,
        training_data: List[List[Float64]],
        targets: List[Float64],
        steps: Int = 20,
    ):
        """Train the network."""
        # TODO: Implement training loop with forward pass, loss calculation, and backprop
        pass

    @staticmethod
    fn create(num_inputs: Int, neurons: List[Int]) -> Network:
        """Create a network with specified architecture."""
        # TODO: Implement network creation
        var net = Network()
        return net^
