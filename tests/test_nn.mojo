from testing import (
    assert_equal,
    assert_true,
    TestSuite,
)

from unnet import Neuron, Layer, NetworkMLP, Node, clear_global_registry


def test_neuron_creation():
    """Test that a Neuron can be created."""
    clear_global_registry()

    var weights: List[Float64] = [0.1, 0.2]
    var neuron = Neuron(weights, 0.0)
    var params = neuron.parameters()
    assert_equal(len(params), 3)  # 2 weights + 1 bias


def test_neuron_forward_pass():
    """Test the forward pass through a neuron."""
    clear_global_registry()

    var weights: List[Float64] = [0.1, 0.2]
    var neuron = Neuron(weights, 0.0)

    # Forward pass with inputs [1.0, 2.0]
    # w=[0.1, 0.2], b=0.0
    # sum = 0.1*1.0 + 0.2*2.0 = 0.5
    # tanh(0.5) ≈ 0.462
    var result = neuron([1.0, 2.0])

    assert_true(abs(result.get_value() - 0.462) < 0.01)


def test_neuron_create_random():
    """Test that Neuron.create_random creates a valid neuron."""
    clear_global_registry()

    var neuron = Neuron.create_random(5)  # 5 inputs
    var params = neuron.parameters()

    # Should have 5 weights + 1 bias = 6 parameters
    assert_equal(len(params), 6)

    # All values should be in range [-1.0, 1.0]
    for p in params:
        assert_true(p.get_value() >= -1.0)
        assert_true(p.get_value() <= 1.0)


def test_layer_creation():
    """Test that a Layer can be created."""
    clear_global_registry()

    var layer = Layer(2, 2)  # num_neurons=2, num_inputs=2
    var params = layer.parameters()
    assert_equal(
        len(params), 6
    )  # 2 neurons * 3 params each (2 weights + 1 bias)


def test_layer_create_random():
    """Test that Layer.create_random creates a valid layer."""
    clear_global_registry()

    var layer = Layer.create_random(3, 4)  # 3 neurons, 4 inputs
    var params = layer.parameters()

    # Should have 3 neurons * (4 weights + 1 bias) = 15 parameters
    assert_equal(len(params), 15)

    # All values should be in range [-1.0, 1.0]
    for p in params:
        assert_true(p.get_value() >= -1.0)
        assert_true(p.get_value() <= 1.0)


def test_layer_forward_pass():
    """Test the forward pass through a layer."""
    clear_global_registry()

    var layer = Layer(2, 2)  # num_neurons=2, num_inputs=2
    var outputs = layer([1.0, 2.0])

    assert_equal(len(outputs), 2)
    assert_true(outputs[0].get_value() >= -1.0)
    assert_true(outputs[0].get_value() <= 1.0)


def test_network_creation():
    """Test that a NetworkMLP can be created."""
    clear_global_registry()

    var net = NetworkMLP(num_layers=2, input_size=2, hidden_size=2)
    var params = net.parameters()
    # Hidden: 2 * 3 = 6, Hidden: 2 * 3 = 6, Output: 1 * 3 = 3 => Total = 15
    assert_equal(len(params), 15)


def test_network_forward_pass():
    """Test the forward pass through a network."""
    clear_global_registry()

    var net = NetworkMLP(num_layers=2, input_size=2, hidden_size=2)
    var outputs = net([1.0, 2.0])

    assert_equal(len(outputs), 1)
    assert_true(outputs[0].get_value() >= -1.0)
    assert_true(outputs[0].get_value() <= 1.0)


def test_network_backward_pass():
    """Test backward pass through the network."""
    clear_global_registry()

    var net = NetworkMLP(num_layers=2, input_size=2, hidden_size=2)
    var outputs = net([1.0, 1.0])
    outputs[0].backward()

    # Check that gradients are computed
    var params = net.parameters()
    var has_grad = False
    for p in params:
        if p.get_grad() != 0.0:
            has_grad = True
            break

    assert_true(has_grad)


def test_network_zero_grads():
    """Test zeroing gradients in the network."""
    clear_global_registry()

    var net = NetworkMLP(num_layers=2, input_size=2, hidden_size=2)
    var outputs = net([1.0, 1.0])
    outputs[0].backward()

    net.zero_grads()

    var params = net.parameters()
    for p in params:
        assert_equal(p.get_grad(), 0.0)


def test_network_multiple_outputs():
    """Test network with multiple outputs."""
    clear_global_registry()

    var net = NetworkMLP(
        num_layers=1, input_size=2, hidden_size=3, output_size=2
    )
    var outputs = net([1.0, 2.0])

    assert_equal(len(outputs), 2)
    assert_true(outputs[0].get_value() >= -1.0)
    assert_true(outputs[0].get_value() <= 1.0)
    assert_true(outputs[1].get_value() >= -1.0)
    assert_true(outputs[1].get_value() <= 1.0)


def test_network_create_random():
    """Test that NetworkMLP.create_random creates a valid network."""
    clear_global_registry()

    var net = NetworkMLP.create_random(
        num_layers=2, input_size=3, hidden_size=4, output_size=2
    )
    var params = net.parameters()

    # Hidden 1: 4 * (3 + 1) = 16, Hidden 2: 4 * (4 + 1) = 20, Output: 2 * (4 + 1) = 10
    # Total = 46 parameters
    assert_equal(len(params), 46)

    # All values should be in range [-1.0, 1.0]
    for p in params:
        assert_true(p.get_value() >= -1.0)
        assert_true(p.get_value() <= 1.0)

    # Test forward pass works
    var outputs = net([1.0, 2.0, 3.0])
    assert_equal(len(outputs), 2)


def test_network_train_single_output():
    """Test that NetworkMLP.train performs training for single output."""
    clear_global_registry()

    # Create a simple network: 2 inputs -> 2 hidden -> 1 output
    var net = NetworkMLP.create_random(
        num_layers=1, input_size=2, hidden_size=2, output_size=1
    )

    # Simple training data: learn XOR function
    var training_data: List[List[Float64]] = [
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
    ]

    var desired_output: List[List[Float64]] = [
        [0.0],
        [1.0],
        [1.0],
        [0.0],
    ]

    # Train for 10 steps
    var losses = net.train(training_data, desired_output, steps=10)

    # Check that we got loss values
    assert_equal(len(losses), 10)

    # Loss should generally decrease (allow some tolerance for small steps)
    assert_true(losses[0] >= losses[9] or abs(losses[0] - losses[9]) < 0.5)


def test_network_train_multi_output():
    """Test that NetworkMLP.train performs training for multiple outputs."""
    clear_global_registry()

    # Create a network: 2 inputs -> 3 hidden -> 2 outputs
    var net = NetworkMLP.create_random(
        num_layers=1, input_size=2, hidden_size=3, output_size=2
    )

    # Training data
    var training_data: List[List[Float64]] = [
        [0.0, 1.0],
        [1.0, 0.0],
    ]

    var desired_output: List[List[Float64]] = [
        [0.0, 1.0],
        [1.0, 0.0],
    ]

    # Train for 10 steps
    var losses = net.train(training_data, desired_output, steps=10)

    # Check that we got loss values
    assert_equal(len(losses), 10)


def main():
    TestSuite.discover_tests[__functions_in_module()]().run()
