from testing import (
    assert_equal,
    assert_true,
    TestSuite,
)

from unnet import Neuron, Layer2, NetworkMLP, Node, clear_global_registry


def test_neuron_creation():
    """Test that a Neuron can be created."""
    clear_global_registry()

    var weights = List[Float64]()
    weights.append(0.1)
    weights.append(0.2)
    var neuron = Neuron(weights, 0.0)
    var params = neuron.parameters()
    assert_equal(len(params), 3)  # 2 weights + 1 bias


def test_neuron_forward_pass():
    """Test the forward pass through a neuron."""
    clear_global_registry()

    var weights = List[Float64]()
    weights.append(0.1)
    weights.append(0.2)
    var neuron = Neuron(weights, 0.0)

    # Forward pass with inputs [1.0, 2.0]
    # w=[0.1, 0.2], b=0.0
    # sum = 0.1*1.0 + 0.2*2.0 = 0.5
    # tanh(0.5) ≈ 0.462
    var result = neuron([1.0, 2.0])

    assert_true(abs(result.value - 0.462) < 0.01)


def test_layer_creation():
    """Test that a Layer2 can be created."""
    clear_global_registry()

    var layer = Layer2(2)
    var params = layer.parameters()
    assert_equal(len(params), 6)  # 2 neurons * 3 params


def test_layer_forward_pass():
    """Test the forward pass through a layer."""
    clear_global_registry()

    var layer = Layer2(2)
    var outputs = layer([1.0, 2.0])

    assert_equal(len(outputs), 2)
    assert_true(outputs[0].value >= -1.0)
    assert_true(outputs[0].value <= 1.0)


def test_network_creation():
    """Test that a NetworkMLP can be created."""
    clear_global_registry()

    var net = NetworkMLP()
    var params = net.parameters()
    # Hidden: 2 * 3 = 6, Output: 1 * 3 = 3, Total: 9
    assert_equal(len(params), 9)


def test_network_forward_pass():
    """Test the forward pass through a network."""
    clear_global_registry()

    var net = NetworkMLP()
    var output = net([1.0, 2.0])

    assert_true(output.value >= -1.0)
    assert_true(output.value <= 1.0)


def test_network_backward_pass():
    """Test backward pass through the network."""
    clear_global_registry()

    var net = NetworkMLP()
    var output = net([1.0, 1.0])
    output.backward()

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

    var net = NetworkMLP()
    var output = net([1.0, 1.0])
    output.backward()

    net.zero_grads()

    var params = net.parameters()
    for p in params:
        assert_equal(p.get_grad(), 0.0)


def main():
    TestSuite.discover_tests[__functions_in_module()]().run()
