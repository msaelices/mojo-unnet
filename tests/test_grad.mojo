from testing import (
    assert_equal,
    assert_false,
    assert_raises,
    assert_true,
    TestSuite,
)

from unnet.grad import Node, Op, calculate_gradients


def test_calculate_gradients_none():
    """Test calculate_gradients with Op.NONE."""
    var node = Node(3.0, "x")
    var result = Node(5.0, "result")
    result.grad = 2.0
    var other = Optional[Node](None)

    calculate_gradients(Op.NONE, result, node, other)

    assert_equal(node.grad, 0.0)


def test_calculate_gradients_add():
    """Test calculate_gradients with Op.ADD."""
    var node = Node(3.0, "x")
    var other_node = Node(5.0, "y")
    var result = Node(8.0, "result")
    result.grad = 2.0
    var other = Optional[Node](other_node)

    calculate_gradients(Op.ADD, result, node, other)

    assert_equal(node.grad, 2.0)
    assert_equal(other.value().grad, 2.0)


def test_calculate_gradients_sub():
    """Test calculate_gradients with Op.SUB."""
    var node = Node(5.0, "x")
    var other_node = Node(3.0, "y")
    var result = Node(2.0, "result")
    result.grad = 2.0
    var other = Optional[Node](other_node)

    calculate_gradients(Op.SUB, result, node, other)

    assert_equal(node.grad, -2.0)
    assert_equal(other.value().grad, -2.0)


def test_calculate_gradients_mul():
    """Test calculate_gradients with Op.MUL using chain rule."""
    var node = Node(3.0, "x")
    var other_node = Node(5.0, "y")
    var result = Node(15.0, "result")
    result.grad = 2.0
    var other = Optional[Node](other_node)

    calculate_gradients(Op.MUL, result, node, other)

    # For multiplication: d/dx(x*y) = y, d/dy(x*y) = x
    # node.grad += other.value * result.grad = 5.0 * 2.0 = 10.0
    # other.grad += node.value * result.grad = 3.0 * 2.0 = 6.0
    assert_equal(node.grad, 10.0)
    assert_equal(other.value().grad, 6.0)


def test_calculate_gradients_pow():
    """Test calculate_gradients with Op.POW using power rule."""
    var node = Node(3.0, "x")
    var exponent_node = Node(2.0, "exp")
    var result = Node(9.0, "result")
    result.grad = 2.0
    var other = Optional[Node](exponent_node)

    calculate_gradients(Op.POW, result, node, other)

    # For power: d/dx(x^n) = n * x^(n-1)
    # node.grad += n * x^(n-1) * result.grad = 2.0 * 3.0^1.0 * 2.0 = 2.0 * 3.0 * 2.0 = 12.0
    assert_equal(node.grad, 12.0)


def test_calculate_gradients_tanh():
    """Test calculate_gradients with Op.TANH."""
    var node = Node(0.0, "x")
    var result = Node(0.0, "result")
    result.grad = 2.0
    var other = Optional[Node](None)

    calculate_gradients(Op.TANH, result, node, other)

    # For tanh: d/dx(tanh(x)) = 1 - tanh(x)^2
    # At x=0, tanh(0)=0, so derivative is 1 - 0^2 = 1
    # node.grad += (1 - result.value^2) * result.grad = (1 - 0) * 2.0 = 2.0
    assert_equal(node.grad, 2.0)


def test_calculate_gradients_tanh_nonzero():
    """Test calculate_gradients with Op.TANH for non-zero values."""
    var node = Node(1.0, "x")
    var result = Node(0.7616, "result")
    result.grad = 3.0
    var other = Optional[Node](None)

    calculate_gradients(Op.TANH, result, node, other)

    # For tanh: d/dx(tanh(x)) = 1 - tanh(x)^2
    # node.grad += (1 - result.value^2) * result.grad
    # node.grad += (1 - 0.7616^2) * 3.0 = (1 - 0.58003456) * 3.0 ≈ 0.42 * 3.0 ≈ 1.26
    var expected_grad = (1.0 - result.value**2) * result.grad
    assert_true(
        node.grad > expected_grad - 0.01 and node.grad < expected_grad + 0.01
    )


def test_calculate_gradients_accumulation():
    """Test that calculate_gradients accumulates gradients within a single call.
    """
    var node = Node(2.0, "x")
    node.grad = 1.0
    var other_node = Node(3.0, "y")
    other_node.grad = 0.5
    var result = Node(6.0, "result")
    result.grad = 2.0
    var other = Optional[Node](other_node)

    calculate_gradients(Op.MUL, result, node, other)

    # Gradients should accumulate on top of existing values
    # node.grad = 1.0 + (other.value * result.grad) = 1.0 + (3.0 * 2.0) = 1.0 + 6.0 = 7.0
    # other.grad = 0.5 + (node.value * result.grad) = 0.5 + (2.0 * 2.0) = 0.5 + 4.0 = 4.5
    assert_equal(node.grad, 7.0)
    assert_equal(other.value().grad, 4.5)


def main():
    TestSuite.discover_tests[__functions_in_module()]().run()
