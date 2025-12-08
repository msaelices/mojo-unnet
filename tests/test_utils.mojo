"""Tests for unnet/utils.mojo visualization and graph traversal functions."""

from testing import (
    assert_equal,
    assert_false,
    assert_raises,
    assert_true,
    TestSuite,
)

from unnet.grad import Node, Op
from unnet.utils import walk, draw, calculate_gradients


def test_node_creation():
    """Test basic Node creation for utils testing."""
    var node1 = Node(3.0, "x")
    var node2 = Node(5.0, "y")

    assert_equal(node1.value, 3.0)
    assert_equal(node2.value, 5.0)
    assert_equal(node1.name, "x")
    assert_equal(node2.name, "y")
    assert_equal(node1.grad, 0.0)
    assert_equal(node2.grad, 0.0)


def test_node_operations():
    """Test Node operations that will be used in computation graphs."""
    var a = Node(2.0, "a")
    var b = Node(3.0, "b")

    # Test addition
    var c = a + b
    assert_equal(c.value, 5.0)

    # Test subtraction
    var d = a - b
    assert_equal(d.value, -1.0)

    # Test multiplication
    var e = a * b
    assert_equal(e.value, 6.0)

    # Test power
    var f = a**2.0
    assert_equal(f.value, 4.0)


def test_node_tanh():
    """Test tanh activation function."""
    var node = Node(0.0, "zero")
    var result = node.tanh()

    # tanh(0) should be 0
    assert_equal(result.value, 0.0)

    # Test with positive value
    var node2 = Node(1.0, "one")
    var result2 = node2.tanh()

    # tanh(1) ≈ 0.7615941559557649
    # Using approximate comparison (within tolerance)
    assert_true(result2.value > 0.76 and result2.value < 0.77)


def test_node_backward():
    """Test backward pass initialization."""
    var node = Node(5.0, "test")
    assert_equal(node.grad, 0.0)

    node.backward()
    assert_equal(node.grad, 1.0)


# TODO: Uncomment and implement these tests once Node has op and parents fields
# def test_walk_single_node():
#     """Test walk function with a single node."""
#     var node = Node(5.0, "x")
#     var nodes, edges = walk(node)
#
#     assert_equal(len(nodes), 1)
#     assert_equal(len(edges), 0)
#     assert_equal(nodes[0].value, 5.0)


# def test_walk_simple_graph():
#     """Test walk function with a simple computation graph."""
#     # Create a simple graph: c = a + b
#     var a = Node(2.0, "a")
#     var b = Node(3.0, "b")
#     var c = a + b  # This should create edges from a and b to c
#
#     var nodes, edges = walk(c)
#
#     # Should have 3 nodes (a, b, c)
#     assert_equal(len(nodes), 3)
#     # Should have 2 edges (a->c, b->c)
#     assert_equal(len(edges), 2)


# def test_walk_complex_graph():
#     """Test walk function with a more complex computation graph."""
#     # Create graph: d = (a + b) * c
#     var a = Node(2.0, "a")
#     var b = Node(3.0, "b")
#     var c = Node(4.0, "c")
#     var sum_node = a + b
#     var d = sum_node * c
#
#     var nodes, edges = walk(d)
#
#     # Should have 4 nodes (a, b, c, sum_node, d)
#     assert_true(len(nodes) >= 3)
#     # Should have multiple edges
#     assert_true(len(edges) >= 2)


# def test_draw_single_node():
#     """Test draw function with a single node."""
#     var node = Node(5.0, "x")
#     var plot = draw(node)
#
#     # Just verify that draw returns something (PythonObject)
#     # More detailed testing would require graphviz inspection
#     assert_true(plot is not None)


# def test_draw_simple_graph():
#     """Test draw function with a simple computation graph."""
#     var a = Node(2.0, "a")
#     var b = Node(3.0, "b")
#     var c = a + b
#
#     var plot = draw(c)
#
#     # Verify plot object is created
#     assert_true(plot is not None)


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
