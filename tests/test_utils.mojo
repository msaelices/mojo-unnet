"""Tests for unnet/utils.mojo visualization and graph traversal functions."""

from testing import (
    assert_equal,
    assert_true,
    TestSuite,
)

from unnet import clear_global_registry
from unnet.grad import Node, Op
from unnet.utils import walk, draw


def test_node_creation():
    """Test basic Node creation for utils testing."""
    var node1 = Node(3.0, "x")
    var node2 = Node(5.0, "y")

    assert_equal(node1.get_value(), 3.0)
    assert_equal(node2.get_value(), 5.0)
    assert_equal(node1.name, "x")
    assert_equal(node2.name, "y")
    assert_equal(node1.get_grad(), 0.0)
    assert_equal(node2.get_grad(), 0.0)


def test_node_operations():
    """Test Node operations that will be used in computation graphs."""
    var a = Node(2.0, "a")
    var b = Node(3.0, "b")

    # Test addition
    var c = a + b
    assert_equal(c.get_value(), 5.0)

    # Test subtraction
    var d = a - b
    assert_equal(d.get_value(), -1.0)

    # Test multiplication
    var e = a * b
    assert_equal(e.get_value(), 6.0)

    # Test power
    var f = a**2.0
    assert_equal(f.get_value(), 4.0)


def test_node_tanh():
    """Test tanh activation function."""
    var node = Node(0.0, "zero")
    var result = node.tanh()

    # tanh(0) should be 0
    assert_equal(result.get_value(), 0.0)

    # Test with positive value
    var node2 = Node(1.0, "one")
    var result2 = node2.tanh()

    # tanh(1) ≈ 0.7615941559557649
    # Using approximate comparison (within tolerance)
    assert_true(result2.get_value() > 0.76 and result2.get_value() < 0.77)


def test_node_backward():
    """Test backward pass initialization."""
    var node = Node(5.0, "test")
    assert_equal(node.get_grad(), 0.0)

    node.backward()
    assert_equal(node.get_grad(), 1.0)


def test_walk_single_node():
    """Test walk function with a single node."""
    clear_global_registry()
    var node = Node(5.0, "x")
    ref nodes, edges = walk(node)

    assert_equal(len(nodes), 1)
    assert_equal(len(edges), 0)
    assert_equal(nodes[0].get_value(), 5.0)


def test_walk_simple_graph():
    """Test walk function with a simple computation graph."""
    from unnet import clear_global_registry

    clear_global_registry()
    # Create a simple graph: c = a + b
    var a = Node(2.0, "a")
    var b = Node(3.0, "b")
    var c = a + b  # This should create edges from a and b to c

    ref nodes, edges = walk(c)

    # Should have 3 nodes (a, b, c)
    assert_equal(len(nodes), 3)
    # Should have 2 edges (a->c, b->c)
    assert_equal(len(edges), 2)


def test_walk_complex_graph():
    """Test walk function with a more complex computation graph."""
    from unnet import clear_global_registry

    clear_global_registry()
    # Create graph: d = (a + b) * c
    var a = Node(2.0, "a")
    var b = Node(3.0, "b")
    var c = Node(4.0, "c")
    var sum_node = a + b
    var d = sum_node * c

    ref nodes, edges = walk(d)

    # Should have 5 nodes (a, b, c, sum_node, d)
    assert_equal(len(nodes), 5)
    # Should have 4 edges (a->sum_node, b->sum_node, sum_node->d, c->d)
    assert_equal(len(edges), 4)


def test_draw_single_node():
    """Test draw function with a single node."""
    from unnet import clear_global_registry

    clear_global_registry()
    var node = Node(5.0, "x")
    # Just verify that draw runs without error
    # More detailed testing would require graphviz inspection
    _ = draw(node)


def test_draw_simple_graph():
    """Test draw function with a simple computation graph."""
    from unnet import clear_global_registry

    clear_global_registry()
    var a = Node(2.0, "a")
    var b = Node(3.0, "b")
    var c = a + b

    # Verify draw runs without error
    _ = draw(c)


def main():
    TestSuite.discover_tests[__functions_in_module()]().run()
