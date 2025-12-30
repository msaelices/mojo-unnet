from testing import (
    assert_equal,
    assert_true,
    TestSuite,
)

from unnet import Node, Op, UUID


def test_backward_simple_addition():
    """Test backward propagation with simple addition chain e = (d + c) + a."""
    var a = Node(2.0, "a")
    var b = Node(3.0, "b")
    var c = Node(1.5, "c")
    var d = Node(4.0, "d")

    var dplusc = d + c
    dplusc.name = "dplusc"
    var e = dplusc + a
    e.name = "e"

    # Create a registry of all nodes for backward propagation
    var registry = Dict[UUID, Node]()
    registry[a.uuid] = a
    registry[b.uuid] = b
    registry[c.uuid] = c
    registry[d.uuid] = d
    registry[dplusc.uuid] = dplusc
    registry[e.uuid] = e

    # Perform backpropagation
    e.backward(registry)

    # Expected gradients for e = (d + c) + a:
    # de/de = 1
    # de/da = 1 (from e = dplusc + a)
    # de/ddplusc = 1
    # de/dd = 1 (from dplusc = d + c)
    # de/dc = 1 (from dplusc = d + c)
    assert_equal(registry[e.uuid].grad, 1.0)
    assert_equal(registry[a.uuid].grad, 1.0)
    assert_equal(registry[c.uuid].grad, 1.0)
    assert_equal(registry[d.uuid].grad, 1.0)
    assert_equal(registry[dplusc.uuid].grad, 1.0)
    assert_equal(registry[b.uuid].grad, 0.0)  # b is not used in the computation


def test_backward_multiplication():
    """Test backward propagation with multiplication."""
    var x = Node(3.0, "x")
    var y = Node(4.0, "y")
    var z = x * y
    z.name = "z"

    # Create registry
    var registry = Dict[UUID, Node]()
    registry[x.uuid] = x
    registry[y.uuid] = y
    registry[z.uuid] = z

    # Perform backpropagation
    z.backward(registry)

    # For z = x * y:
    # dz/dz = 1
    # dz/dx = y = 4.0
    # dz/dy = x = 3.0
    assert_equal(registry[z.uuid].grad, 1.0)
    assert_equal(registry[x.uuid].grad, 4.0)
    assert_equal(registry[y.uuid].grad, 3.0)


def test_backward_subtraction():
    """Test backward propagation with subtraction."""
    var x = Node(5.0, "x")
    var y = Node(3.0, "y")
    var z = x - y
    z.name = "z"

    # Create registry
    var registry = Dict[UUID, Node]()
    registry[x.uuid] = x
    registry[y.uuid] = y
    registry[z.uuid] = z

    # Perform backpropagation
    z.backward(registry)

    # For z = x - y:
    # dz/dz = 1
    # dz/dx = 1
    # dz/dy = -1
    assert_equal(registry[z.uuid].grad, 1.0)
    assert_equal(registry[x.uuid].grad, 1.0)
    assert_equal(registry[y.uuid].grad, -1.0)


def test_backward_tanh():
    """Test backward propagation with tanh activation."""
    var x = Node(0.0, "x")
    var y = x.tanh()
    y.name = "y"

    # Create registry
    var registry = Dict[UUID, Node]()
    registry[x.uuid] = x
    registry[y.uuid] = y

    # Perform backpropagation
    y.backward(registry)

    # For y = tanh(x):
    # dy/dx = 1 - tanh(x)^2 = 1 - y^2
    # At x=0, tanh(0) = 0, so dy/dx = 1 - 0 = 1
    assert_equal(registry[y.uuid].grad, 1.0)
    assert_equal(registry[x.uuid].grad, 1.0)


def test_backward_complex_graph():
    """Test backward propagation with a more complex computation graph."""
    # Build graph: ((a + b) * c) - d
    var a = Node(2.0, "a")
    var b = Node(3.0, "b")
    var c = Node(4.0, "c")
    var d = Node(1.0, "d")

    var sum = a + b
    sum.name = "sum"
    var product = sum * c
    product.name = "product"
    var result = product - d
    result.name = "result"

    # Create registry
    var registry = Dict[UUID, Node]()
    registry[a.uuid] = a
    registry[b.uuid] = b
    registry[c.uuid] = c
    registry[d.uuid] = d
    registry[sum.uuid] = sum
    registry[product.uuid] = product
    registry[result.uuid] = result

    # Perform backpropagation
    result.backward(registry)

    # result = product - d, where product = sum * c, where sum = a + b
    # d(result)/d(result) = 1
    # d(result)/d(product) = 1
    # d(result)/d(d) = -1
    # d(product)/d(sum) = c = 4.0
    # d(product)/d(c) = sum = 5.0
    # d(sum)/d(a) = 1
    # d(sum)/d(b) = 1
    #
    # By chain rule:
    # d(result)/da = d(result)/d(product) * d(product)/d(sum) * d(sum)/da
    #                = 1 * 4.0 * 1 = 4.0
    # d(result)/db = d(result)/d(product) * d(product)/d(sum) * d(sum)/db
    #                = 1 * 4.0 * 1 = 4.0
    # d(result)/dc = d(result)/d(product) * d(product)/dc
    #                = 1 * 5.0 = 5.0
    # d(result)/dd = -1

    assert_equal(registry[result.uuid].grad, 1.0)
    assert_equal(registry[product.uuid].grad, 1.0)
    assert_equal(registry[d.uuid].grad, -1.0)
    assert_equal(registry[sum.uuid].grad, 4.0)
    assert_equal(registry[a.uuid].grad, 4.0)
    assert_equal(registry[b.uuid].grad, 4.0)
    assert_equal(registry[c.uuid].grad, 5.0)


def test_backward_multiple_uses():
    """Test backward propagation when a node is used multiple times."""
    # x * x (using same node twice)
    var x = Node(3.0, "x")
    var x_copy = Node(3.0, "x_copy")  # Create a copy with same value
    var y = x * x_copy
    y.name = "y"

    # Create registry
    var registry = Dict[UUID, Node]()
    registry[x.uuid] = x
    registry[x_copy.uuid] = x_copy
    registry[y.uuid] = y

    # Perform backpropagation
    y.backward(registry)

    # For y = x * x_copy (where x and x_copy both equal 3.0):
    # dy/dx = x_copy = 3.0
    # dy/dx_copy = x = 3.0
    assert_equal(registry[y.uuid].grad, 1.0)
    assert_equal(registry[x.uuid].grad, 3.0)
    assert_equal(registry[x_copy.uuid].grad, 3.0)


def test_backward_resets_gradients():
    """Test that backward() resets gradients before computation."""
    var x = Node(2.0, "x")
    var y = Node(3.0, "y")
    var z = x + y
    z.name = "z"

    # Create registry
    var registry = Dict[UUID, Node]()
    registry[x.uuid] = x
    registry[y.uuid] = y
    registry[z.uuid] = z

    # Set some non-zero gradients
    registry[x.uuid].grad = 5.0
    registry[y.uuid].grad = 10.0
    registry[z.uuid].grad = 15.0

    # Perform backpropagation - should reset gradients first
    z.backward(registry)

    # All gradients should be reset and recalculated
    assert_equal(registry[z.uuid].grad, 1.0)
    assert_equal(registry[x.uuid].grad, 1.0)
    assert_equal(registry[y.uuid].grad, 1.0)


def main():
    TestSuite.discover_tests[__functions_in_module()]().run()
