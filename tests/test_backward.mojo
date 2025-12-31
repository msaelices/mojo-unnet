from testing import (
    assert_equal,
    assert_true,
    TestSuite,
)

from unnet import (
    Node,
    Op,
    UUID,
    clear_global_registry,
    get_global_registry_ptr,
)


def test_backward_simple_addition():
    """Test backward propagation with simple addition chain e = (d + c) + a."""
    # Clear registry before test
    clear_global_registry()

    var a = Node(2.0, "a")
    var b = Node(3.0, "b")
    var c = Node(1.5, "c")
    var d = Node(4.0, "d")

    var dplusc = d + c
    dplusc.name = "dplusc"
    var e = dplusc + a
    e.name = "e"

    # Perform backpropagation using global registry
    e.backward()

    # Get the registry to check gradients
    var registry_ptr = get_global_registry_ptr()

    # Expected gradients for e = (d + c) + a:
    # de/de = 1
    # de/da = 1 (from e = dplusc + a)
    # de/ddplusc = 1
    # de/dd = 1 (from dplusc = d + c)
    # de/dc = 1 (from dplusc = d + c)
    assert_equal(registry_ptr[][e.uuid].grad, 1.0)
    assert_equal(registry_ptr[][a.uuid].grad, 1.0)
    assert_equal(registry_ptr[][c.uuid].grad, 1.0)
    assert_equal(registry_ptr[][d.uuid].grad, 1.0)
    assert_equal(registry_ptr[][dplusc.uuid].grad, 1.0)
    assert_equal(
        registry_ptr[][b.uuid].grad, 0.0
    )  # b is not used in the computation


def test_backward_multiplication():
    """Test backward propagation with multiplication."""
    # Clear registry before test
    clear_global_registry()

    var x = Node(3.0, "x")
    var y = Node(4.0, "y")
    var z = x * y
    z.name = "z"

    # Perform backpropagation
    z.backward()

    # Get the registry to check gradients
    var registry_ptr = get_global_registry_ptr()

    # For z = x * y:
    # dz/dz = 1
    # dz/dx = y = 4.0
    # dz/dy = x = 3.0
    assert_equal(registry_ptr[][z.uuid].grad, 1.0)
    assert_equal(registry_ptr[][x.uuid].grad, 4.0)
    assert_equal(registry_ptr[][y.uuid].grad, 3.0)


def test_backward_subtraction():
    """Test backward propagation with subtraction."""
    # Clear registry before test
    clear_global_registry()

    var x = Node(5.0, "x")
    var y = Node(3.0, "y")
    var z = x - y
    z.name = "z"

    # Perform backpropagation
    z.backward()

    # Get the registry to check gradients
    var registry_ptr = get_global_registry_ptr()

    # For z = x - y:
    # dz/dz = 1
    # dz/dx = 1
    # dz/dy = -1
    assert_equal(registry_ptr[][z.uuid].grad, 1.0)
    assert_equal(registry_ptr[][x.uuid].grad, 1.0)
    assert_equal(registry_ptr[][y.uuid].grad, -1.0)


def test_backward_tanh():
    """Test backward propagation with tanh activation."""
    # Clear registry before test
    clear_global_registry()

    var x = Node(0.0, "x")
    var y = x.tanh()
    y.name = "y"

    # Perform backpropagation
    y.backward()

    # Get the registry to check gradients
    var registry_ptr = get_global_registry_ptr()

    # For y = tanh(x):
    # dy/dx = 1 - tanh(x)^2 = 1 - y^2
    # At x=0, tanh(0) = 0, so dy/dx = 1 - 0 = 1
    assert_equal(registry_ptr[][y.uuid].grad, 1.0)
    assert_equal(registry_ptr[][x.uuid].grad, 1.0)


def test_backward_complex_graph():
    """Test backward propagation with a more complex computation graph."""
    # Clear registry before test
    clear_global_registry()

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

    # Perform backpropagation
    result.backward()

    # Get the registry to check gradients
    var registry_ptr = get_global_registry_ptr()

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

    assert_equal(registry_ptr[][result.uuid].grad, 1.0)
    assert_equal(registry_ptr[][product.uuid].grad, 1.0)
    assert_equal(registry_ptr[][d.uuid].grad, -1.0)
    assert_equal(registry_ptr[][sum.uuid].grad, 4.0)
    assert_equal(registry_ptr[][a.uuid].grad, 4.0)
    assert_equal(registry_ptr[][b.uuid].grad, 4.0)
    assert_equal(registry_ptr[][c.uuid].grad, 5.0)


def test_backward_multiple_uses():
    """Test backward propagation when a node is used multiple times."""
    # Clear registry before test
    clear_global_registry()

    # x * x (using same node twice)
    var x = Node(3.0, "x")
    var x_copy = Node(3.0, "x_copy")  # Create a copy with same value
    var y = x * x_copy
    y.name = "y"

    # Perform backpropagation
    y.backward()

    # Get the registry to check gradients
    var registry_ptr = get_global_registry_ptr()

    # For y = x * x_copy (where x and x_copy both equal 3.0):
    # dy/dx = x_copy = 3.0
    # dy/dx_copy = x = 3.0
    assert_equal(registry_ptr[][y.uuid].grad, 1.0)
    assert_equal(registry_ptr[][x.uuid].grad, 3.0)
    assert_equal(registry_ptr[][x_copy.uuid].grad, 3.0)


def main():
    TestSuite.discover_tests[__functions_in_module()]().run()
