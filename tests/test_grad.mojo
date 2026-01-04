from testing import (
    assert_equal,
    assert_false,
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


def test_node_implicit_init():
    """Test that a Node can be created implicitly from a Float64."""
    clear_global_registry()

    var n: Node = 4.0
    assert_equal(n.get_value(), 4.0)
    assert_equal(n.name, "N/A")
    assert_true(n.op == Op.NONE)
    assert_false(n.parent1_uuid)
    assert_false(n.parent2_uuid)


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

    # Also verify that node.get_grad() returns the same values
    assert_equal(e.get_grad(), 1.0)
    assert_equal(a.get_grad(), 1.0)
    assert_equal(c.get_grad(), 1.0)
    assert_equal(d.get_grad(), 1.0)
    assert_equal(dplusc.get_grad(), 1.0)
    assert_equal(b.get_grad(), 0.0)


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

    # Also verify that node.get_grad() returns the same values
    assert_equal(z.get_grad(), 1.0)
    assert_equal(x.get_grad(), 4.0)
    assert_equal(y.get_grad(), 3.0)


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

    # Also verify that node.get_grad() returns the same values
    assert_equal(z.get_grad(), 1.0)
    assert_equal(x.get_grad(), 1.0)
    assert_equal(y.get_grad(), -1.0)


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

    # Also verify that node.get_grad() returns the same values
    assert_equal(y.get_grad(), 1.0)
    assert_equal(x.get_grad(), 1.0)


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

    # Also verify that node.get_grad() returns the same values
    assert_equal(result.get_grad(), 1.0)
    assert_equal(product.get_grad(), 1.0)
    assert_equal(d.get_grad(), -1.0)
    assert_equal(sum.get_grad(), 4.0)
    assert_equal(a.get_grad(), 4.0)
    assert_equal(b.get_grad(), 4.0)
    assert_equal(c.get_grad(), 5.0)


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

    # Also verify that node.get_grad() returns the same values
    assert_equal(y.get_grad(), 1.0)
    assert_equal(x.get_grad(), 3.0)
    assert_equal(x_copy.get_grad(), 3.0)


def test_zero_grad():
    """Test that zero_grad() zeroes gradients from root to leaf nodes."""
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

    # Perform backpropagation to compute gradients
    result.backward()

    # Verify gradients are computed
    assert_equal(result.get_grad(), 1.0)
    assert_equal(a.get_grad(), 4.0)
    assert_equal(b.get_grad(), 4.0)
    assert_equal(c.get_grad(), 5.0)

    # Now zero gradients from result
    result.zero_grad()

    # All gradients in the computation graph should be zero
    assert_equal(result.get_grad(), 0.0)
    assert_equal(product.get_grad(), 0.0)
    assert_equal(sum.get_grad(), 0.0)
    assert_equal(a.get_grad(), 0.0)
    assert_equal(b.get_grad(), 0.0)
    assert_equal(c.get_grad(), 0.0)
    assert_equal(d.get_grad(), 0.0)


def test_backward_accumulates_gradients():
    """Test that backward() accumulates gradients when called multiple times.

    This mimics PyTorch behavior where gradients accumulate if zero_grad()
    is not called between backward() passes.

    Note: The root node's gradient is always set to 1.0 on each backward pass
    (it's the starting point), but input/leaf node gradients accumulate.
    """
    # Clear registry before test
    clear_global_registry()

    # Build simple graph: c = a + b
    var a = Node(2.0, "a")
    var b = Node(3.0, "b")
    var c = a + b
    c.name = "c"

    # First backward pass
    c.backward()
    assert_equal(c.get_grad(), 1.0)
    assert_equal(a.get_grad(), 1.0)
    assert_equal(b.get_grad(), 1.0)

    # Second backward pass without zero_grad()
    # Gradients accumulate for input nodes (a, b), but root (c) is reset to 1.0
    c.backward()
    assert_equal(c.get_grad(), 1.0)  # Root is always set to 1.0
    assert_equal(a.get_grad(), 2.0)  # Input gradients accumulate
    assert_equal(b.get_grad(), 2.0)  # Input gradients accumulate

    # Third backward pass - gradients continue to accumulate
    c.backward()
    assert_equal(c.get_grad(), 1.0)
    assert_equal(a.get_grad(), 3.0)
    assert_equal(b.get_grad(), 3.0)

    # Now zero_grad() and backward() again
    c.zero_grad()
    c.backward()
    assert_equal(c.get_grad(), 1.0)
    assert_equal(a.get_grad(), 1.0)
    assert_equal(b.get_grad(), 1.0)


def test_node_walk():
    """Test that Node.walk() returns all reachable nodes from root to leaves."""
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

    # Walk from result
    var visited = result.walk()

    # Should contain all 7 nodes in the graph
    assert_equal(len(visited), 7)

    # Verify all expected UUIDs are in the visited list
    assert_true(result.uuid in visited)
    assert_true(product.uuid in visited)
    assert_true(sum.uuid in visited)
    assert_true(a.uuid in visited)
    assert_true(b.uuid in visited)
    assert_true(c.uuid in visited)
    assert_true(d.uuid in visited)


def test_walk_topo_simple_graph():
    """Test that walk_topo() returns nodes in topological order (inputs first).
    """
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

    # Get topological order
    var topo = result.walk_topo()

    # Should contain all 7 nodes
    assert_equal(len(topo), 7)

    # In topological order, inputs (leaves) come before outputs
    # a and b are leaves (no parents)
    # sum = a + b
    # product = sum * c
    # result = product - d

    # Find indices
    var idx_a = 0
    var idx_b = 0
    var idx_c = 0
    var idx_d = 0
    var idx_sum = 0
    var idx_product = 0
    var idx_result = 0

    for i in range(len(topo)):
        if topo[i] == a.uuid:
            idx_a = i
        if topo[i] == b.uuid:
            idx_b = i
        if topo[i] == c.uuid:
            idx_c = i
        if topo[i] == d.uuid:
            idx_d = i
        if topo[i] == sum.uuid:
            idx_sum = i
        if topo[i] == product.uuid:
            idx_product = i
        if topo[i] == result.uuid:
            idx_result = i

    # Verify topological ordering constraints:
    # - a and b must come before sum
    assert_true(idx_a < idx_sum)
    assert_true(idx_b < idx_sum)
    # - sum and c must come before product
    assert_true(idx_sum < idx_product)
    assert_true(idx_c < idx_product)
    # - product and d must come before result
    assert_true(idx_product < idx_result)
    assert_true(idx_d < idx_result)


def test_walk_topo_chain():
    """Test walk_topo() with a simple linear chain."""
    # Clear registry before test
    clear_global_registry()

    # Build chain: e = d + c + a
    var a = Node(2.0, "a")
    var c = Node(1.5, "c")
    var d = Node(4.0, "d")

    var dplusc = d + c
    dplusc.name = "dplusc"
    var e = dplusc + a
    e.name = "e"

    # Get topological order
    var topo = e.walk_topo()

    # Should contain all 5 nodes
    assert_equal(len(topo), 5)

    # Find indices
    var idx_a = 0
    var idx_c = 0
    var idx_d = 0
    var idx_dplusc = 0
    var idx_e = 0

    for i in range(len(topo)):
        if topo[i] == a.uuid:
            idx_a = i
        if topo[i] == c.uuid:
            idx_c = i
        if topo[i] == d.uuid:
            idx_d = i
        if topo[i] == dplusc.uuid:
            idx_dplusc = i
        if topo[i] == e.uuid:
            idx_e = i

    # Verify ordering: leaves before intermediate before root
    assert_true(idx_a < idx_e)
    assert_true(idx_c < idx_dplusc)
    assert_true(idx_d < idx_dplusc)
    assert_true(idx_dplusc < idx_e)


def test_walk_topo_single_node():
    """Test walk_topo() with a single node (leaf)."""
    # Clear registry before test
    clear_global_registry()

    var a = Node(2.0, "a")

    # Get topological order
    var topo = a.walk_topo()

    # Should contain only this node
    assert_equal(len(topo), 1)
    assert_equal(topo[0], a.uuid)


def test_iadd_creates_addition_node():
    """Test that __iadd__ creates a proper addition node in the computation graph.
    """
    # Clear registry before test
    clear_global_registry()

    var a = Node(2.0, "a")
    var b = Node(3.0, "b")

    # Use += operator (should create addition node)
    a += b
    a.name = "result"

    # Verify the result value
    assert_equal(a.value, 5.0)

    # Verify it has the ADD operation
    assert_true(a.op == Op.ADD)

    # Perform backpropagation to verify the graph structure
    a.backward()

    # Check gradients flow correctly through the iadd operation
    assert_equal(a.get_grad(), 1.0)


def main():
    TestSuite.discover_tests[__functions_in_module()]().run()
