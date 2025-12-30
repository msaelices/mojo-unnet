from unnet import Node, Op, UUID
from testing import assert_equal


fn main() raises:
    var a = Node(2.0, "a")
    var b = Node(3.0, "b")
    var c = Node(1.5, "c")
    var d = Node(4.0, "d")

    var dplusc = d + c
    dplusc.name = "dplusc"
    var e = dplusc + a
    e.name = "e"
    print("e.value =", e.value)

    # Create a registry of all nodes for backward propagation
    var registry = Dict[UUID, Node]()
    registry[a.uuid] = a
    registry[b.uuid] = b
    registry[c.uuid] = c
    registry[d.uuid] = d
    registry[dplusc.uuid] = dplusc
    registry[e.uuid] = e

    # Perform backpropagation
    print("\nCalling e.backward(registry)...")
    e.backward(registry)

    # After backward(), read the gradients from the registry (not the original variables)
    print("\nAfter calling e.backward(registry):")
    print("e.grad (from registry) =", registry[e.uuid].grad)
    print("a.grad (from registry) =", registry[a.uuid].grad)
    print("b.grad (from registry) =", registry[b.uuid].grad)
    print("c.grad (from registry) =", registry[c.uuid].grad)
    print("d.grad (from registry) =", registry[d.uuid].grad)
    print("dplusc.grad (from registry) =", registry[dplusc.uuid].grad)

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
    print("\nAll assertions passed!")
