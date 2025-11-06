from testing import (
    assert_equal,
    assert_false,
    assert_raises,
    assert_true,
    TestSuite,
)


def test_dummy():
    assert_true(True)
    assert_equal(1 + 1, 2)


def main():
    TestSuite.discover_tests[__functions_in_module()]().run()
