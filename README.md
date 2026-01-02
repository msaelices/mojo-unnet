# mojo-unnet

Micro Neural Network in pure Mojo - Educational proof-of-concept for learning neural networks.

This is a pure Mojo implementation inspired by [unnet](https://github.com/yourusername/unnet), featuring all neural network operations implemented natively in Mojo for maximum performance.

## Implementation Status

### `grad.mojo` - Autograd
- [x] Node struct with computation graph tracking
- [x] Operator overloading (+, -, *, ^)
- [x] Activation functions (tanh)
- [x] Backward propagation
- [x] Graph traversal (walk, walk_topo)
- [x] Gradient zeroing (zero_grad)
- [x] Global registry for gradient storage
- [x] Implicit Float64 to Node conversion
- [x] Representable conformance for debugging

### `nn.mojo` - Neural Network Components
- [x] Neuron struct (weights, bias, forward pass with tanh activation)
- [x] Layer struct (multiple neurons, configurable size)
- [x] NetworkMLP struct (multi-layer perceptron with parameterized architecture)
- [x] Parameter access for gradient computation
- [x] Gradient zeroing across network
- [ ] Training loop with gradient descent

### `utils.mojo` - Utilities
- [x] Graph traversal (walk)
- [x] Visualization helpers (draw with graphviz)
- [x] Node data extraction helpers

### `uuid.mojo` - UUID Generation
- [x] UUID struct for node identification
- [x] UUID generation and comparison

## Features

- **Pure Mojo**: Entire codebase written in Mojo for maximum performance
- **Educational**: Clear, simple implementation for learning neural network fundamentals
- **Autograd**: Automatic differentiation with full backpropagation support
- **Computational Graph**: Node-based computation graph with gradient tracking
- **Operator Overloading**: Natural syntax for math operations (`+`, `-`, `*`, `^`)
- **Activation Functions**: Built-in tanh activation with proper gradients
- **Graph Traversal**: Walk the computation graph (both DFS and topological order)
- **Visualization**: Graphviz integration for computation graph visualization
- **Global Registry**: Centralized gradient storage and node management

## Installation

### Prerequisites

- [Mojo](https://docs.modular.com/mojo/)
- [pixi](https://github.com/prefix-dev/pixi) package manager (for dev tools)

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/mojo-unnet.git
cd mojo-unnet

# Install development dependencies (pre-commit, mojo tools)
pixi install
```

## Usage

### Basic Autograd Example

```mojo
from unnet import Node, clear_global_registry

# Clear the registry before starting
clear_global_registry()

# Create input nodes (leaf nodes)
var a = Node(2.0, "a")
var b = Node(-3.0, "b")
var c = Node(10.0, "c")

# Build computation graph: d = a * b + c
var d = a * b
var e = d + c

# Forward pass: compute value
print(e.value)  # 4.0

# Backward pass: compute gradients
e.backward()

# Access gradients
print(a.get_grad())  # -3.0 (de/da = b = -3.0)
print(b.get_grad())  # 2.0 (de/db = a = 2.0)
print(c.get_grad())  # 1.0
```

### Neural Network Example

```mojo
from unnet import NetworkMLP, Node, clear_global_registry

# Clear the registry before starting
clear_global_registry()

# Create a 2-layer MLP with 2 inputs, 2 hidden neurons, 1 output
var net = NetworkMLP(num_layers=2, input_size=2, hidden_size=2)

# Forward pass with inputs
var inputs = List[Node]()
inputs.append(Node(1.0))
inputs.append(Node(2.0))
var output = net(inputs)

# Backward pass to compute gradients
output.backward()

# Zero gradients for next iteration
net.zero_grads()
```

## Project Structure

```
mojo-unnet/
├── unnet/               # Main package
│   ├── __init__.mojo    # Package initialization
│   ├── grad.mojo        # Autograd implementation (Node, Op, GradRegistry)
│   ├── uuid.mojo        # UUID generation and comparison
│   ├── nn.mojo          # Neural network components (Neuron, Layer, NetworkMLP)
│   └── utils.mojo       # Utilities (graph traversal, graphviz visualization)
├── tests/               # Test suite (Mojo tests)
│   ├── test_grad.mojo   # Tests for autograd (13 tests, all passing)
│   ├── test_nn.mojo     # Tests for neural network components (8 tests, all passing)
│   └── test_utils.mojo  # Tests for utilities (9 tests, all passing)
├── .github/workflows/   # CI/CD
├── pyproject.toml       # Project configuration
└── pixi.toml            # Pixi package manager configuration
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, testing, and code quality guidelines.

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

- Inspired by [micrograd](https://github.com/karpathy/micrograd) by Andrej Karpathy
- Based on [unnet](https://github.com/msaelices/unnet)
- Powered by [Mojo](https://www.modular.com/mojo)
