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

### `nn.mojo` - Neural Network Components
- [ ] Neuron struct (weights, bias, forward pass)
- [ ] Layer struct (multiple neurons)
- [ ] Network struct (multiple layers)
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
d.name = "d"
var e = d + c
e.name = "e"

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
from unnet import Network

# TODO: Neural network components not yet implemented
# This section will be updated once Neuron, Layer, and Network are implemented
```

## Project Structure

```
mojo-unnet/
├── unnet/               # Main package (pure Mojo)
│   ├── __init__.mojo    # Package initialization
│   ├── grad.mojo        # Autograd implementation (Node, Op, GradRegistry)
│   ├── uuid.mojo        # UUID generation and comparison
│   ├── nn.mojo          # Neural network components (Neuron, Layer, Network) - TODO
│   └── utils.mojo       # Utilities (graph traversal, graphviz visualization)
├── tests/               # Test suite (Mojo tests)
│   ├── test_grad.mojo   # Tests for autograd (12 tests, all passing)
│   └── test_utils.mojo  # Tests for utilities (9 tests, all passing)
├── .github/workflows/   # CI/CD
├── pyproject.toml       # Project configuration
└── pixi.toml            # Pixi package manager configuration
```

## Development

### Setup Development Environment

```bash
# Install all dependencies including dev tools
pixi install

# Install pre-commit hooks
pixi run pre-commit install
```

### Code Quality

```bash
# Format Mojo files
pixi run mojo format unnet/

# Check formatting
pixi run mojo format --check unnet/

# Run all pre-commit checks
pixi run pre-commit run --all-files
```

### Building

```bash
# Build Mojo package
pixi run mojo package unnet -o unnet.mojopkg
```

### Testing

```bash
# Run all tests
pixi run test

# Run specific test file
mojo -I . tests/test_grad.mojo
mojo -I . tests/test_utils.mojo
```

Current test status:
- **test_grad.mojo**: 12 tests, all passing
  - Basic operations (add, multiply, subtract, tanh)
  - Complex graph backpropagation
  - Gradient accumulation
  - Graph traversal (walk, walk_topo)
  - Gradient zeroing
- **test_utils.mojo**: 9 tests, all passing
  - Node creation and operations
  - Graph traversal (walk)
  - Graph visualization (draw)


## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development guidelines.

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

- Inspired by [micrograd](https://github.com/karpathy/micrograd) by Andrej Karpathy
- Based on [unnet](https://github.com/msaelices/unnet)
- Powered by [Mojo](https://www.modular.com/mojo)
