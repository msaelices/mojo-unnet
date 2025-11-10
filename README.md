# mojo-unnet

Micro Neural Network in pure Mojo - Educational proof-of-concept for learning neural networks.

This is a pure Mojo implementation inspired by [unnet](https://github.com/yourusername/unnet), featuring all neural network operations implemented natively in Mojo for maximum performance.

## Implementation Status

This project provides scaffolding with empty implementations. The following components need to be implemented:

### `grad.mojo` - Autograd
- [ ] Node struct with computation graph tracking
- [ ] Operator overloading (+, -, *, ^)
- [ ] Activation functions (tanh)
- [ ] Backward propagation

### `nn.mojo` - Neural Network Components
- [ ] Neuron struct (weights, bias, forward pass)
- [ ] Layer struct (multiple neurons)
- [ ] Network struct (multiple layers)
- [ ] Training loop with gradient descent

### `utils.mojo` - Utilities
- [ ] Graph traversal (walk)
- [ ] Visualization helpers

## Features

- **Pure Mojo**: Entire codebase written in Mojo for maximum performance
- **Educational**: Clear, simple implementation for learning neural network fundamentals
- **Autograd**: Automatic differentiation for backpropagation
- **Computational Graph**: Node-based computation graph with gradient tracking

## Installation

### Prerequisites

- [Mojo](https://docs.modular.com/mojo/)
- [uv](https://github.com/astral-sh/uv) package manager (for dev tools)

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/mojo-unnet.git
cd mojo-unnet

# Install development dependencies (mblack, mojo-compiler)
uv sync --all-groups
```

## Usage

```mojo
from unnet import Network

# TODO: Add usage example once implementation is complete
fn main():
    print("mojo-unnet - Neural networks in pure Mojo!")
```

## Project Structure

```
mojo-unnet/
├── unnet/          # Main package (pure Mojo)
│   ├── __init__.mojo    # Package initialization
│   ├── grad.mojo        # Autograd implementation (Node struct)
│   ├── nn.mojo          # Neural network components (Neuron, Layer, Network)
│   └── utils.mojo       # Utilities (graph traversal, visualization)
├── tests/               # Test suite (Mojo tests)
├── .github/workflows/   # CI/CD
└── pyproject.toml       # Project configuration
```

## Development

### Setup Development Environment

```bash
# Install all dependencies including dev tools
uv sync --all-groups

# Install pre-commit hooks
uv run pre-commit install
```

### Code Quality

```bash
# Format Mojo files
uv run mojo format unnet/

# Check formatting
uv run mojo format --check unnet/

# Run all pre-commit checks
uv run pre-commit run --all-files
```

### Building

```bash
# Build Mojo package
uv run mojo package unnet -o unnet.mojopkg
```


## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development guidelines.

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

- Inspired by [micrograd](https://github.com/karpathy/micrograd) by Andrej Karpathy
- Based on [unnet](https://github.com/msaelices/unnet)
- Powered by [Mojo](https://www.modular.com/mojo)
