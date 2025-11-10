# Contributing to mojo-unnet

Thank you for your interest in contributing to mojo-unnet! This document provides guidelines and instructions for setting up your development environment and contributing to the project.

## Development Environment Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/mojo-unnet.git
cd mojo-unnet
```

2. Install Mojo SDK:
Follow the instructions at https://docs.modular.com/mojo/

3. Install development dependencies:
```bash
uv sync --all-groups
```

This will install development tools (mblack formatter, mojo-compiler).

## Code Quality

### Formatting

Format Mojo files:
```bash
uv run mojo format unnet/
```

Check formatting without modifying files:
```bash
uv run mojo format --check unnet/
```

### Building

Build the Mojo package:
```bash
uv run mojo package unnet -o unnet.mojopkg
```

## Pre-commit Hooks

We use pre-commit hooks to automatically check code quality before commits.

### Install pre-commit hooks:
```bash
uv run pre-commit install
```

### Run pre-commit on all files manually:
```bash
uv run pre-commit run --all-files
```
The pre-commit hooks will automatically run on every commit and check:
- UV lock file sync
- Mojo formatting

### Running Tests

```bash
# Run one test file
uv run mojo tests/test_utils.mojo
```

## Continuous Integration

All pull requests and pushes to the repository automatically run our CI pipeline via GitHub Actions. The CI workflow includes two jobs:

### Lint Job
- Runs all pre-commit checks

### Build Job
- Checks Mojo formatting (`mojo format --check`)
- Builds the Mojo package

You can view the CI status in the Actions tab of the GitHub repository. Make sure all checks pass before submitting a pull request.

## Implementation Guidelines

This project is in the early stages with scaffolding code. When implementing features:

1. **Follow Mojo best practices**: Use Mojo's type system, lifetimes, and ownership model correctly
2. **Keep it educational**: Code should be clear and easy to understand for learning purposes
3. **Document thoroughly**: Add docstrings and comments explaining the neural network concepts
4. **Test your changes**: Add tests as you implement features
5. **Format before committing**: Always run `mojo format` on your code

## Questions or Issues?

If you have questions or run into issues, please open an issue on GitHub.
