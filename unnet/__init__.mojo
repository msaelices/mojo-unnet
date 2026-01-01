"""Mojo-unnet: Micro Neural Network in pure Mojo.

Educational proof-of-concept for learning neural networks implemented entirely in Mojo.
"""

from .grad import (
    Node,
    Op,
    get_global_registry_ptr,
    clear_global_registry,
)
from .nn import NetworkMLP, Layer, Neuron
from .uuid import UUID
