"""Tests for optimization functions."""

from cods.base.optim import (
    BinarySearchOptimizer,
    GaussianProcessOptimizer,
    MonteCarloOptimizer,
)


def test_binary_search_optimizer_init():
    """Test BinarySearchOptimizer initialization."""
    optimizer = BinarySearchOptimizer()
    assert optimizer is not None


def test_gaussian_process_optimizer_init():
    """Test GaussianProcessOptimizer initialization."""
    optimizer = GaussianProcessOptimizer()
    assert optimizer is not None


def test_monte_carlo_optimizer_init():
    """Test MonteCarloOptimizer initialization."""
    optimizer = MonteCarloOptimizer()
    assert optimizer is not None
