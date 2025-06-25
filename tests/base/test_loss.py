"""Tests for base loss functions."""
import torch
import pytest

from cods.base.loss import NCScore


def test_ncscore_instantiation():
    """Test that NCScore can be instantiated."""
    score = NCScore()
    assert score is not None


def test_ncscore_methods():
    """Test that NCScore has expected methods."""
    # Check that these attributes exist (they might be implemented by subclasses)
    assert hasattr(NCScore, '__init__')


class MockNCScore(NCScore):
    """Mock implementation of NCScore for testing."""
    
    def __call__(self, y_pred, y_true, **kwargs):
        """Mock score calculation."""
        return torch.tensor([0.1, 0.2, 0.3])
    
    def get_set(self, y_pred, quantile, **kwargs):
        """Mock prediction set generation."""
        return torch.tensor([0, 1])


def test_mock_ncscore():
    """Test MockNCScore implementation."""
    mock_score = MockNCScore()
    assert mock_score is not None
    
    # Test score calculation
    y_pred = torch.tensor([[0.1, 0.7, 0.2]])
    y_true = torch.tensor([1])
    
    scores = mock_score(y_pred, y_true)
    assert isinstance(scores, torch.Tensor)
    assert len(scores) == 3
    
    # Test prediction set generation
    pred_set = mock_score.get_set(y_pred, quantile=0.5)
    assert isinstance(pred_set, torch.Tensor)
    assert len(pred_set) == 2


def test_ncscore_device_parameter():
    """Test NCScore with device parameter."""
    # Test that we can pass device parameter without error
    class TestNCScore(NCScore):
        def __init__(self, device="cpu"):
            self.device = device
        
        def __call__(self, y_pred, y_true, **kwargs):
            return torch.tensor([0.1]).to(self.device)
        
        def get_set(self, y_pred, quantile, **kwargs):
            return torch.tensor([0]).to(self.device)
    
    score_cpu = TestNCScore(device="cpu")
    assert score_cpu.device == "cpu"
    
    result = score_cpu(None, None)
    assert result.device.type == "cpu"