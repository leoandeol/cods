"""Tests for base conformal prediction classes."""
import pytest

from cods.base.cp import Conformalizer


def test_conformalizer_instantiation():
    """Test that Conformalizer can be instantiated."""
    conf = Conformalizer()
    assert conf is not None


def test_conformalizer_abstract_methods():
    """Test that Conformalizer has required abstract methods."""
    # Check that the abstract methods exist
    assert hasattr(Conformalizer, 'calibrate')
    assert hasattr(Conformalizer, 'conformalize')
    assert hasattr(Conformalizer, 'evaluate')


class MockConformalizer(Conformalizer):
    """Mock implementation of Conformalizer for testing."""
    
    def calibrate(self, predictions, alpha=0.1, **kwargs):
        """Mock calibrate method."""
        return 0.5
    
    def conformalize(self, predictions, **kwargs):
        """Mock conformalize method."""
        return predictions
    
    def evaluate(self, predictions, verbose=True):
        """Mock evaluate method."""
        return {"coverage": 0.9, "efficiency": 0.8}


def test_mock_conformalizer():
    """Test that MockConformalizer can be instantiated."""
    mock_conf = MockConformalizer()
    assert mock_conf is not None
    
    # Test abstract method implementations
    result = mock_conf.calibrate(None)
    assert result == 0.5
    
    predictions = "test_predictions"
    conf_result = mock_conf.conformalize(predictions)
    assert conf_result == predictions
    
    eval_result = mock_conf.evaluate(None)
    assert isinstance(eval_result, dict)
    assert "coverage" in eval_result
    assert "efficiency" in eval_result