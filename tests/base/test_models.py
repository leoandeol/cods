"""Tests for base model classes."""
import torch
import pytest
from unittest.mock import Mock, patch

from cods.base.models import Model


def test_model_is_abstract():
    """Test that Model is an abstract base class."""
    with pytest.raises(TypeError):
        Model()


def test_model_abstract_methods():
    """Test that Model has required abstract methods."""
    assert hasattr(Model, 'build_predictions')


class MockModel(Model):
    """Mock implementation of Model for testing."""
    
    def __init__(self, model_name="test_model", save_dir_path="/tmp"):
        super().__init__(model_name, save_dir_path)
    
    def build_predictions(self, dataset, device="cpu", **kwargs):
        """Mock build_predictions method."""
        # Simulate building predictions
        return {
            "predictions": torch.randn(10, 5),
            "targets": torch.randint(0, 5, (10,)),
            "metadata": {"model": self.model_name}
        }


def test_mock_model():
    """Test MockModel implementation."""
    model = MockModel("test_classifier")
    assert model is not None
    assert model.model_name == "test_classifier"
    
    # Test build_predictions
    result = model.build_predictions(None)
    
    assert isinstance(result, dict)
    assert "predictions" in result
    assert "targets" in result
    assert "metadata" in result
    
    predictions = result["predictions"]
    targets = result["targets"]
    
    assert predictions.shape == (10, 5)
    assert targets.shape == (10,)
    assert torch.all(targets >= 0) and torch.all(targets < 5)


def test_model_with_device():
    """Test model behavior with different devices."""
    model = MockModel()
    
    # Test with CPU
    result_cpu = model.build_predictions(None, device="cpu")
    predictions_cpu = result_cpu["predictions"]
    assert predictions_cpu.device.type == "cpu"
    
    # Test with CUDA if available
    if torch.cuda.is_available():
        result_cuda = model.build_predictions(None, device="cuda")
        predictions_cuda = result_cuda["predictions"]
        # Note: Our mock doesn't actually use device, but this tests the interface
        assert "predictions" in result_cuda


def test_model_caching_interface():
    """Test model caching interface concepts."""
    # Test that models can implement caching mechanisms
    class CachingMockModel(MockModel):
        def __init__(self, model_name="cached_model", save_dir_path="/tmp"):
            super().__init__(model_name, save_dir_path)
            self.cache = {}
        
        def build_predictions(self, dataset, device="cpu", use_cache=True, **kwargs):
            cache_key = f"{dataset}_{device}"
            
            if use_cache and cache_key in self.cache:
                return self.cache[cache_key]
            
            result = super().build_predictions(dataset, device, **kwargs)
            
            if use_cache:
                self.cache[cache_key] = result
            
            return result
        
        def clear_cache(self):
            """Clear the model cache."""
            self.cache = {}
    
    model = CachingMockModel()
    
    # First call should compute
    result1 = model.build_predictions("dataset1", use_cache=True)
    assert len(model.cache) == 1
    
    # Second call should use cache
    result2 = model.build_predictions("dataset1", use_cache=True)
    assert torch.equal(result1["predictions"], result2["predictions"])
    
    # Clear cache
    model.clear_cache()
    assert len(model.cache) == 0


def test_model_error_handling():
    """Test model error handling."""
    class ErrorModel(Model):
        def __init__(self, model_name="error_model", save_dir_path="/tmp"):
            super().__init__(model_name, save_dir_path)
            
        def build_predictions(self, dataset, device="cpu", **kwargs):
            if dataset is None:
                raise ValueError("Dataset cannot be None")
            return {"status": "success"}
    
    model = ErrorModel()
    
    # Test successful case
    result = model.build_predictions("valid_dataset")
    assert result["status"] == "success"
    
    # Test error case
    with pytest.raises(ValueError, match="Dataset cannot be None"):
        model.build_predictions(None)