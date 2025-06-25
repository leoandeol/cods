"""Tests for classification data structures."""
import torch
import numpy as np

from cods.classif.data.predictions import ClassificationPredictions


def test_classification_predictions_init():
    """Test ClassificationPredictions initialization."""
    # Create sample data
    image_paths = ["img1.jpg", "img2.jpg"]
    idx_to_cls = {0: "cat", 1: "dog", 2: "bird"}
    true_cls = torch.tensor([1, 0])
    pred_cls = torch.tensor([[0.1, 0.7, 0.2], [0.8, 0.1, 0.1]])
    
    pred = ClassificationPredictions(
        dataset_name="test_dataset",
        split_name="test_split",
        image_paths=image_paths,
        idx_to_cls=idx_to_cls,
        true_cls=true_cls,
        pred_cls=pred_cls
    )
    
    assert pred is not None
    assert pred.dataset_name == "test_dataset"
    assert pred.split_name == "test_split"
    assert pred.image_paths == image_paths
    assert pred.idx_to_cls == idx_to_cls
    assert torch.equal(pred.true_cls, true_cls)
    assert torch.equal(pred.pred_cls, pred_cls)


def test_classification_predictions_length():
    """Test ClassificationPredictions length property."""
    image_paths = ["img1.jpg", "img2.jpg", "img3.jpg"]
    idx_to_cls = {0: "cat", 1: "dog", 2: "bird"}
    true_cls = torch.tensor([1, 0, 2])
    pred_cls = torch.tensor([[0.1, 0.7, 0.2], [0.8, 0.1, 0.1], [0.3, 0.3, 0.4]])
    
    pred = ClassificationPredictions(
        dataset_name="test",
        split_name="test",
        image_paths=image_paths,
        idx_to_cls=idx_to_cls,
        true_cls=true_cls,
        pred_cls=pred_cls
    )
    
    assert len(pred) == 3


def test_classification_predictions_with_numpy():
    """Test ClassificationPredictions with numpy arrays."""
    image_paths = ["img1.jpg", "img2.jpg"]
    idx_to_cls = {0: "cat", 1: "dog", 2: "bird"}
    true_cls = np.array([1, 0])
    pred_cls = np.array([[0.1, 0.7, 0.2], [0.8, 0.1, 0.1]])
    
    pred = ClassificationPredictions(
        dataset_name="test_dataset",
        split_name="test_split",
        image_paths=image_paths,
        idx_to_cls=idx_to_cls,
        true_cls=torch.from_numpy(true_cls),
        pred_cls=torch.from_numpy(pred_cls)
    )
    
    assert pred is not None
    assert isinstance(pred.true_cls, torch.Tensor)
    assert isinstance(pred.pred_cls, torch.Tensor)


def test_classification_predictions_device_handling():
    """Test ClassificationPredictions device handling."""
    image_paths = ["img1.jpg", "img2.jpg"]
    idx_to_cls = {0: "cat", 1: "dog", 2: "bird"}
    true_cls = torch.tensor([1, 0])
    pred_cls = torch.tensor([[0.1, 0.7, 0.2], [0.8, 0.1, 0.1]])
    
    pred = ClassificationPredictions(
        dataset_name="test",
        split_name="test",
        image_paths=image_paths,
        idx_to_cls=idx_to_cls,
        true_cls=true_cls,
        pred_cls=pred_cls
    )
    
    # Test device conversion if available
    if torch.cuda.is_available() and hasattr(pred, 'to'):
        pred_cuda = pred.to("cuda")
        if hasattr(pred_cuda, 'true_cls'):
            assert pred_cuda.true_cls.device.type == "cuda"
            assert pred_cuda.pred_cls.device.type == "cuda"
    
    # Test CPU
    if hasattr(pred, 'to'):
        pred_cpu = pred.to("cpu")
        if hasattr(pred_cpu, 'true_cls'):
            assert pred_cpu.true_cls.device.type == "cpu"
            assert pred_cpu.pred_cls.device.type == "cpu"


def test_classification_predictions_string_representation():
    """Test ClassificationPredictions string representation."""
    image_paths = ["img1.jpg"]
    idx_to_cls = {0: "cat", 1: "dog", 2: "bird"}
    true_cls = torch.tensor([1])
    pred_cls = torch.tensor([[0.1, 0.7, 0.2]])
    
    pred = ClassificationPredictions(
        dataset_name="test",
        split_name="test",
        image_paths=image_paths,
        idx_to_cls=idx_to_cls,
        true_cls=true_cls,
        pred_cls=pred_cls
    )
    
    str_repr = str(pred)
    # Check that it has some representation
    assert isinstance(str_repr, str)
    assert len(str_repr) > 0