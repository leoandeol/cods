"""Tests for object detection data structures."""
import torch

from cods.od.data.predictions import (
    ODConformalizedPredictions,
    ODParameters,
    ODPredictions,
    ODResults,
)


def test_od_predictions_import():
    """Test ODPredictions can be imported."""
    assert ODPredictions is not None


def test_od_parameters_simple():
    """Test ODParameters initialization with required args."""
    param = ODParameters(
        global_alpha=0.1,
        confidence_threshold=0.5,
        predictions_id=123
    )
    assert param is not None
    assert param.global_alpha == 0.1
    assert param.confidence_threshold == 0.5
    assert param.predictions_id == 123


def test_od_predictions_with_minimal_data():
    """Test ODPredictions initialization with minimal data."""
    # Create minimal test data
    pred = ODPredictions(
        dataset_name="test_dataset",
        split_name="test_split",
        image_paths=["test.jpg"],
        image_shapes=[torch.tensor([100, 100])],
        true_boxes=[torch.tensor([[0, 0, 10, 10]])],
        pred_boxes=[torch.tensor([[1, 1, 9, 9]])],
        confidences=[torch.tensor([0.8])],
        true_cls=[torch.tensor([1])],
        pred_cls=[torch.tensor([[1, 0]])],
        names=["class1"]
    )
    assert pred is not None
    assert pred.dataset_name == "test_dataset"
    assert pred.split_name == "test_split"


def test_od_conformalized_predictions_init():
    """Test ODConformalizedPredictions can be created."""
    # Create minimal ODPredictions first
    predictions = ODPredictions(
        dataset_name="test",
        split_name="test",
        image_paths=["test.jpg"],
        image_shapes=[torch.tensor([100, 100])],
        true_boxes=[torch.tensor([[0, 0, 10, 10]])],
        pred_boxes=[torch.tensor([[1, 1, 9, 9]])],
        confidences=[torch.tensor([0.8])],
        true_cls=[torch.tensor([1])],
        pred_cls=[torch.tensor([[1, 0]])],
        names=["class1"]
    )

    # Create ODParameters
    parameters = ODParameters(
        global_alpha=0.1,
        confidence_threshold=0.5,
        predictions_id=123
    )

    # Create ODConformalizedPredictions
    conf_pred = ODConformalizedPredictions(
        predictions=predictions,
        parameters=parameters
    )
    assert conf_pred is not None


def test_od_classes_import():
    """Test all OD classes can be imported."""
    assert ODPredictions is not None
    assert ODParameters is not None
    assert ODConformalizedPredictions is not None
    assert ODResults is not None
