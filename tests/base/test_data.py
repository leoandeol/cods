"""Tests for base data structures."""

from cods.base.data import (
    ConformalizedPredictions,
    Parameters,
    Predictions,
    Results,
)


def test_predictions_init():
    """Test Predictions initialization."""
    pred = Predictions("test_dataset", "test_split", "test_task")
    assert pred is not None
    assert pred.dataset_name == "test_dataset"
    assert pred.split_name == "test_split"
    assert pred.task_name == "test_task"


def test_parameters_init():
    """Test Parameters initialization."""
    param = Parameters(predictions_id=123)
    assert param is not None
    assert param.predictions_id == 123


def test_conformalized_predictions_init():
    """Test ConformalizedPredictions initialization."""
    conf_pred = ConformalizedPredictions(predictions_id=123, parameters_id=456)
    assert conf_pred is not None
    assert conf_pred.predictions_id == 123
    assert conf_pred.parameters_id == 456


def test_results_init():
    """Test Results initialization."""
    result = Results(
        predictions_id=123, parameters_id=456, conformalized_id=789
    )
    assert result is not None
    assert result.predictions_id == 123
    assert result.parameters_id == 456
    assert result.conformalized_id == 789
