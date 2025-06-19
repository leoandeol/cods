# FILE: tests/conftest.py

import numpy as np
import pytest
import torch

from cods.od.data import ODPredictions

from cods.od.optimizers import (
    FirstStepMonotonizingOptimizer,
    SecondStepMonotonizingOptimizer,
)


# Fixture for the device
@pytest.fixture(scope="session")
def device():
    return "cpu"


# Fixture to create mock prediction data
@pytest.fixture
def mock_predictions(device):
    """Provides a consistent set of mock predictions for testing."""
    return ODPredictions(
        true_boxes=[
            torch.tensor(
                [[10, 10, 50, 50]], dtype=torch.float32, device=device
            ),
            torch.tensor(
                [[20, 20, 60, 60]], dtype=torch.float32, device=device
            ),
        ],
        pred_boxes=[
            torch.tensor(
                [[12, 12, 52, 52], [80, 80, 90, 90]],
                dtype=torch.float32,
                device=device,
            ),
            torch.tensor(
                [[25, 25, 65, 65], [100, 100, 110, 110]],
                dtype=torch.float32,
                device=device,
            ),
        ],
        confidences=[
            torch.tensor([0.9, 0.4], dtype=torch.float32, device=device),
            torch.tensor([0.8, 0.7], dtype=torch.float32, device=device),
        ],
        true_cls=[
            torch.tensor([[1.0, 0.0]], dtype=torch.float32, device=device),
            torch.tensor([[0.0, 1.0]], dtype=torch.float32, device=device),
        ],
        pred_cls=[
            torch.tensor(
                [[0.9, 0.1], [0.3, 0.7]], dtype=torch.float32, device=device
            ),
            torch.tensor(
                [[0.8, 0.2], [0.2, 0.8]], dtype=torch.float32, device=device
            ),
        ],
        image_shapes=[
            np.array([100, 100]),
            np.array([120, 120]),
        ],
    )


# Fixture for a mock loss function
@pytest.fixture
def mock_od_loss():
    """A mock loss class that returns a simple deterministic loss."""

    class MockODLoss:
        def __call__(self, true_boxes, true_cls, pred_boxes, pred_cls):
            if pred_boxes is None or pred_boxes.nelement() == 0:
                return torch.tensor(1.0)
            return torch.tensor(pred_boxes.shape[0] * 0.1)

    return MockODLoss()


# Fixture for a mock matching function (the name is what's important)
@pytest.fixture
def mock_matching_func():
    return lambda box1, box2: 0.0


# Fixture for a mock build predictions function
@pytest.fixture
def mock_build_predictions():
    def _build(matched_pred_boxes, matched_pred_cls, lbd):
        if matched_pred_boxes.nelement() == 0:
            return matched_pred_boxes, matched_pred_cls
        return matched_pred_boxes * (1 + lbd), matched_pred_cls

    return _build


# Fixtures for parameters
@pytest.fixture
def alpha():
    return 0.5


@pytest.fixture
def B():
    return 1.0


def test_first_step_optimizer(
    mock_predictions,
    mock_od_loss,
    mock_matching_func,
    alpha,
    device,
    B,
):
    """Tests the FirstStepMonotonizingOptimizer."""
    print("\n--- Testing FirstStepMonotonizingOptimizer ---")
    optimizer = FirstStepMonotonizingOptimizer()

    result_lambda = optimizer.optimize(
        predictions=mock_predictions,
        confidence_loss=mock_od_loss,
        localization_loss=mock_od_loss,
        classification_loss=mock_od_loss,
        matching_function=mock_matching_func,
        alpha=alpha,
        device=device,
        B=B,
        init_lambda=1.0,
        verbose=False,
    )

    print("FirstStepMonotonizingOptimizer finished.")
    print(f"Final Lambda (Confidence Threshold): {result_lambda}")

    assert isinstance(result_lambda, float)
    assert 0.0 <= result_lambda <= 1.0
    # The result should not be the default initial value if the loop runs
    assert result_lambda != 1.0


def test_second_step_optimizer(
    mock_predictions,
    mock_build_predictions,
    mock_od_loss,
    mock_matching_func,
    alpha,
    device,
    B,
):
    """Tests the SecondStepMonotonizingOptimizer."""
    print("\n--- Testing SecondStepMonotonizingOptimizer ---")
    optimizer = SecondStepMonotonizingOptimizer()

    # Set the pre-determined confidence threshold for this step
    mock_predictions.confidence_threshold = 0.25

    result_lambda = optimizer.optimize(
        predictions=mock_predictions,
        build_predictions=mock_build_predictions,
        loss=mock_od_loss,
        matching_function=mock_matching_func,
        alpha=alpha,
        device=device,
        B=B,
        bounds=[0.0, 1.0],
        steps=10,
        verbose=False,
    )

    print("SecondStepMonotonizingOptimizer finished.")
    print(f"Final Lambda (Localization/Classification): {result_lambda}")

    assert isinstance(result_lambda, float)
    assert 0.0 <= result_lambda <= 1.0
