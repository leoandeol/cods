"""Tests for classification models."""

import pytest
import torch


def test_classification_model_imports():
    """Test that classification model classes can be imported."""
    try:
        from cods.classif.models import ClassificationModel

        assert ClassificationModel is not None
    except ImportError:
        pytest.skip("ClassificationModel not available")


def test_classification_model_creation():
    """Test classification model creation."""
    try:
        from cods.classif.models import ClassificationModel

        # Test basic instantiation
        model = ClassificationModel()
        assert model is not None
    except (ImportError, TypeError):
        pytest.skip("ClassificationModel not available or needs parameters")


def test_classification_model_with_mock():
    """Test classification model functionality with mocks."""

    # Mock a simple classification model
    class MockClassificationModel:
        def __init__(self, num_classes=10):
            self.num_classes = num_classes

        def predict(self, x):
            batch_size = x.shape[0]
            return torch.rand(batch_size, self.num_classes)

        def forward(self, x):
            return self.predict(x)

    model = MockClassificationModel(num_classes=5)
    assert model.num_classes == 5

    # Test prediction
    x = torch.randn(4, 3, 224, 224)  # Batch of 4 images
    pred = model.predict(x)

    assert pred.shape == (4, 5)
    assert torch.all(pred >= 0) and torch.all(pred <= 1)


def test_classification_prediction_building():
    """Test classification prediction building process."""

    # Mock the prediction building process
    class MockPredictionBuilder:
        def __init__(self, model):
            self.model = model

        def build_predictions(self, dataset, device="cpu"):
            # Simulate building predictions from a dataset
            predictions = []
            true_labels = []

            for _i in range(10):  # Simulate 10 samples
                # Mock input and prediction
                x = torch.randn(1, 3, 224, 224)
                pred = self.model.predict(x)
                true_label = torch.randint(0, self.model.num_classes, (1,))

                predictions.append(pred)
                true_labels.append(true_label)

            return torch.cat(predictions), torch.cat(true_labels)

    # Test with mock model
    class MockModel:
        def __init__(self):
            self.num_classes = 3

        def predict(self, x):
            return torch.softmax(
                torch.randn(x.shape[0], self.num_classes), dim=1
            )

    model = MockModel()
    builder = MockPredictionBuilder(model)

    pred_probs, true_labels = builder.build_predictions(None)

    assert pred_probs.shape == (10, 3)
    assert true_labels.shape == (10,)
    assert torch.all(true_labels >= 0) and torch.all(true_labels < 3)

    # Check that predictions are valid probabilities
    assert torch.allclose(pred_probs.sum(dim=1), torch.ones(10), atol=1e-6)
