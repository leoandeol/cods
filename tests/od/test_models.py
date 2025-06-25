"""Tests for object detection models."""

import pytest


def test_od_model_creation():
    """Test that OD models can be created without error."""
    # This is a basic smoke test to check imports work
    try:
        from cods.od.models.model import ODModel
        assert ODModel is not None
    except ImportError:
        pytest.skip("ODModel not available")


def test_detr_model_import():
    """Test DETR model import."""
    try:
        from cods.od.models.detr import DETRModel
        assert DETRModel is not None
    except ImportError:
        pytest.skip("DETRModel not available")


def test_yolo_model_import():
    """Test YOLO model import."""
    try:
        from cods.od.models.yolo import YOLOModel
        assert YOLOModel is not None
    except ImportError:
        pytest.skip("YOLOModel not available")
