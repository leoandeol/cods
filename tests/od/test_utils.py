"""Tests for object detection utility functions."""

import torch

from cods.od.utils import (
    apply_margins,
    assymetric_hausdorff_distance,
    f_iou,
    f_lac,
    generalized_iou,
    rank_distance,
)


def test_f_iou():
    """Test IoU calculation function."""
    # Create simple test boxes
    box1 = torch.tensor([[0, 0, 10, 10]])  # Box from (0,0) to (10,10)
    box2 = torch.tensor([[5, 5, 15, 15]])  # Box from (5,5) to (15,15)

    iou = f_iou(box1, box2)

    # Expected IoU: intersection area = 5*5 = 25, union area = 100 + 100 - 25 = 175
    # IoU = 25/175 ≈ 0.143
    assert isinstance(iou, torch.Tensor)
    assert 0.1 < iou.item() < 0.2


def test_f_iou_no_overlap():
    """Test IoU with non-overlapping boxes."""
    box1 = torch.tensor([[0, 0, 5, 5]])
    box2 = torch.tensor([[10, 10, 15, 15]])

    iou = f_iou(box1, box2)
    assert iou.item() == 0.0


def test_f_iou_identical_boxes():
    """Test IoU with identical boxes."""
    box1 = torch.tensor([[0, 0, 10, 10]])
    box2 = torch.tensor([[0, 0, 10, 10]])

    iou = f_iou(box1, box2)
    assert iou.item() == 1.0


def test_generalized_iou():
    """Test GIoU calculation function."""
    # Create test boxes as tensors
    box_a = torch.tensor([0, 0, 10, 10])
    box_b = torch.tensor([5, 5, 15, 15])

    giou = generalized_iou(box_a, box_b)

    assert isinstance(giou, torch.Tensor)
    # GIoU should be between -1 and 1
    assert -1 <= giou.item() <= 1


def test_f_lac():
    """Test LAC score calculation."""
    true_cls = torch.tensor([[1], [0]])
    pred_cls = torch.tensor([[0.1, 0.8, 0.1], [0.9, 0.05, 0.05]])

    result = f_lac(true_cls, pred_cls)

    assert isinstance(result, torch.Tensor)
    assert result.shape == (2, 2)
    # Values should be between 0 and 1 (1 - probability)
    assert torch.all(result >= 0)
    assert torch.all(result <= 1)


def test_rank_distance():
    """Test rank distance calculation."""
    true_cls = torch.tensor([[1], [0]])
    pred_cls = torch.tensor([[0.1, 0.8, 0.1], [0.9, 0.05, 0.05]])

    ranks = rank_distance(true_cls, pred_cls)

    assert isinstance(ranks, torch.Tensor)
    # Ranks should be non-negative integers
    assert torch.all(ranks >= 0)


def test_apply_margins_additive():
    """Test additive margin application."""
    pred_boxes = [torch.tensor([[10, 10, 20, 20]])]
    margins = [2, 2, 2, 2]

    result = apply_margins(pred_boxes, margins, mode="additive")

    assert len(result) == 1
    expected = torch.tensor([[8, 8, 22, 22]])  # [10-2, 10-2, 20+2, 20+2]
    assert torch.allclose(result[0], expected)


def test_apply_margins_multiplicative():
    """Test multiplicative margin application."""
    pred_boxes = [torch.tensor([[10, 10, 20, 20]])]
    margins = [1.1, 1.1, 1.1, 1.1]

    result = apply_margins(pred_boxes, margins, mode="multiplicative")

    assert len(result) == 1
    # For multiplicative, the box should expand proportionally
    assert result[0][0] < 10  # x1 should decrease
    assert result[0][1] < 10  # y1 should decrease
    assert result[0][2] > 20  # x2 should increase
    assert result[0][3] > 20  # y2 should increase


def test_apply_margins_empty_boxes():
    """Test margin application with empty boxes."""
    pred_boxes = [torch.tensor([])]
    margins = [2, 2, 2, 2]

    result = apply_margins(pred_boxes, margins, mode="additive")

    assert len(result) == 1
    assert result[0].numel() == 0


def test_assymetric_hausdorff_distance():
    """Test asymmetric Hausdorff distance calculation."""
    true_boxes = torch.tensor([[0, 0, 10, 10], [5, 5, 15, 15]])
    pred_boxes = torch.tensor([[1, 1, 11, 11], [4, 4, 14, 14]])

    distances = assymetric_hausdorff_distance(true_boxes, pred_boxes)

    assert isinstance(distances, torch.Tensor)
    assert distances.shape == (2, 2)
    # Distances should be non-negative
    assert torch.all(distances >= 0)


def test_device_consistency():
    """Test that functions handle device consistency properly."""
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    # Test with tensors on specific device
    pred_boxes = [torch.tensor([[10, 10, 20, 20]]).to(device)]
    margins = [2, 2, 2, 2]

    result = apply_margins(pred_boxes, margins, mode="additive")

    assert result[0].device.type == device
