import numpy as np
import torch

from cods.base.loss import Loss
from cods.od.utils import get_covered_areas_of_gt_max, get_covered_areas_of_gt_union


# Object Detection Loss, many are wrappers of Segmentation losses
class LocalizationLoss(Loss):
    def __init__(self, upper_bound: int, **kwargs):
        """
        Initialize the Object Detection Loss.

        Parameters:
        - upper_bound (int): The upper bound of the loss.

        Returns:
        - None
        """
        super().__init__()
        self.upper_bound = upper_bound

    def __call__(self, **kwargs):
        """
        Call the Object Detection Loss.

        Returns:
        - None
        """
        raise NotImplementedError("ODLoss is an abstract class.")


# LAC STYLE
class ConfidenceLoss(Loss):
    def __init__(self, upper_bound: int = 1, **kwargs):
        """
        Initialize the Confidence Loss.

        Parameters:
        - upper_bound (int): The upper bound of the loss.

        Returns:
        - None
        """
        super().__init__()
        self.upper_bound = upper_bound

    def __call__(
        self, n_gt: int, confidence: torch.Tensor, lbd: float, **kwargs
    ) -> torch.Tensor:
        """
        Call the Confidence Loss.

        Parameters:
        - n_gt (int): The number of ground truth objects.
        - confidence (torch.Tensor): The confidence scores.
        - lbd (float): The lambda value.

        Returns:
        - torch.Tensor: The loss value.
        """
        conf = torch.sort(confidence)[0][-n_gt]
        return torch.zeros(1).cuda() if conf >= 1 - lbd else torch.ones(1).cuda()


# Todo: formulate in classical conformal sense!
class HausdorffSignedDistanceLoss(LocalizationLoss):
    def __init__(self, beta: float = 0.25):
        """
        Initialize the Hausdorff Signed Distance Loss.

        Parameters:
        - beta (float): The beta value.

        Returns:
        - None
        """
        self.upper_bound = 1
        self.beta = beta

    def __call__(self, conf_boxes: torch.Tensor, true_boxes: torch.Tensor) -> float:
        """
        Call the Hausdorff Signed Distance Loss.

        Parameters:
        - conf_boxes (torch.Tensor): The conformal boxes.
        - true_boxes (torch.Tensor): The true boxes.

        Returns:
        - float: The loss value.
        """
        if len(true_boxes) == 0:
            return 0
        elif len(conf_boxes) == 0:
            return 1
        else:
            areas = get_covered_areas_of_gt_union(conf_boxes, true_boxes)
            is_not_covered = (
                torch.FloatTensor(areas) < 0.999
            )  # because doubt on the computation of the overlap, check formula TODO
            miscoverage = torch.mean(is_not_covered)
            loss = 1 if miscoverage > self.beta else 0
            return loss


class BoxWiseRecallLoss(LocalizationLoss):
    def __init__(self, union_of_boxes: bool = True):
        """
        Initialize the Box-wise Recall Loss.

        Parameters:
        - union_of_boxes (bool): Whether to use the union of boxes.

        Returns:
        - None
        """
        self.upper_bound = 1
        self.union_of_boxes = union_of_boxes
        self.get_covered_areas = (
            get_covered_areas_of_gt_union
            if union_of_boxes
            else get_covered_areas_of_gt_max
        )

    def __call__(
        self, conf_boxes: torch.Tensor, true_boxes: torch.Tensor
    ) -> torch.Tensor:
        """
        Call the Box-wise Recall Loss.

        Parameters:
        - conf_boxes (torch.Tensor): The conformal boxes.
        - true_boxes (torch.Tensor): The true boxes.

        Returns:
        - float: The loss value.
        """
        if len(true_boxes) == 0:
            return torch.zeros(1).cuda()
        elif len(conf_boxes) == 0:
            return torch.ones(1).cuda()
        else:
            areas = self.get_covered_areas(conf_boxes, true_boxes)
            is_not_covered = (
                areas < 0.999
            )  # because doubt on the computation of the overlap, check formula TODO
            miscoverage = torch.mean(is_not_covered)
            return miscoverage


class PixelWiseRecallLoss(LocalizationLoss):
    def __init__(self, union_of_boxes: bool = True):
        """
        Initialize the Pixel-wise Recall Loss.

        Parameters:
        - union_of_boxes (bool): Whether to use the union of boxes.

        Returns:
        - None
        """
        self.upper_bound = 1
        self.union_of_boxes = union_of_boxes
        self.get_covered_areas = (
            get_covered_areas_of_gt_union
            if union_of_boxes
            else get_covered_areas_of_gt_max
        )

    def __call__(
        self, conf_boxes: torch.Tensor, true_boxes: torch.Tensor
    ) -> torch.Tensor:
        """
        Call the Pixel-wise Recall Loss.

        Parameters:
        - conf_boxes (torch.Tensor): The conformal boxes.
        - true_boxes (torch.Tensor): The true boxes.

        Returns:
        - torch.Tensor: The loss value.
        """
        if len(true_boxes) == 0:
            return torch.zeros(1).cuda()
        elif len(conf_boxes) == 0:
            return torch.ones(1).cuda()
        else:
            areas = self.get_covered_areas(conf_boxes, true_boxes)
            loss = torch.ones(1).cuda() - torch.mean(areas)
            return loss


class NumberPredictionsGapLoss(LocalizationLoss):
    def __init__(self):
        """
        Initialize the Number Predictions Gap Loss.

        Returns:
        - None
        """
        self.upper_bound = 1

    def __call__(self, conf_boxes: torch.Tensor, true_boxes: torch.Tensor) -> float:
        """
        Call the Number Predictions Gap Loss.

        Parameters:
        - conf_boxes (torch.Tensor): The conformal boxes.
        - true_boxes (torch.Tensor): The true boxes.

        Returns:
        - float: The loss value.
        """
        raise NotImplementedError("NumberPredictionsGapLoss is not implemented yet")
        loss = (len(true_boxes) - len(conf_boxes)) / max(len(true_boxes), 1)
        return min(loss, 1)
