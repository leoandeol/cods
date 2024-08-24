from typing import List, Optional

import numpy as np
import torch

from cods.base.loss import Loss
from cods.od.data import ODConformalizedPredictions, ODPredictions
from cods.od.utils import (
    get_covered_areas_of_gt_max,
    get_covered_areas_of_gt_union,
)


# Object Detection Loss, many are wrappers of Segmentation losses
class ODLoss(Loss):
    def __init__(self, upper_bound: int, **kwargs):
        """Initialize the Object Detection Loss.

        Parameters
        ----------
        - upper_bound (int): The upper bound of the loss.

        Returns
        -------
        - None

        """
        super().__init__()
        self.upper_bound = upper_bound

    def __call__(
        self,
        true_boxes: torch.Tensor,
        true_cls: torch.Tensor,
        conf_boxes: torch.Tensor,
        conf_cls: torch.Tensor,
    ) -> torch.Tensor:
        """Call the Object Detection Loss.

        Returns
        -------
        - None

        """
        raise NotImplementedError("ODLoss is an abstract class.")


# LAC STYLE
class ConfidenceLoss(ODLoss):
    def __init__(
        self,
        upper_bound: int = 1,
        other_losses: Optional[List[Loss]] = None,
        **kwargs,
    ):
        """Initialize the Confidence Loss.

        Parameters
        ----------
        - upper_bound (int): The upper bound of the loss.

        Returns
        -------
        - None

        """
        super().__init__()
        self.upper_bound = upper_bound

    def __call__(
        self,
        true_boxes: torch.Tensor,
        true_cls: torch.Tensor,
        conf_boxes: torch.Tensor,
        conf_cls: torch.Tensor,
    ) -> torch.Tensor:
        """Call the Confidence Loss.

        Parameters
        ----------
        - predictions (ODPredictions): The predictions.
        - conformalized_predictions (ODConformalizedPredictions): The conformalized predictions.

        Returns
        -------
        - torch.Tensor: The loss value.

        """
        return max(
            [
                (
                    torch.zeros(1).cuda()
                    if len(conf_boxes) >= len(true_boxes)
                    else torch.ones(1).cuda()
                ),
            ]
            + [
                loss(true_boxes, true_cls, conf_boxes, conf_cls)
                for loss in self.other_losses
            ],
        )


from cods.classif.loss import ClassificationLoss


class LACLoss(ClassificationLoss):
    def __init__(self):
        super().__init__()
        self.upper_bound = 1

    def __call__(
        self,
        conf_cls,
        true_cls,
    ) -> torch.Tensor:
        """ """
        return torch.logical_not(torch.isin(true_cls, conf_cls)).float()


# IMAGE WISE VS BOX WISE GUARANTEE
# wrapping classification loss, by converting the predictions from the od to the classification format
class ClassificationLossWrapper(Loss):
    def __init__(self, classification_loss, **kwargs):
        """Initialize the Classification Loss Wrapper.

        Parameters
        ----------
        - classification_loss (Loss): The classification loss.

        Returns
        -------
        - None

        """
        super().__init__()
        self.classification_loss = classification_loss

    def __call__(
        self,
        true_boxes: torch.Tensor,
        true_cls: torch.Tensor,
        conf_boxes: torch.Tensor,
        conf_cls: torch.Tensor,
    ) -> torch.Tensor:
        """ """
        losses = []
        for i in range(len(conf_cls)):
            losses.append(self.classification_loss(conf_cls[i], true_cls[i]))
        return torch.mean(torch.stack(losses))


# # MaximumLoss : maximum of a list of losses with a list of parameters


# # maximum of risk =!= of losses
# class MaximumLoss(Loss):
#     def __init__(self, *losses):
#         """Initialize the Maximum Loss.

#         Parameters
#         ----------
#         - losses (list): The list of losses.

#         Returns
#         -------
#         - None

#         """
#         super().__init__()
#         self.losses = losses

#     def __call__(
#         self,
#         predictions: ODPredictions,
#         conformalized_predictions: ODConformalizedPredictions,
#     ) -> torch.Tensor:
#         """Call the Maximum Loss.

#         Returns
#         -------
#         - torch.Tensor: The loss value.

#         """
#         conf_boxes = conformalized_predictions.conf_boxes
#         true_boxes = predictions.true_boxes
#         return max([loss(conf_boxes, true_boxes) for loss in self.losses])


# TODO: formulate in classical conformal sense!
class HausdorffSignedDistanceLoss(ODLoss):
    def __init__(self, beta: float = 0.25):
        """Initialize the Hausdorff Signed Distance Loss.

        Parameters
        ----------
        - beta (float): The beta value.

        Returns
        -------
        - None

        """
        self.upper_bound = 1
        self.beta = beta

    def __call__(
        self,
        true_boxes: torch.Tensor,
        true_cls: torch.Tensor,
        conf_boxes: torch.Tensor,
        conf_cls: torch.Tensor,
    ) -> float:
        """Call the Hausdorff Signed Distance Loss.

        Parameters
        ----------
        - conf_boxes (torch.Tensor): The conformal boxes.
        - true_boxes (torch.Tensor): The true boxes.

        Returns
        -------
        - float: The loss value.

        """
        if len(true_boxes) == 0:
            return 0
        if len(conf_boxes) == 0:
            return 1
        areas = get_covered_areas_of_gt_union(conf_boxes, true_boxes)
        is_not_covered = (
            torch.FloatTensor(areas) < 0.999
        )  # because doubt on the computation of the overlap, check formula TODO
        miscoverage = torch.mean(is_not_covered)
        loss = 1 if miscoverage > self.beta else 0
        return loss


class ClassBoxWiseRecallLoss(ODLoss):
    def __init__(self, union_of_boxes: bool = True):
        """Initialize the Box-wise Recall Loss.

        Parameters
        ----------
        - union_of_boxes (bool): Whether to use the union of boxes.

        Returns
        -------
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
        self,
        true_boxes: torch.Tensor,
        true_cls,
        conf_boxes: torch.Tensor,
        conf_cls,
    ) -> torch.Tensor:
        """Call the Box-wise Recall Loss.

        Parameters
        ----------
        - conf_boxes (torch.Tensor): The conformal boxes.
        - true_boxes (torch.Tensor): The true boxes.

        Returns
        -------
        - float: The loss value.

        """
        if len(true_boxes) == 0:
            return torch.zeros(1).cuda()
        if len(conf_boxes) == 0:
            return torch.ones(1).cuda()
        areas = self.get_covered_areas(conf_boxes, true_boxes)
        is_not_covered_loc = (
            areas < 0.999
        )  # because doubt on the computation of the overlap, check formula TODO
        is_not_covered_cls = torch.tensor(
            [tc not in cc for (tc, cc) in zip(true_cls, conf_cls)],
            dtype=torch.float,
        ).cuda()
        is_not_covered = torch.logical_or(
            is_not_covered_loc,
            is_not_covered_cls,
        ).float()
        miscoverage = torch.zeros(1).cuda() + torch.mean(
            is_not_covered,
        )  # TODO: bugfix
        return miscoverage


class BoxWiseRecallLoss(ODLoss):
    """Box-wise recall loss: 1 - mean(areas of the union of the boxes),

    This loss function calculates the recall loss based on the areas of the union of the predicted and true bounding boxes.
    The recall loss is defined as 1 minus the mean of the areas of the union of the boxes.
    """

    def __init__(self, union_of_boxes: bool = True):
        """Initialize the Box-wise Recall Loss.

        Parameters
        ----------
        - union_of_boxes (bool): Whether to use the union of boxes.

        Returns
        -------
        - None

        """
        self.upper_bound = 1
        self.union_of_boxes = union_of_boxes
        self.get_covered_areas = get_covered_areas_of_gt_union
        if not union_of_boxes:
            raise NotImplementedError(
                "Box-wise Recall Loss only supports union of boxes.",
            )

    def __call__(
        self,
        true_boxes: torch.Tensor,
        true_cls: torch.Tensor,
        conf_boxes: torch.Tensor,
        conf_cls: torch.Tensor,
    ) -> torch.Tensor:
        """Call the Box-wise Recall Loss.

        Parameters
        ----------
        - predictions (ODPredictions): The predictions.
        - conformalized_predictions (ODConformalizedPredictions): The conformalized predictions.

        Returns
        -------
        - float: The loss value.

        """
        if len(true_boxes) == 0:
            return torch.zeros(1).cuda()
        if len(conf_boxes) == 0:
            return torch.ones(1).cuda()
        areas = self.get_covered_areas(conf_boxes, true_boxes)
        is_not_covered = (
            areas < 0.999
        ).float()  # because doubt on the computation of the overlap, check formula TODO
        miscoverage = torch.zeros(1).cuda() + torch.mean(
            is_not_covered,
        )  # TODO: tmp for bugfix
        return miscoverage


class PixelWiseRecallLoss(ODLoss):
    def __init__(self, union_of_boxes: bool = True):
        """Initialize the Pixel-wise Recall Loss.

        Parameters
        ----------
        - union_of_boxes (bool): Whether to use the union of boxes.

        Returns
        -------
        - None

        """
        self.upper_bound = 1
        self.union_of_boxes = union_of_boxes
        self.get_covered_areas = get_covered_areas_of_gt_union
        if not union_of_boxes:
            raise NotImplementedError(
                "Pixel-wise Recall Loss only supports union of boxes.",
            )

    def __call__(
        self,
        true_boxes: torch.Tensor,
        conf_boxes: torch.Tensor,
        conf_cls: torch.Tensor,
    ) -> torch.Tensor:
        """Call the Pixel-wise Recall Loss.

        Parameters
        ----------
        - conf_boxes (torch.Tensor): The conformal boxes.
        - true_boxes (torch.Tensor): The true boxes.

        Returns
        -------
        - torch.Tensor: The loss value.

        """
        if len(true_boxes) == 0:
            return torch.zeros(1).cuda()
        if len(conf_boxes) == 0:
            return torch.ones(1).cuda()
        areas = self.get_covered_areas(conf_boxes, true_boxes)
        loss = torch.ones(1).cuda() - torch.mean(areas)
        return loss


class NumberPredictionsGapLoss(ODLoss):
    def __init__(self):
        """Initialize the Number Predictions Gap Loss.

        Returns
        -------
        - None

        """
        self.upper_bound = 1

    def __call__(
        self,
        conf_boxes: torch.Tensor,
        true_boxes: torch.Tensor,
    ) -> float:
        """Call the Number Predictions Gap Loss.

        Parameters
        ----------
        - conf_boxes (torch.Tensor): The conformal boxes.
        - true_boxes (torch.Tensor): The true boxes.

        Returns
        -------
        - float: The loss value.

        """
        raise NotImplementedError(
            "NumberPredictionsGapLoss is not implemented yet",
        )
        loss = (len(true_boxes) - len(conf_boxes)) / max(len(true_boxes), 1)
        return min(loss, 1)
