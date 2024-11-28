from logging import getLogger
from typing import List, Optional

import torch

from cods.base.loss import Loss
from cods.classif.loss import ClassificationLoss
from cods.od.utils import (
    # get_covered_areas_of_gt_max,
    get_covered_areas_of_gt_union,
)

logger = getLogger("cods")


# Object Detection Loss, many are wrappers of Segmentation losses
class ODLoss(Loss):
    def __init__(self, upper_bound: int, device: str = "cpu", **kwargs):
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
        self.device = device

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
class BoxCountThresholdConfidenceLoss(ODLoss):
    def __init__(
        self,
        upper_bound: int = 1,
        other_losses: Optional[List[Loss]] = None,
        device: str = "cpu",
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
        super().__init__(upper_bound=upper_bound, device=device)
        self.other_losses = other_losses if other_losses is not None else []

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
        #print(len(conf_boxes), len(true_boxes))
        return max(
            [
                (
                    torch.zeros(1).to(self.device)
                    if len(conf_boxes) >= len(true_boxes)
                    else torch.ones(1).to(self.device)
                ),
            ]
            + [
                loss(true_boxes, true_cls, conf_boxes, conf_cls)
                for loss in self.other_losses
            ],
        )


class BoxCountRecallConfidenceLoss(ODLoss):
    def __init__(
        self,
        upper_bound: int = 1,
        distance_threshold: float = 100,
        other_losses: Optional[List[Loss]] = None,
        device: str = "cpu",
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
        super().__init__(upper_bound=upper_bound, device=device)
        self.distance_threshold = distance_threshold
        self.other_losses = other_losses if other_losses is not None else []

    def __call__(
        self,
        true_boxes: torch.Tensor | List[torch.Tensor],
        true_cls: torch.Tensor | List[torch.Tensor],
        conf_boxes: torch.Tensor | List[torch.Tensor],
        conf_cls: List[torch.Tensor],
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
        if len(true_boxes) == 0:
            loss = torch.zeros(1).to(self.device)
        else:
            loss = torch.maximum(
                torch.zeros(1).to(self.device),
                torch.tensor(
                    (len(true_boxes) - len(conf_boxes)) / len(true_boxes)
                ),
            ).to(self.device)
        return max(
            [
                (loss),
            ]
            + [
                loss(true_boxes, true_cls, conf_boxes, conf_cls)
                for loss in self.other_losses
            ],
        )
        # losses = []
        # if len(true_boxes) == 0:
        #     loss = torch.zeros(1).to(device)
        # elif len(conf_boxes) == 0:
        #     loss = torch.ones(1).to(device)
        # else:
        #     for i, true_boxes_i in enumerate(true_boxes):
        #         # search for closest box
        #         distances = []
        #         for conf_boxes_i in conf_boxes[i]:
        #             # TODO doesn"r work because of expansion fo boxes for localization
        #             center_true_x = (true_boxes_i[0] + true_boxes_i[2]) / 2
        #             center_true_y = (true_boxes_i[1] + true_boxes_i[3]) / 2
        #             center_conf_x = (conf_boxes_i[0] + conf_boxes_i[2]) / 2
        #             center_conf_y = (conf_boxes_i[1] + conf_boxes_i[3]) / 2
        #             distance = torch.sqrt(
        #                 (center_true_x - center_conf_x) ** 2
        #                 + (center_true_y - center_conf_y) ** 2
        #             )
        #             # TODO: improve distance
        #             distances.append(distance)
        #         # print(f"distances: {distances}")
        #         # TODO arbirtrary

        #         if len(distances) == 0:
        #             loss_i = torch.ones(1).to(device)
        #             losses.append(loss_i)
        #             continue

        #         loss_i = (
        #             torch.zeros(1).to(device)
        #             if torch.min(torch.stack(distances))
        #             < self.distance_threshold
        #             else torch.ones(1).to(device)
        #         )
        #         losses.append(loss_i)
        #     loss = torch.mean(torch.stack(losses))
        # return max(
        #     [loss]
        #     + [
        #         loss(true_boxes, true_cls, conf_boxes, conf_cls)
        #         for loss in self.other_losses
        #     ],
        # )


class ODBinaryClassificationLoss(ClassificationLoss):
    def __init__(self):
        super().__init__(upper_bound=1)

    def __call__(
        self,
        conf_cls,
        true_cls,
    ) -> torch.Tensor:
        """ """
        # if len(conf_cls) == 0:
        #    logger.warning(f"conf_cls is empty : {conf_cls}")
        loss = (
            torch.logical_not(torch.isin(true_cls, conf_cls)).float().expand(1)
        )
        # if loss == 0:
        #    logger.info(f"true_cls: {true_cls}, conf_cls: {conf_cls}")
        return loss


# IMAGE WISE VS BOX WISE GUARANTEE
# wrapping classification loss, by converting the predictions from the od to the classification format
class ClassificationLossWrapper(ODLoss):
    def __init__(self, classification_loss, device: str = "cpu", **kwargs):
        """Initialize the Classification Loss Wrapper.

        Parameters
        ----------
        - classification_loss (Loss): The classification loss.

        Returns
        -------
        - None

        """
        self.classification_loss = classification_loss
        super().__init__(
            upper_bound=classification_loss.upper_bound, device=device
        )

    def __call__(
        self,
        true_boxes: torch.Tensor,
        true_cls: torch.Tensor,
        conf_boxes: torch.Tensor,
        conf_cls: torch.Tensor,
    ) -> torch.Tensor:
        """ """
        losses = []
        if len(true_cls) == 0:
            logger.warning(f"true_cls is empty : {true_cls}")
            return torch.zeros(1).to(self.device)
        if len(conf_cls) == 0:
            logger.warning(f"conf_cls is empty : {conf_cls}")
            return torch.ones(1).to(self.device)
        for i in range(len(conf_cls)):
            loss = self.classification_loss(conf_cls[i], true_cls[i])
            # print(f"loss: {loss}")
            losses.append(loss)
        return torch.zeros(1).to(self.device) + torch.mean(torch.stack(losses))


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
class ThresholdedRecallLoss(ODLoss):
    def __init__(
        self,
        beta: float = 0.25,
        device: str = "cpu",
    ):
        """Initialize the Hausdorff Signed Distance Loss.

        Parameters
        ----------
        - beta (float): The beta value.

        Returns
        -------
        - None

        """
        super().__init__(upper_bound=1, device=device)
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
            return torch.zeros(1).to(self.device)
        if len(conf_boxes) == 0:
            return torch.ones(1).to(self.device)
        areas = get_covered_areas_of_gt_union(conf_boxes, true_boxes)
        is_not_covered = (
            areas < 0.999
        ).float()  # because doubt on the computation of the overlap, check formula TODO
        miscoverage = torch.mean(is_not_covered)
        loss = (
            torch.ones(1).to(self.device)
            if miscoverage > self.beta
            else torch.zeros(1).to(self.device)
        )
        return loss


class ClassBoxWiseRecallLoss(ODLoss):
    def __init__(
        self,
        union_of_boxes: bool = True,
        device: str = "cpu",
    ):
        """Initialize the Box-wise Recall Loss.

        Parameters
        ----------
        - union_of_boxes (bool): Whether to use the union of boxes.

        Returns
        -------
        - None

        """
        super().__init__(upper_bound=1, device=device)
        self.union_of_boxes = union_of_boxes
        self.get_covered_areas = (
            get_covered_areas_of_gt_union
            # if union_of_boxes
            # else get_covered_areas_of_gt_max
        )
        if not union_of_boxes:
            raise NotImplementedError(
                "Box-wise Recall Loss only supports union of boxes.",
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
            return torch.zeros(1).to(self.device)
        if len(conf_boxes) == 0:
            return torch.ones(1).to(self.device)
        areas = self.get_covered_areas(conf_boxes, true_boxes)
        is_not_covered_loc = (
            areas < 0.999
        )  # because doubt on the computation of the overlap, check formula TODO
        is_not_covered_cls = torch.tensor(
            [tc not in cc for (tc, cc) in zip(true_cls, conf_cls)],
            dtype=torch.float,
        ).to(self.device)
        is_not_covered = torch.logical_or(
            is_not_covered_loc,
            is_not_covered_cls,
        ).float()
        miscoverage = torch.zeros(1).to(self.device) + torch.mean(
            is_not_covered,
        )  # TODO: bugfix
        return miscoverage


class BoxWiseRecallLoss(ODLoss):
    """Box-wise recall loss: 1 - mean(areas of the union of the boxes),

    This loss function calculates the recall loss based on the areas of the union of the predicted and true bounding boxes.
    The recall loss is defined as 1 minus the mean of the areas of the union of the boxes.
    """

    def __init__(
        self,
        union_of_boxes: bool = True,
        device: str = "cpu",
    ):
        """Initialize the Box-wise Recall Loss.

        Parameters
        ----------
        - union_of_boxes (bool): Whether to use the union of boxes.

        Returns
        -------
        - None

        """
        super().__init__(upper_bound=1, device=device)
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
            return torch.zeros(1).to(self.device)
        if len(conf_boxes) == 0:
            return torch.ones(1).to(self.device)
        areas = self.get_covered_areas(conf_boxes, true_boxes)
        is_not_covered = (
            areas < 0.999
        ).float()  # because doubt on the computation of the overlap, check formula TODO
        miscoverage = torch.zeros(1).to(self.device) + torch.mean(
            is_not_covered,
        )  # TODO: tmp for bugfix
        return miscoverage


class PixelWiseRecallLoss(ODLoss):
    def __init__(
        self,
        union_of_boxes: bool = True,
        device: str = "cpu",
    ):
        """Initialize the Pixel-wise Recall Loss.

        Parameters
        ----------
        - union_of_boxes (bool): Whether to use the union of boxes.

        Returns
        -------
        - None

        """
        super().__init__(upper_bound=1, device=device)
        self.union_of_boxes = union_of_boxes
        self.get_covered_areas = get_covered_areas_of_gt_union
        if not union_of_boxes:
            raise NotImplementedError(
                "Pixel-wise Recall Loss only supports union of boxes.",
            )

    def __call__(
        self,
        true_boxes: torch.Tensor,
        true_cls: torch.Tensor,
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
            return torch.zeros(1).to(self.device)
        if len(conf_boxes) == 0:
            return torch.ones(1).to(self.device)
        areas = self.get_covered_areas(conf_boxes, true_boxes)
        loss = torch.ones(1).to(self.device) - torch.mean(areas)
        return loss


class NumberPredictionsGapLoss(ODLoss):
    def __init__(
        self,
        device: str = "cpu",
    ):
        """Initialize the Number Predictions Gap Loss.

        Returns
        -------
        - None

        """
        super().__init__(upper_bound=1, device=device)

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
