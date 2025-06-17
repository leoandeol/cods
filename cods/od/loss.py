from __future__ import annotations

from logging import getLogger

import torch

from cods.base.loss import Loss
from cods.classif.loss import ClassificationLoss
from cods.od.utils import (
    assymetric_hausdorff_distance,
    f_lac,
    fast_covered_areas_of_gt,
    get_covered_areas_of_gt_union,
    vectorized_generalized_iou,
)

logger = getLogger("cods")


# Object Detection Loss, many are wrappers of Segmentation losses
class ODLoss(Loss):
    """Base class for Object Detection losses."""

    def __init__(self, upper_bound: int, device: str = "cpu", **kwargs):
        """Initialize the Object Detection Loss.

        Args:
        ----
            upper_bound (int): The upper bound of the loss.
            device (str, optional): Device to use for tensors. Defaults to "cpu".
            **kwargs: Additional keyword arguments.

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
        """Compute the object detection loss.

        Args:
        ----
            true_boxes (torch.Tensor): Ground truth bounding boxes.
            true_cls (torch.Tensor): Ground truth class labels.
            conf_boxes (torch.Tensor): Conformalized/predicted bounding boxes.
            conf_cls (torch.Tensor): Conformalized/predicted class labels.

        Returns:
        -------
            torch.Tensor: The loss value.

        """
        raise NotImplementedError("ODLoss is an abstract class.")


# LAC STYLE
class BoxCountThresholdConfidenceLoss(ODLoss):
    """Confidence loss based on whether the count of conformalized boxes meets or exceeds the count of true boxes.

    The loss is 0 if `len(conf_boxes) >= len(true_boxes)`, and 1 otherwise.
    """

    def __init__(
        self,
        upper_bound: int = 1,
        device: str = "cpu",
        **kwargs,
    ):
        """Initialize the BoxCountThresholdConfidenceLoss.

        Args:
        ----
            upper_bound (int, optional): The upper bound of the loss. Defaults to 1.
            device (str, optional): Device to use for tensors. Defaults to "cpu".
            **kwargs: Additional keyword arguments.

        """
        super().__init__(upper_bound=upper_bound, device=device)

    def __call__(
        self,
        true_boxes: torch.Tensor,
        true_cls: torch.Tensor,
        conf_boxes: torch.Tensor,
        conf_cls: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the confidence loss based on box count threshold.

        Args:
        ----
            true_boxes (torch.Tensor): Ground truth bounding boxes for a single image.
            true_cls (torch.Tensor): Ground truth class labels for a single image.
            conf_boxes (torch.Tensor): Conformalized/predicted bounding boxes for a single image.
            conf_cls (torch.Tensor): Conformalized/predicted class labels for a single image.

        Returns:
        -------
            torch.Tensor: The loss value.

        """
        return (
            torch.zeros(1).to(self.device)
            if len(conf_boxes) >= len(true_boxes)
            else torch.ones(1).to(self.device)
        )


class BoxCountTwosidedConfidenceLoss(ODLoss):
    """Confidence loss based on whether the absolute difference between true and predicted box counts exceeds a threshold.

    The loss is 1 if `abs(len(true_boxes) - len(conf_boxes)) > self.threshold`, and 0 otherwise.
    If there are no true boxes, the loss is 0.
    """

    def __init__(
        self,
        upper_bound: int = 1,
        threshold: int = 3,
        device: str = "cpu",
        **kwargs,
    ):
        """Initialize the BoxCountTwosidedConfidenceLoss.

        Args:
        ----
            upper_bound (int, optional): The upper bound of the loss. Defaults to 1.
            threshold (int, optional): Allowed difference in box counts. Defaults to 3.
            device (str, optional): Device to use for tensors. Defaults to "cpu".
            **kwargs: Additional keyword arguments.

        """
        super().__init__(upper_bound=upper_bound, device=device)
        self.threshold = threshold

    def __call__(
        self,
        true_boxes: torch.Tensor | list[torch.Tensor],
        true_cls: torch.Tensor | list[torch.Tensor],
        conf_boxes: torch.Tensor | list[torch.Tensor],
        conf_cls: list[torch.Tensor],
    ) -> torch.Tensor:
        """Compute the confidence loss based on two-sided box count threshold.

        Args:
        ----
            true_boxes (torch.Tensor | List[torch.Tensor]): Ground truth bounding boxes for a single image.
            true_cls (torch.Tensor | List[torch.Tensor]): Ground truth class labels for a single image.
            conf_boxes (torch.Tensor | List[torch.Tensor]): Conformalized/predicted bounding boxes for a single image.
            conf_cls (List[torch.Tensor]): Conformalized/predicted class labels for a single image.

        Returns:
        -------
            torch.Tensor: The loss value.

        """
        if len(true_boxes) == 0:
            loss = torch.zeros(1).to(self.device)
        else:
            loss = (
                torch.ones(1).to(self.device)
                if abs(len(true_boxes) - len(conf_boxes)) > self.threshold
                else torch.zeros(1).to(self.device)
            )
            loss = torch.tensor(loss).float().to(self.device).expand(1)
        return loss


class BoxCountRecallConfidenceLoss(ODLoss):
    """Confidence loss based on the recall of box counts.

    Calculates `max(0, (len(true_boxes) - len(conf_boxes)) / len(true_boxes))`.
    If there are no true boxes, the loss is 0.
    """

    def __init__(
        self,
        upper_bound: int = 1,
        device: str = "cpu",
        **kwargs,
    ):
        """Initialize the BoxCountRecallConfidenceLoss.

        Args:
        ----
            upper_bound (int, optional): The upper bound of the loss. Defaults to 1.
            device (str, optional): Device to use for tensors. Defaults to "cpu".
            **kwargs: Additional keyword arguments.

        """
        super().__init__(upper_bound=upper_bound, device=device)

    def __call__(
        self,
        true_boxes: torch.Tensor | list[torch.Tensor],
        true_cls: torch.Tensor | list[torch.Tensor],
        conf_boxes: torch.Tensor | list[torch.Tensor],
        conf_cls: list[torch.Tensor],
    ) -> torch.Tensor:
        """Compute the confidence loss based on recall of box counts.

        Args:
        ----
            true_boxes (torch.Tensor | List[torch.Tensor]): Ground truth bounding boxes for a single image.
            true_cls (torch.Tensor | List[torch.Tensor]): Ground truth class labels for a single image.
            conf_boxes (torch.Tensor | List[torch.Tensor]): Conformalized/predicted bounding boxes for a single image.
            conf_cls (List[torch.Tensor]): Conformalized/predicted class labels for a single image.

        Returns:
        -------
            torch.Tensor: The loss value.

        """
        if len(true_boxes) == 0:
            loss = torch.zeros(1).to(self.device)
        else:
            loss = torch.maximum(
                torch.zeros(1).to(self.device),
                torch.tensor(
                    (len(true_boxes) - len(conf_boxes)) / len(true_boxes),
                ),
            ).to(self.device)
        return loss


class ThresholdedBoxDistanceConfidenceLoss(ODLoss):
    """Confidence loss based on a thresholded distance between true and predicted boxes.

    This loss computes a combined distance (Hausdorff and LAC) between true and predicted boxes.
    The loss is the mean of indicators where this distance exceeds `self.distance_threshold`.
    """

    def __init__(
        self,
        upper_bound: int = 1,
        distance_threshold: float = 0.5,
        device: str = "cpu",
        **kwargs,
    ):
        """Initialize the ThresholdedBoxDistanceConfidenceLoss.

        Args:
        ----
            upper_bound (int, optional): The upper bound of the loss. Defaults to 1.
            distance_threshold (float, optional): Distance threshold for loss. Defaults to 0.5.
            device (str, optional): Device to use for tensors. Defaults to "cpu".
            **kwargs: Additional keyword arguments.

        """
        super().__init__(upper_bound=upper_bound, device=device)
        self.distance_threshold = distance_threshold
        # self.other_losses = other_losses if other_losses is not None else []

        # TODO redefine ConfidenceLoss abstract class, we need preds here not conformalized preds

    def __call__(
        self,
        true_boxes: torch.Tensor | list[torch.Tensor],
        true_cls: torch.Tensor | list[torch.Tensor],
        pred_boxes: torch.Tensor | list[torch.Tensor],
        pred_cls: list[torch.Tensor],
    ) -> torch.Tensor:
        """Compute the confidence loss based on thresholded box distance.

        Args:
        ----
            true_boxes (torch.Tensor | List[torch.Tensor]): Ground truth bounding boxes for a single image.
            true_cls (torch.Tensor | List[torch.Tensor]): Ground truth class labels for a single image.
            pred_boxes (torch.Tensor | List[torch.Tensor]): Predicted bounding boxes for a single image.
            pred_cls (List[torch.Tensor]): Predicted class labels for a single image.

        Returns:
        -------
            torch.Tensor: The loss value.

        """
        if len(true_boxes) == 0:
            loss = torch.zeros(1).to(self.device)
        elif len(pred_boxes) == 0:
            loss = torch.ones(1).to(self.device)
        else:
            class_factor = 0.22
            l_ass = assymetric_hausdorff_distance(true_boxes, pred_boxes)
            l_ass /= torch.max(l_ass)
            l_lac = f_lac(true_cls, pred_cls)
            distance_matrix = class_factor * l_lac + (1 - class_factor) * l_ass
            shortest_distances, _ = torch.min(distance_matrix, dim=1)
            # print("Distances", shortest_distances)
            loss = torch.mean(
                (shortest_distances > self.distance_threshold).float(),
            ).expand(1)
            # print("Final Loss", loss)
        return loss


class ODBinaryClassificationLoss(ClassificationLoss):
    """Binary classification loss for object detection.

    This loss is 1 if the true class is not in the conformalized class set, and 0 otherwise.
    """

    def __init__(self):
        """Initialize the ODBinaryClassificationLoss."""
        super().__init__(upper_bound=1)

    def __call__(
        self,
        conf_cls: torch.Tensor,
        true_cls: torch.Tensor,
    ) -> torch.Tensor:
        """Calculate the binary classification loss.

        Args:
        ----
            conf_cls (torch.Tensor): The conformalized set of class predictions for a single object.
            true_cls (torch.Tensor): The ground truth class label for a single object (scalar tensor).

        Returns:
        -------
            torch.Tensor: The loss value (0 or 1), expanded to a 1-element tensor.

        """
        loss = (
            torch.logical_not(torch.isin(true_cls, conf_cls)).float().expand(1)
        )
        # if loss == 0:
        #    logger.info(f"true_cls: {true_cls}, conf_cls: {conf_cls}")
        return loss


class ClassificationLossWrapper(ODLoss):
    """Wraps a standard classification loss for use in object detection.

    This class applies a given classification loss to each true object and its
    corresponding conformalized class predictions, then averages the losses.
    """

    def __init__(self, classification_loss, device: str = "cpu", **kwargs):
        """Initialize the ClassificationLossWrapper.

        Args:
        ----
            classification_loss (Loss): The classification loss to wrap.
            device (str, optional): Device to use for tensors. Defaults to "cpu".
            **kwargs: Additional keyword arguments.

        """
        self.classification_loss = classification_loss
        super().__init__(
            upper_bound=classification_loss.upper_bound,
            device=device,
        )

    def __call__(
        self,
        true_boxes: torch.Tensor,
        true_cls: torch.Tensor,
        conf_boxes: torch.Tensor,
        conf_cls: torch.Tensor,
        verbose: bool = False,
    ) -> torch.Tensor:
        """Calculate the wrapped classification loss for an image.

        Args:
        ----
            true_boxes (torch.Tensor): Ground truth bounding boxes for the image (not directly used by this loss).
            true_cls (torch.Tensor): Ground truth class labels for each object in the image.
            conf_boxes (torch.Tensor): Conformalized bounding boxes for the image (not directly used by this loss).
            conf_cls (torch.Tensor): List or Tensor of conformalized class prediction sets, one for each object.
            verbose (bool, optional): If True, log warnings for empty inputs. Defaults to False.

        Returns:
        -------
            torch.Tensor: The mean classification loss over all objects in the image, expanded to a 1-element tensor.

        """
        losses = []
        if len(true_cls) == 0:
            if verbose:
                logger.warning(f"true_cls is empty : {true_cls}")
            return torch.zeros(1).to(self.device)
        if len(conf_cls) == 0:
            if verbose:
                logger.warning(f"conf_cls is empty : {conf_cls}")
            return torch.ones(1).to(self.device)
        for i in range(len(conf_cls)):
            loss = self.classification_loss(conf_cls[i], true_cls[i])
            # print(f"loss: {loss}")
            losses.append(loss)
        return torch.mean(torch.stack(losses)).expand(1)


class ThresholdedRecallLoss(ODLoss):
    """A recall loss that is 1 if the miscoverage (1 - recall) exceeds a threshold `beta`, and 0 otherwise.

    Miscoverage is calculated based on the proportion of true boxes not sufficiently covered
    by the union of conformalized boxes.
    """

    def __init__(
        self,
        beta: float = 0.25,
        device: str = "cpu",
    ):
        """Initialize the ThresholdedRecallLoss.

        Args:
        ----
            beta (float, optional): The beta value. Defaults to 0.25.
            device (str, optional): Device to use for tensors. Defaults to "cpu".

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
        """Compute the thresholded recall loss.

        Args:
        ----
            true_boxes (torch.Tensor): Ground truth bounding boxes for a single image.
            true_cls (torch.Tensor): Ground truth class labels for a single image (not used).
            conf_boxes (torch.Tensor): Conformalized bounding boxes for a single image.
            conf_cls (torch.Tensor): Conformalized class labels for a single image (not used).

        Returns:
        -------
            float: The loss value.

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
    """A combined recall loss for both localization (box-wise recall) and classification.

    The loss is the mean of indicators where either the true box is not sufficiently covered
    by the union of conformalized boxes, OR the true class is not in the conformalized class set.
    """

    def __init__(
        self,
        union_of_boxes: bool = True,
        device: str = "cpu",
    ):
        """Initialize the ClassBoxWiseRecallLoss.

        Args:
        ----
            union_of_boxes (bool, optional): Whether to use the union of boxes. Defaults to True.
            device (str, optional): Device to use for tensors. Defaults to "cpu".

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
        true_cls: torch.Tensor,
        conf_boxes: torch.Tensor,
        conf_cls: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the class and box-wise recall loss.

        Args:
        ----
            true_boxes (torch.Tensor): Ground truth bounding boxes for a single image.
            true_cls (torch.Tensor): Ground truth class labels for a single image.
            conf_boxes (torch.Tensor): Conformalized bounding boxes for a single image.
            conf_cls (torch.Tensor): Conformalized class labels for a single image.

        Returns:
        -------
            torch.Tensor: The loss value.

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
    """Box-wise recall loss: 1 - mean(areas of the union of the boxes).

    This loss function calculates the recall loss based on the areas of the union of the predicted and true bounding boxes.
    The recall loss is defined as 1 minus the mean of the areas of the union of the boxes.
    """

    def __init__(
        self,
        union_of_boxes: bool = True,
        device: str = "cpu",
    ):
        """Initialize the BoxWiseRecallLoss.

        Args:
        ----
            union_of_boxes (bool, optional): Whether to use the union of boxes. Defaults to True.
            device (str, optional): Device to use for tensors. Defaults to "cpu".

        """
        super().__init__(upper_bound=1, device=device)
        self.union_of_boxes = union_of_boxes
        self.get_covered_areas = (
            fast_covered_areas_of_gt  # get_covered_areas_of_gt_union
        )
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
        """Compute the box-wise recall loss.

        Args:
        ----
            true_boxes (torch.Tensor): Ground truth bounding boxes for a single image.
            true_cls (torch.Tensor): Ground truth class labels for a single image (not used).
            conf_boxes (torch.Tensor): Conformalized bounding boxes for a single image.
            conf_cls (torch.Tensor): Conformalized class labels for a single image (not used).

        Returns:
        -------
            torch.Tensor: The loss value.

        """
        if len(true_boxes) == 0:
            return torch.zeros(1).to(self.device)
        if len(conf_boxes) == 0:
            return torch.ones(1).to(self.device)
        # try:
        areas = self.get_covered_areas(conf_boxes, true_boxes)
        # except Exception as e:
        #     # print shapes
        #     print(conf_boxes.shape, true_boxes.shape)
        #     print(f"Error in get_covered_areas: {e}")
        #     raise e
        is_not_covered = (
            areas < 0.999
        ).float()  # because doubt on the computation of the overlap, check formula TODO
        miscoverage = torch.mean(
            is_not_covered,
        ).expand(1)  # TODO: tmp for bugfix
        return miscoverage


class PixelWiseRecallLoss(ODLoss):
    """Pixel-wise recall loss.

    Calculates `1 - mean(areas)`, where `areas` are the fractions of each true box
    covered by the corresponding (matched) conformalized box.
    """

    def __init__(
        self,
        union_of_boxes: bool = True,
        device: str = "cpu",
    ):
        """Initialize the PixelWiseRecallLoss.

        Args:
        ----
            union_of_boxes (bool, optional): Whether to use the union of boxes. Defaults to True.
            device (str, optional): Device to use for tensors. Defaults to "cpu".

        """
        super().__init__(upper_bound=1, device=device)
        self.union_of_boxes = union_of_boxes
        self.get_covered_areas = (
            fast_covered_areas_of_gt  # get_covered_areas_of_gt_union
        )
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
        """Compute the pixel-wise recall loss.

        Args:
        ----
            true_boxes (torch.Tensor): Ground truth bounding boxes for a single image.
            true_cls (torch.Tensor): Ground truth class labels for a single image (not used).
            conf_boxes (torch.Tensor): Conformalized bounding boxes for a single image.
            conf_cls (torch.Tensor): Conformalized class labels for a single image (not used).

        Returns:
        -------
            torch.Tensor: The loss value.

        """
        if len(true_boxes) == 0:
            return torch.zeros(1).to(self.device)
        if len(conf_boxes) == 0:
            return torch.ones(1).to(self.device)
        areas = self.get_covered_areas(conf_boxes, true_boxes)
        loss = torch.ones(1).to(self.device) - torch.mean(areas)
        return loss


#### TESTS:
class BoxWisePrecisionLoss(ODLoss):
    """Box-wise precision loss.

    For each conformalized box, it finds the maximum overlap (area of contained part of true box)
    with any true box. The loss is the mean of indicators where this maximum overlap is insufficient (e.g., < 0.999).
    """

    def __init__(
        self,
        union_of_boxes: bool = True,
        device: str = "cpu",
    ):
        """Initialize the BoxWisePrecisionLoss.

        Args:
        ----
            union_of_boxes (bool, optional): Whether to use the union of boxes. Defaults to True.
            device (str, optional): Device to use for tensors. Defaults to "cpu".

        """
        super().__init__(upper_bound=1, device=device)
        self.union_of_boxes = union_of_boxes
        self.get_covered_areas = (
            fast_covered_areas_of_gt  # get_covered_areas_of_gt_union
        )
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
        """Compute the box-wise precision loss.

        Args:
        ----
            true_boxes (torch.Tensor): Ground truth bounding boxes for a single image.
            true_cls (torch.Tensor): Ground truth class labels for a single image (not used).
            conf_boxes (torch.Tensor): Conformalized bounding boxes for a single image.
            conf_cls (torch.Tensor): Conformalized class labels for a single image (not used).

        Returns:
        -------
            torch.Tensor: The loss value.

        """
        if len(true_boxes) == 0:
            return torch.zeros(1).to(self.device)
        if len(conf_boxes) == 0:
            return torch.ones(1).to(self.device)
        # try:
        losses = []
        for conf_box in conf_boxes:
            area = self.get_covered_areas(conf_box[None, :], true_boxes).max()
            loss = (area < 0.999).float()
            losses.append(loss)
        miscoverage = torch.mean(torch.stack(losses)).expand(1)
        # TODO: here, there's always as many preds as gt, because this is done after matching
        # TODO: should the matching be done... inside the loss ?
        return miscoverage


class BoxWiseIoULoss(ODLoss):
    """Box-wise IoU loss.

    Calculates the mean of indicators where the Generalized IoU (GIoU) between true boxes and
    conformalized boxes is less than a threshold (e.g., 0.9).
    """

    def __init__(
        self,
        union_of_boxes: bool = True,
        device: str = "cpu",
    ):
        """Initialize the BoxWiseIoULoss.

        Args:
        ----
            union_of_boxes (bool, optional): Whether to use the union of boxes. Defaults to True.
            device (str, optional): Device to use for tensors. Defaults to "cpu".

        """
        super().__init__(upper_bound=1, device=device)
        self.union_of_boxes = union_of_boxes
        self.get_covered_areas = (
            fast_covered_areas_of_gt  # get_covered_areas_of_gt_union
        )
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
        """Compute the box-wise IoU loss.

        Args:
        ----
            true_boxes (torch.Tensor): Ground truth bounding boxes for a single image.
            true_cls (torch.Tensor): Ground truth class labels for a single image (not used).
            conf_boxes (torch.Tensor): Conformalized bounding boxes for a single image.
            conf_cls (torch.Tensor): Conformalized class labels for a single image (not used).

        Returns:
        -------
            torch.Tensor: The loss value.

        """
        if len(true_boxes) == 0:
            return torch.zeros(1).to(self.device)
        if len(conf_boxes) == 0:
            return torch.ones(1).to(self.device)
        # try:
        ious = vectorized_generalized_iou(true_boxes, conf_boxes)
        miscoverage = (ious < 0.9).float().mean().expand(1)
        # except Exception as e:
        #     # print shapes
        #     print(conf_boxes.shape, true_boxes.shape)
        #     print(f"Error in get_covered_areas: {e}")
        #     raise e
        return miscoverage


class NumberPredictionsGapLoss(ODLoss):
    """Loss based on the normalized difference between the number of true boxes and conformalized boxes.

    Calculates `(len(true_boxes) - len(conf_boxes)) / max(len(true_boxes), 1)`, capped at 1.
    Note: This loss is currently not implemented.
    """

    def __init__(
        self,
        device: str = "cpu",
    ):
        """Initialize the NumberPredictionsGapLoss.

        Args:
        ----
            device (str, optional): Device to use for tensors. Defaults to "cpu".

        """
        super().__init__(upper_bound=1, device=device)

    def __call__(
        self,
        conf_boxes: torch.Tensor,
        true_boxes: torch.Tensor,
    ) -> float:
        """Compute the number predictions gap loss.

        Args:
        ----
            conf_boxes (torch.Tensor): Conformalized bounding boxes for a single image.
            true_boxes (torch.Tensor): Ground truth bounding boxes for a single image.

        Returns:
        -------
            float: The loss value.

        """
        raise NotImplementedError(
            "NumberPredictionsGapLoss is not implemented yet",
        )
        loss = (len(true_boxes) - len(conf_boxes)) / max(len(true_boxes), 1)
        return min(loss, 1)
