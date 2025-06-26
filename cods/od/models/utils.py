"""Utility functions and classes for object detection models.

This module provides utility functions and classes for object detection models,
including channel resizing, Bayesian object detection postprocessing, and
prediction filtering utilities.
"""

import torch
from torch import nn
from torchvision.ops import box_iou


class ResizeChannels(nn.Module):
    """Module to resize image channels.
    
    Converts single-channel images to 3-channel by repeating the channel,
    useful for ensuring model input compatibility.
    """

    def __init__(self, num_channels):
        """Initialize the ResizeChannels module.
        
        Args:
            num_channels (int): Target number of channels.

        """
        super().__init__()
        self.num_channels = num_channels

    # for if 1 channel, repeat 3 times, if 3 channels, don't change the image
    def forward(self, image):
        """Forward pass to resize image channels.
        
        Args:
            image (torch.Tensor): Input image tensor.
            
        Returns:
            torch.Tensor: Image with resized channels.

        """
        if image.shape[0] == 1:
            return image.repeat(3, 1, 1)
        return image


def bayesod(
    pred_boxes: torch.Tensor,
    confidences: torch.Tensor,
    pred_cls: torch.Tensor,
    iou_threshold: float,
):
    """_summary_.

    Args:
    ----
        pred_boxes (torch.Tensor): _description_
        confidences (torch.Tensor): _description_
        pred_cls (torch.Tensor): _description_
        iou_threshold (float): _description_

    """
    # TODO
    raise NotImplementedError("BayesOD is not implemented yet")

    # Sort the predictions by confidence in descending order
    args = torch.argsort(confidences, descending=True)
    pred_boxes = pred_boxes[args]
    pred_cls = pred_cls[args]
    confidences = confidences[args]

    ious = box_iou(pred_boxes, pred_boxes)

    ious_overlap = ious > iou_threshold

    already_used = []

    clusters = []

    # TODO @luca: check logic
    for i, row in enumerate(ious_overlap):
        if i in already_used:
            continue

        tmp_cluster = torch.where(row).cpu().numpy().tolist()

        # TODO(leo) @luca : there's absolutely a better way to do that
        cluster = [
            element for element in tmp_cluster if element not in already_used
        ]

        if len(cluster) > 0:
            clusters.append(cluster)
            already_used.expand(cluster)

    new_pred_boxes = []
    new_confidences = []
    new_pred_cls = []
    new_pred_boxes_unc = []

    # Bayesian Fusion : estimate mean and variance of the bounding boxes
    # First attempt: weighted mean and variance
    for cluster in clusters:
        # nx4
        curr_boxes = torch.stack([pred_boxes[i] for i in cluster])
        # weighted mean
        new_box = (curr_boxes * confidences[cluster].reshape(-1, 1)).sum(
            0,
        ) / confidences[cluster].sum()
        # variance (?)
        new_box_unc = (curr_boxes - new_box).pow(2).sum(0)
        new_pred_boxes.append(new_box)
        new_pred_boxes_unc.append(new_box_unc)

        curr_confidences = confidences[cluster]
        new_confidence = curr_confidences.max()
        new_confidences.append(new_confidence)

        # merge the softmax probabilities in a weighted way
        # such that it is guaranteed to still be probability distribution
        new_cls = torch.zeros_like(pred_cls[0])
        for c in cluster:
            new_cls += pred_cls[c] * confidences[c]
        new_cls /= confidences[cluster].sum()
        assert (new_cls.sum().item() - 1) < 1e-8
        new_pred_cls.append(new_cls)

    return (
        torch.stack(new_pred_boxes),
        torch.stack(new_confidences),
        torch.stack(new_pred_cls),
        torch.stack(new_pred_boxes_unc),
    )


# Filter the preds_cal and preds_val with confidence below 0.001


def filter_preds(preds, confidence_threshold=0.001):
    """Filter predictions based on confidence threshold.
    
    Args:
        preds: Predictions object containing boxes, confidences, and classes.
        confidence_threshold (float, optional): Minimum confidence threshold. Defaults to 0.001.
        
    Returns:
        Filtered predictions object with low-confidence predictions removed.

    """
    filters = [
        conf > confidence_threshold
        if (conf > confidence_threshold).any()
        else conf.argmin(0)[None]
        for conf in preds.confidences
    ]
    preds.pred_boxes = [pbs[f] for pbs, f in zip(preds.pred_boxes, filters)]
    preds.pred_cls = [pcs[f] for pcs, f in zip(preds.pred_cls, filters)]
    preds.confidences = [
        conf[f] for conf, f in zip(preds.confidences, filters)
    ]
    return preds
