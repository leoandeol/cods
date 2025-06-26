"""Utility functions for object detection tasks and conformal prediction.

This module provides various utility functions for object detection tasks,
including geometric computations, IoU calculations, optimization utilities,
and distance metrics used in conformal prediction workflows.
"""

from logging import getLogger
from typing import List

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

logger = getLogger("cods")


def mesh_func(
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    pbs: torch.Tensor,
) -> torch.Tensor:
    """Compute mesh function.

    Args:
    ----
        x1 (int): x-coordinate of the top-left corner of the bounding box.
        y1 (int): y-coordinate of the top-left corner of the bounding box.
        x2 (int): x-coordinate of the bottom-right corner of the bounding box.
        y2 (int): y-coordinate of the bottom-right corner of the bounding box.
        pbs (torch.Tensor): List of predicted bounding boxes.

    Returns:
    -------
        torch.Tensor: Mesh function.

    """
    device = pbs.device
    xx, yy = torch.meshgrid(
        torch.linspace(x1, x2, x2 - x1 + 1).to(device),
        torch.linspace(y1, y2, y2 - y1 + 1).to(device),
        indexing="xy",
    )
    outxx = (xx.reshape((1, -1)) >= pbs[:, 0, None]) & (
        xx.reshape((1, -1)) <= (pbs[:, 2, None])
    )
    outyy = (yy.reshape((1, -1)) >= pbs[:, 1, None]) & (
        yy.reshape((1, -1)) <= (pbs[:, 3, None])
    )

    Z = torch.any(outxx & outyy, dim=0).reshape((x2 - x1 + 1, y2 - y1 + 1))
    return Z


def get_covered_areas_of_gt_union(pred_boxes, true_boxes):
    """Calculate covered areas of ground truth union."""
    device = pred_boxes[0].device
    areas = []
    for tb, pbs in zip(true_boxes, pred_boxes):
        if len(pbs) == 0:
            areas.append(torch.tensor(0).float().to(device))
            continue
        if len(pbs.shape) == 1:
            pbs = pbs.unsqueeze(0)
        x1, y1, x2, y2 = tb
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        Z = mesh_func(x1, y1, x2, y2, pbs)

        area = Z.sum() / ((x2 - x1 + 1) * (y2 - y1 + 1))
        areas.append(area)
    areas = torch.stack(areas)
    return areas


def fast_covered_areas_of_gt(pred_boxes, true_boxes):
    """Fast calculation of covered areas of ground truth."""
    # device = pred_boxes[0].device
    # areas = []
    # for tb, pb in zip(true_boxes, pred_boxes):
    #     if len(pb) == 0:
    #         areas.append(torch.tensor(0).float().to(device))
    #         continue
    #     if len(pb.shape) > 1:
    #         pb = pb[0]
    #     if tb.shape[0] != 4 or pb.shape[0] != 4:
    #         assert False
    #     area = contained(tb, pb)
    #     areas.append(area)
    # areas = torch.stack(areas)
    areas = contained(true_boxes, pred_boxes)
    return areas


def contained(tb: torch.Tensor, pb: torch.Tensor) -> torch.Tensor:
    """Compute the intersection over union (IoU) between two bounding boxes.

    Args:
    ----
        tb (torch.Tensor): Ground truth bounding boxes (N, 4).
        pb (torch.Tensor): Predicted bounding boxes (N, 4).

    Returns:
    -------
        torch.Tensor: IoU values (N,).

    """
    # TODO: only considering the case where all matchings exist, or none exist
    if pb.nelement() == 0:
        return torch.zeros(tb.size(0), device=tb.device)

    xA = torch.maximum(tb[:, 0], pb[:, 0])
    yA = torch.maximum(tb[:, 1], pb[:, 1])
    xB = torch.minimum(tb[:, 2], pb[:, 2])
    yB = torch.minimum(tb[:, 3], pb[:, 3])

    interArea = (xB - xA + 1).clamp(min=0) * (yB - yA + 1).clamp(min=0)
    tbArea = (tb[:, 2] - tb[:, 0] + 1) * (tb[:, 3] - tb[:, 1] + 1)

    ratio = interArea / tbArea
    return ratio  # .clamp(min=0) # TODO tmp fix to avoid


def contained_old(tb, pb):
    """Compute the intersection over union (IoU) between two bounding boxes.

    Args:
    ----
        tb (List[int]): Ground truth bounding box.
        pb (List[int]): Predicted bounding box.

    Returns:
    -------
        float: Intersection over union (IoU) value.

    """
    xA = max(tb[0], pb[0])
    yA = max(tb[1], pb[1])
    xB = min(tb[2], pb[2])
    yB = min(tb[3], pb[3])

    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    tbArea = (tb[2] - tb[0] + 1) * (tb[3] - tb[1] + 1)

    iou = interArea / tbArea
    return iou


def f_iou(boxA, boxB):
    """Compute the intersection over union (IoU) between two bounding boxes.

    Args:
    ----
        boxA (List[int]): First bounding box.
        boxB (List[int]): Second bounding box.

    Returns:
    -------
        float: Intersection over union (IoU) value.

    """
    # Handle tensor inputs by flattening them
    if hasattr(boxA, 'flatten'):
        boxA = boxA.flatten()
    if hasattr(boxB, 'flatten'):
        boxB = boxB.flatten()

    xA = torch.max(boxA[0], boxB[0])
    yA = torch.max(boxA[1], boxB[1])
    xB = torch.min(boxA[2], boxB[2])
    yB = torch.min(boxA[3], boxB[3])

    interArea = torch.max(torch.tensor(0), xB - xA + 1) * torch.max(torch.tensor(0), yB - yA + 1)
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # To ensure numerical stability
    iou = interArea / (boxAArea + boxBArea - interArea + 1e-12)
    return iou


def generalized_iou(boxA, boxB):
    """Compute the Generalized Intersection over Union (GIoU) between two bounding boxes.

    Args:
    ----
        boxA (List[int]): First bounding box.
        boxB (List[int]): Second bounding box.

    Returns:
    -------
        float: Generalized Intersection over Union (GIoU) value.

    """
    # Calculate the intersection
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # Calculate the union
    unionArea = boxAArea + boxBArea - interArea

    # Calculate IoU
    iou = interArea / (unionArea + 1e-12)

    # Calculate the convex hull
    xC1 = min(boxA[0], boxB[0])
    yC1 = min(boxA[1], boxB[1])
    xC2 = max(boxA[2], boxB[2])
    yC2 = max(boxA[3], boxB[3])

    convexHullArea = (xC2 - xC1 + 1) * (yC2 - yC1 + 1)

    # Calculate GIoU
    giou = iou - (convexHullArea - unionArea) / (convexHullArea + 1e-12)

    return giou


def vectorized_generalized_iou(
    boxesA: np.ndarray,
    boxesB: np.ndarray,
) -> np.ndarray:
    """Compute the Generalized Intersection over Union (GIoU) between two sets of
    bounding boxes.

    Calculates the GIoU for every pair of boxes between boxesA and boxesB.

    Args:
    ----
        boxesA (np.ndarray): A NumPy array of shape (N, 4) representing N bounding boxes.
                             Each row is [x1, y1, x2, y2].
        boxesB (np.ndarray): A NumPy array of shape (M, 4) representing M bounding boxes.
                             Each row is [x1, y1, x2, y2].

    Returns:
    -------
        np.ndarray: A NumPy array of shape (N, M) containing the GIoU values for
                    each pair of boxes (boxesA[i], boxesB[j]).

    Raises:
    ------
        ValueError: If input arrays do not have shape (N, 4) or (M, 4).

    """
    if boxesA.ndim != 2 or boxesA.shape[1] != 4:
        raise ValueError(
            f"boxesA must have shape (N, 4), but got {boxesA.shape}",
        )
    if boxesB.ndim != 2 or boxesB.shape[1] != 4:
        raise ValueError(
            f"boxesB must have shape (M, 4), but got {boxesB.shape}",
        )

    zero_tensor = torch.tensor(0).to(boxesA.device)

    # Ensure inputs are float arrays for calculations
    boxesA = boxesA.float()
    boxesB = boxesB.float()

    # Add a dimension to boxesA and boxesB for broadcasting
    # boxesA becomes (N, 1, 4), boxesB becomes (1, M, 4)
    boxesA_reshaped = boxesA[:, None, :]
    boxesB_reshaped = boxesB[None, :, :]

    # --- Calculate Intersection ---
    # Top-left corner of intersection (xA, yA)
    # Shape: (N, M)
    xA = torch.maximum(boxesA_reshaped[:, :, 0], boxesB_reshaped[:, :, 0])
    yA = torch.maximum(boxesA_reshaped[:, :, 1], boxesB_reshaped[:, :, 1])

    # Bottom-right corner of intersection (xB, yB)
    # Shape: (N, M)
    xB = torch.minimum(boxesA_reshaped[:, :, 2], boxesB_reshaped[:, :, 2])
    yB = torch.minimum(boxesA_reshaped[:, :, 3], boxesB_reshaped[:, :, 3])

    # Area of intersection
    # Add 1 because coordinates are inclusive (as in the original code)
    # Use np.maximum(0, ...) to handle cases with no overlap
    # Shape: (N, M)
    interWidth = torch.maximum(zero_tensor, xB - xA + 1)
    interHeight = torch.maximum(zero_tensor, yB - yA + 1)
    interArea = interWidth * interHeight

    # --- Calculate Individual Box Areas ---
    # Add 1 because coordinates are inclusive
    # Shape: (N, 1) and (1, M) - will broadcast correctly later
    boxAArea = (boxesA_reshaped[:, :, 2] - boxesA_reshaped[:, :, 0] + 1) * (
        boxesA_reshaped[:, :, 3] - boxesA_reshaped[:, :, 1] + 1
    )
    boxBArea = (boxesB_reshaped[:, :, 2] - boxesB_reshaped[:, :, 0] + 1) * (
        boxesB_reshaped[:, :, 3] - boxesB_reshaped[:, :, 1] + 1
    )

    # --- Calculate Union ---
    # Shape: (N, M)
    unionArea = boxAArea + boxBArea - interArea

    # Add a small epsilon to avoid division by zero
    epsilon = 1e-12

    # --- Calculate IoU ---
    # Shape: (N, M)
    iou = interArea / (unionArea + epsilon)

    # --- Calculate Convex Hull (Enclosing Box) ---
    # Top-left corner of convex hull (xC1, yC1)
    # Shape: (N, M)
    xC1 = torch.minimum(boxesA_reshaped[:, :, 0], boxesB_reshaped[:, :, 0])
    yC1 = torch.minimum(boxesA_reshaped[:, :, 1], boxesB_reshaped[:, :, 1])

    # Bottom-right corner of convex hull (xC2, yC2)
    # Shape: (N, M)
    xC2 = torch.maximum(boxesA_reshaped[:, :, 2], boxesB_reshaped[:, :, 2])
    yC2 = torch.maximum(boxesA_reshaped[:, :, 3], boxesB_reshaped[:, :, 3])

    # Area of convex hull
    # Add 1 because coordinates are inclusive
    # Shape: (N, M)
    convexHullWidth = torch.maximum(
        zero_tensor,
        xC2 - xC1 + 1,
    )  # Max with 0 just in case of weird inputs
    convexHullHeight = torch.maximum(zero_tensor, yC2 - yC1 + 1)
    convexHullArea = convexHullWidth * convexHullHeight

    # --- Calculate GIoU ---
    # Shape: (N, M)
    giou = iou - (convexHullArea - unionArea) / (convexHullArea + epsilon)

    return giou


def assymetric_hausdorff_distance_old(true_box, pred_box):
    """Calculate asymmetric Hausdorff distance between true and predicted box (legacy version)."""
    up_distance = pred_box[1] - true_box[1]
    down_distance = true_box[3] - pred_box[3]
    left_distance = pred_box[0] - true_box[0]
    right_distance = true_box[2] - pred_box[2]

    # Return the maximum signed distance
    return max(
        up_distance,
        down_distance,
        left_distance,
        right_distance,
    )


def assymetric_hausdorff_distance(true_boxes, pred_boxes):
    """Calculate asymmetric Hausdorff distance between sets of boxes."""
    true_boxes = true_boxes.clone()
    pred_boxes = pred_boxes.clone()
    true_boxes[:, :2] *= -1
    pred_boxes[:, 2:] *= -1
    distances = true_boxes[:, None, :] + pred_boxes[None, :, :]
    distances = torch.max(distances, dim=-1).values
    return distances


def f_lac(true_cls, pred_cls):
    """Calculate LAC (Loss Adaptive Conformal) score for classification."""
    m = len(pred_cls)
    n = len(true_cls)

    # Handle 2D true_cls input
    if true_cls.dim() > 1:
        true_cls_flat = true_cls.flatten()
    else:
        true_cls_flat = true_cls

    # Create expanded indices for gathering
    indices = true_cls_flat.unsqueeze(0).expand(m, n)
    result = pred_cls.gather(1, indices.T)
    result = 1 - result
    return result


def rank_distance(true_cls, pred_cls):
    """Calculate rank distance between true and predicted classes."""
    # Flatten true_cls if it has extra dimensions
    if true_cls.dim() > 1:
        true_cls = true_cls.flatten()

    sorted_indices = torch.argsort(pred_cls, descending=True, dim=1)
    ranks = (sorted_indices == true_cls.unsqueeze(1)).nonzero(
        as_tuple=True,
    )[1]
    return ranks


def match_predictions_to_true_boxes(  # noqa: C901
    preds,
    distance_function,
    overload_confidence_threshold=None,
    verbose=False,
    hungarian=False,
    idx=None,
    class_factor: float = 0.25,
) -> None:
    """Match predictions to true boxes. Done in place, modifies the preds object."""

    # TODO(leo): switch to gpu
    def dist_iou(x, y):
        return -f_iou(x, y)

    def dist_generalized_iou(x, y):
        return -generalized_iou(x, y)

    DISTANCE_FUNCTIONS = {
        "iou": dist_iou,
        "giou": dist_generalized_iou,
        "hausdorff": assymetric_hausdorff_distance_old,
        "lac": None,
        "mix": None,
    }

    if verbose and distance_function is None:
        print("Using default:  asymmetric Hausdorff distance")

    if distance_function not in DISTANCE_FUNCTIONS:
        raise ValueError(
            f"Distance function {distance_function} not supported, must be one of {DISTANCE_FUNCTIONS.keys()}",
        )

    DISTANCE_FUNCTIONS[distance_function]

    all_matching = []
    if overload_confidence_threshold is not None:
        conf_thr = overload_confidence_threshold
    elif preds.confidence_threshold is not None:
        conf_thr = preds.confidence_threshold
    else:
        conf_thr = 0

    if not isinstance(conf_thr, torch.Tensor):
        conf_thr = torch.tensor(conf_thr)


    # To only update it on a single image
    if idx is not None:
        # filter pred_boxes with low objectness
        pred_boxess = [
            preds.pred_boxes[idx][preds.confidences[idx] >= conf_thr],
        ]
        true_boxess = [preds.true_boxes[idx]]

        preds_clss = [preds.pred_cls[idx][preds.confidences[idx] >= conf_thr]]

        true_clss = [preds.true_cls[idx]]

    else:
        pred_boxess = [
            x[y >= conf_thr]
            for x, y in zip(preds.pred_boxes, preds.confidences)
        ]
        true_boxess = list(preds.true_boxes)

        preds_clss = [
            x[y >= conf_thr] for x, y in zip(preds.pred_cls, preds.confidences)
        ]

        true_clss = list(preds.true_cls)

    for pred_boxes, true_boxes, pred_cls, true_cls in tqdm(
        zip(pred_boxess, true_boxess, preds_clss, true_clss),
        disable=not verbose,
    ):
        if len(true_boxes) == 0:
            matching = []
        elif len(pred_boxes) == 0:
            matching = [[]] * len(true_boxes)
        else:
            true_boxes = true_boxes.clone()
            pred_boxes = pred_boxes.clone()
            if distance_function == "hausdorff":
                distance_matrix = assymetric_hausdorff_distance(
                    true_boxes,
                    pred_boxes,
                )
            elif distance_function == "lac":
                distance_matrix = f_lac(true_cls, pred_cls)
            elif distance_function == "mix":
                l_lac = f_lac(true_cls, pred_cls)
                l_ass = assymetric_hausdorff_distance(true_boxes, pred_boxes)
                l_ass /= torch.max(l_ass)
                distance_matrix = (
                    class_factor * l_lac + (1 - class_factor) * l_ass
                )
            elif distance_function == "giou":
                distance_matrix = vectorized_generalized_iou(
                    true_boxes,
                    pred_boxes,
                )
            else:
                raise NotImplementedError(
                    "Only hausdorff and lac are supported",
                )
            if hungarian:
                # TODO: to test
                row_ind, col_ind = linear_sum_assignment(distance_matrix)
                matching = [[]] * len(true_boxes)
                for x, y in zip(row_ind, col_ind):
                    if x < len(true_boxes) and y < len(pred_boxes):
                        matching[x] = [y]
            else:
                matching = (
                    torch.argmin(distance_matrix, dim=1)
                    .cpu()
                    .numpy()
                    .reshape(-1, 1)
                    .tolist()
                )

        assert len(matching) == len(true_boxes)
        all_matching.append(matching)

    if idx is not None:
        return all_matching[0]
    preds.matching = all_matching
    return all_matching


def apply_margins(pred_boxes: List[torch.Tensor], Qs, mode="additive"):
    """Apply margins to predicted bounding boxes for conformal prediction."""
    n = len(pred_boxes)
    new_boxes = []
    device = pred_boxes[0].device

    # Handle both list and scalar inputs for Qs
    if isinstance(Qs, (list, tuple)):
        Qst = torch.FloatTensor(Qs).to(device)
    else:
        Qst = torch.FloatTensor([Qs]).to(device)

    correction_factor = torch.FloatTensor([[-1, -1, 1, 1]]).to(device)

    for i in range(n):
        if not pred_boxes[i].numel():
            new_boxes.append(torch.tensor([]).float().to(device))
            continue
        if mode == "additive":
            if len(Qst) == 4:  # Individual margins for each coordinate
                margin = torch.mul(correction_factor, Qst)
            else:  # Single margin value
                margin = torch.mul(correction_factor, Qst[0])
            new_box = pred_boxes[i].float() + margin
        elif mode == "multiplicative":
            w = pred_boxes[i][:, 2] - pred_boxes[i][:, 0]
            h = pred_boxes[i][:, 3] - pred_boxes[i][:, 1]
            if len(Qst) == 4:  # Individual scaling factors
                margin = torch.stack(
                    (-w * (Qst[0] - 1), -h * (Qst[1] - 1), w * (Qst[2] - 1), h * (Qst[3] - 1)),
                    dim=-1,
                )
            else:  # Single scaling factor
                margin = torch.mul(
                    torch.stack(
                        (-w, -h, w, h),
                        dim=-1,
                    ),
                    (Qst[0] - 1),
                )
            new_box = pred_boxes[i].float() + margin
        # TODO: implement
        elif mode == "adaptive":
            raise NotImplementedError("adaptive mode not implemented yet")
        new_boxes.append(new_box.float())
    return new_boxes


def compute_risk_object_level(
    conformalized_predictions,
    predictions,
    loss,
    return_list: bool = False,
) -> torch.Tensor:
    """Input : conformal and true boxes of a all images."""
    # filter out boxes with low objectness
    losses = []
    true_boxes = predictions.true_boxes
    true_cls = predictions.true_cls
    conf_boxes = conformalized_predictions.conf_boxes
    conf_cls = conformalized_predictions.conf_cls

    assert true_boxes[0].device == conf_boxes[0].device, (
        true_boxes[0].device,
        conf_boxes[0].device,
    )
    assert true_cls[0].device == conf_cls[0][0].device, (
        true_cls[0].device,
        conf_cls[0][0].device,
    )
    device = conf_boxes[0].device

    for i, tb_all in enumerate(true_boxes):
        true_boxes_i = tb_all  # true_boxes[i]
        true_cls_i = true_cls[i]
        conf_boxes_i = conf_boxes[i]
        conf_cls_i = conf_cls[i]
        matching_i = predictions.matching[i]

        for j, _ in enumerate(true_boxes_i):
            matching_i_j = matching_i[j]

            if len(matching_i_j) == 0:
                matched_conf_boxes_i_j = torch.tensor([]).float().to(device)
                matched_conf_cls_i_j = torch.tensor([]).float().to(device)
                # Unsure above
            else:
                matched_conf_boxes_i_j = torch.stack(
                    [conf_boxes_i[m] for m in matching_i[j]],
                )
                matched_conf_cls_i_j = torch.stack(
                    [conf_cls_i[m] for m in matching_i[j]],
                )
            loss_value = loss(
                [true_boxes_i[j]],  # .to(device)],
                [true_cls_i[j]],  # .to(device)],
                [matched_conf_boxes_i_j],
                [matched_conf_cls_i_j],
            )
            # TODO: investigate why we need to do that
            loss_value = (
                loss_value
                if len(loss_value.shape) > 0
                else loss_value.unsqueeze(0)
            )
            losses.append(loss_value)
    losses = torch.stack(losses).ravel()
    return losses if return_list else torch.mean(losses)


def compute_risk_image_level(
    conformalized_predictions,
    predictions,
    loss,
    # aggregator="mean",
    return_list: bool = False,
) -> torch.Tensor:
    """Compute image-level risk for conformal prediction."""
    # aggregtion_funcs = {
    #     "mean": torch.mean,
    #     "sum": torch.sum,
    #     "max": torch.max,
    # }
    # if aggregator not in aggregtion_funcs:
    #     raise ValueError(
    #         f"Aggregator {aggregator} not supported, must be one of {aggregtion_funcs.keys()}",
    #     )
    # aggregator_func = aggregtion_funcs[aggregator]

    losses = []
    true_boxes = predictions.true_boxes
    true_cls = predictions.true_cls
    conf_boxes = conformalized_predictions.conf_boxes
    conf_cls = conformalized_predictions.conf_cls
    device = conf_boxes[0].device
    for i in range(len(true_boxes)):
        true_boxes_i = true_boxes[i]
        conf_boxes_i = conf_boxes[i]
        true_cls_i = true_cls[i].to(
            device,
        )  # TODO: why the cuda for cls and not boxes
        conf_cls_i = conf_cls[i]
        # TODO(leo): temporary fix
        matching_i = predictions.matching[i]
        # matched_conf_boxes_i = list(
        #     [
        #         (#TODO: here we have a list of n [tensor] while for the other we just have n x tensor, need to pick which
        #             torch.stack([conf_boxes_i[m] for m in matching_i[j]])
        #             if len(matching_i[j]) > 0
        #             else torch.tensor([]).float().to(device)
        #         )
        #         for j in range(len(true_boxes_i))
        #     ],
        # )
        # TODO: only works when you have 1 matching box
        matched_conf_boxes_i = [
            (
                torch.stack([conf_boxes_i[m] for m in matching_i[j]])[0]
                if len(matching_i[j]) > 0
                else torch.tensor([]).float().to(device)
            )
            for j in range(len(true_boxes_i))
        ]
        matched_conf_boxes_i = (
            torch.stack(matched_conf_boxes_i)
            if len(matched_conf_boxes_i) > 0
            and matched_conf_boxes_i[0].numel() > 0
            else torch.tensor([]).float().to(device)
        )
        # print(matched_conf_boxes_i.shape)
        matched_conf_cls_i = [
            (
                torch.stack([conf_cls_i[m] for m in matching_i[j]])
                if len(matching_i[j]) > 0
                else torch.tensor([]).float().to(device)
            )
            for j in range(len(true_boxes_i))
        ]
        # print(type(matched_conf_boxes_i), type(true_boxes_i))
        # print("Shape", true_boxes_i.shape, matched_conf_boxes_i.shape)
        loss_value = loss(
            true_boxes_i,
            true_cls_i,
            matched_conf_boxes_i,
            matched_conf_cls_i,
        )
        # loss_value_i = aggregator_func(losses_i)
        losses.append(loss_value)
    losses = torch.stack(losses).ravel()
    return losses if return_list else torch.mean(losses)


def compute_risk_image_level_confidence(
    conformalized_predictions,
    predictions,
    confidence_loss,
    other_losses=None,
    # aggregator="mean",
    return_list: bool = False,
) -> torch.Tensor:
    """Compute image-level confidence risk for conformal prediction."""
    # aggregtion_funcs = {
    #     "mean": torch.mean,
    #     "sum": torch.sum,
    #     "max": torch.max,
    # }
    # if aggregator not in aggregtion_funcs:
    #     raise ValueError(
    #         f"Aggregator {aggregator} not supported, must be one of {aggregtion_funcs.keys()}",
    #     )
    # aggregator_func = aggregtion_funcs[aggregator]

    losses = []
    true_boxes = predictions.true_boxes
    true_cls = predictions.true_cls
    conf_boxes = conformalized_predictions.conf_boxes
    conf_cls = conformalized_predictions.conf_cls
    device = conf_boxes[0].device
    for i in range(len(true_boxes)):
        true_boxes_i = true_boxes[i]
        conf_boxes_i = conf_boxes[i]
        true_cls_i = true_cls[i].to(
            device,
        )  # TODO: why the cuda for cls and not boxes
        conf_cls_i = conf_cls[i]
        # TODO(leo): temporary fix
        matching_i = predictions.matching[i]
        tmp_matched_boxes = [
            (
                torch.stack([conf_boxes_i[m] for m in matching_i[j]])
                if len(matching_i[j]) > 0
                else torch.tensor([]).float().to(device)
            )
            for j in range(len(true_boxes_i))
        ]
        matched_conf_boxes_i = (
            torch.stack(tmp_matched_boxes)
            if len(tmp_matched_boxes) > 0
            else torch.tensor([]).float().to(device)
        )
        matched_conf_cls_i = [
            (
                torch.stack([conf_cls_i[m] for m in matching_i[j]])
                if len(matching_i[j]) > 0
                else torch.tensor([]).float().to(device)
            )
            for j in range(len(true_boxes_i))
        ]
        conf_loss_value_i = confidence_loss(
            true_boxes_i,
            true_cls_i,
            conf_boxes_i,
            conf_cls_i,
        )

        if other_losses is None:
            other_losses_i = []
            loss_value_i = conf_loss_value_i
        else:
            ## FIXME: MUST BE HANDLED PROPERLY FOR EACH INPUT IMAGE SIZE
            MAGIC_BIG_MARGIN = [2500, 2500, 2500, 2500]
            matched_conf_boxes_i = apply_margins(
                [matched_conf_boxes_i],
                MAGIC_BIG_MARGIN,
                mode="additive",
            )[0]

            # Second, prediction sets for classification with always everything
            n_classes = len(predictions.pred_cls[0][0].squeeze())
            matched_conf_cls_i = [
                torch.arange(n_classes)[None, ...].to(device)
                for _ in range(len(matched_conf_cls_i))
            ]

            other_losses_i = [
                loss(
                    true_boxes_i,
                    true_cls_i,
                    matched_conf_boxes_i,
                    matched_conf_cls_i,
                )
                for loss in other_losses
            ]

            loss_value_i = torch.max(
                torch.stack([conf_loss_value_i] + other_losses_i),
            )

        # loss_value_i = aggregator_func(losses_i)
        losses.append(loss_value_i)
    losses = torch.stack(losses).ravel()
    return losses if return_list else torch.mean(losses)
