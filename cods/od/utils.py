from logging import getLogger
from typing import List

logger = getLogger("cods")


import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm


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
        pbs (List[List[int]]): List of predicted bounding boxes.

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
    """ """
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


def contained(tb, pb):
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

    iou = interArea / float(tbArea)
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
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = max(boxA[2], boxB[2])
    yB = max(boxA[3], boxB[3])

    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
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


def assymetric_hausdorff_distance_old(true_box, pred_box):
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
    true_boxes[:, :2] *= -1
    pred_boxes[:, 2:] *= -1
    distances = true_boxes[:, None, :] + pred_boxes[None, :, :]
    distances = torch.max(distances, dim=-1).values
    return distances


def match_predictions_to_true_boxes(
    preds,
    distance_function,
    overload_confidence_threshold=None,
    verbose=False,
    hungarian=False,
    idx=None,
    class_factor: float = 5,
) -> None:
    """Matching predictions to true boxes. Done in place, modifies the preds object."""
    # TODO(leo): switch to gpu
    dist_iou = lambda x, y: -f_iou(x, y)
    dist_generalized_iou = lambda x, y: -generalized_iou(x, y)
    DISTANCE_FUNCTIONS = {
        "iou": dist_iou,
        "giou": dist_generalized_iou,
        "hausdorff": assymetric_hausdorff_distance_old,
        "lac": None,
    }

    if verbose and distance_function is None:
        print("Using default:  asymmetric Hausdorff distance")

    if distance_function not in DISTANCE_FUNCTIONS.keys():
        raise ValueError(
            f"Distance function {distance_function} not supported, must be one of {DISTANCE_FUNCTIONS.keys()}",
        )

    f_dist = DISTANCE_FUNCTIONS[distance_function]

    all_matching = []
    if overload_confidence_threshold is not None:
        conf_thr = overload_confidence_threshold
    elif preds.confidence_threshold is not None:
        conf_thr = preds.confidence_threshold
    else:
        conf_thr = 0

    if not isinstance(conf_thr, torch.Tensor):
        conf_thr = torch.tensor(conf_thr)

    device = preds.pred_boxes[0].device

    # To only update it on a single image
    if idx is not None:
        # filter pred_boxes with low objectness
        pred_boxess = [
            preds.pred_boxes[idx][preds.confidences[idx] >= conf_thr]
        ]
        true_boxess = [preds.true_boxes[idx]]

        preds_clss = [
            preds.pred_cls[idx][preds.confidences[idx] >= conf_thr]
        ]

        true_clss = [preds.true_cls[idx]]

    else:
        pred_boxess = [
            x[y >= conf_thr]
            for x, y in zip(preds.pred_boxes, preds.confidences)
        ]  
        true_boxess = [
            true_boxes_i for true_boxes_i in preds.true_boxes
        ]

        preds_clss = [
            x[y >= conf_thr]
            for x, y in zip(preds.pred_cls, preds.confidences)
        ]

        true_clss = [true_cls_i for true_cls_i in preds.true_cls]

    if hungarian:
        for pred_boxes, true_boxes in tqdm(
            zip(pred_boxess, true_boxess),
            disable=not verbose,
        ):
            if len(pred_boxes) == 0:
                matching = [[]] * len(true_boxes)
                all_matching.append(matching)
                # print(len(matching), len(true_boxes), matching)
                continue
            elif len(true_boxes) == 0:
                matching = []
                all_matching.append(matching)
                # print(len(matching), len(true_boxes), matching)
                continue
            else:
                image_distances = []
                for i, true_box in enumerate(true_boxes):
                    box_distances = []

                    for j, pred_box in enumerate(pred_boxes):
                        dist = f_dist(true_box, pred_box)
                        dist = (
                            dist.cpu().numpy()
                            if isinstance(dist, torch.Tensor)
                            else dist
                        )
                        box_distances.append(dist)

                    # TODO: replace this by possible a set of matches
                    # TODO: must always be an array
                    image_distances.append(
                        np.array(box_distances).astype(float)
                    )  # np.argmax(ious))
                    # print(matching[-1])

                image_distances = np.array(image_distances)
                row_ind, col_ind = linear_sum_assignment(image_distances)
                matching = [[]] * len(true_boxes)
                # print(len)
                for x, y in zip(row_ind, col_ind):
                    if x < len(true_boxes) and y < len(pred_boxes):
                        matching[x] = [y]
                # matching = list([[x] for x in col_ind])
                # assert len(matching) == len(
                #     true_boxes
                # ), f"{len(matching)}, {len(true_boxes)}, {matching}, {col_ind}"
                all_matching.append(matching)

    else:
        for pred_boxes, true_boxes, pred_cls, true_cls in tqdm(
            zip(pred_boxess, true_boxess, preds_clss, true_clss),
            disable=not verbose,
        ):
            if len(true_boxes) == 0:
                matching = []
            elif len(pred_boxes) == 0:
                matching = [[]] * len(true_boxes)
            else:
                # Assumption: pred_boxes and true_boxes are torch tensors of dimensions [n, 4] and [m,4]
                # where n is the number of predicted boxes and m is the number of true boxes
                # up_distance = pred_box[1] - true_box[1]
                # down_distance = true_box[3] - pred_box[3]
                # left_distance = pred_box[0] - true_box[0]
                # right_distance = true_box[2] - pred_box[2]
                
                true_boxes = true_boxes.clone()
                pred_boxes = pred_boxes.clone()
                distance_matrix = assymetric_hausdorff_distance(true_boxes, pred_boxes)
                matching = torch.argmin(distance_matrix, dim=1).cpu().numpy().reshape(-1, 1).tolist()
                

                #OLD, PROBABLY SLOWER
                # true_boxes = true_boxes.cpu().numpy()
                # pred_boxes = pred_boxes.cpu().numpy()
                # matching = []
                # for i, true_box in enumerate(true_boxes):
                #     cls = true_cls[i]
                #     distances = []
                #     for j, pred_box in enumerate(pred_boxes):
                #         score_true = pred_cls[j][cls].cpu().numpy()
                #         if distance_function == "lac":
                #             dist = 1-score_true
                #         else:
                #             dist = f_dist(true_box, pred_box)
                #         dist = (
                #             dist.cpu().numpy()
                #             if isinstance(dist, torch.Tensor)
                #             else dist
                #         )
                #         if class_factor is not None:
                #             c = class_factor  # 2 #TODO clarify this
                #             dist = dist * (
                #                 1 + c * (1 - score_true)
                #             )  # to rethink the factor
                #         distances.append(dist)  
                #     matching.append([np.argmin(distances)])  
            assert len(matching) == len(true_boxes)
            all_matching.append(matching)

    if idx is not None:
        return all_matching[0]
    else:
        preds.matching = all_matching
        return all_matching


def apply_margins(pred_boxes: List[torch.Tensor], Qs, mode="additive"):
    n = len(pred_boxes)
    new_boxes = []
    device = pred_boxes[0].device
    Qst = torch.FloatTensor([Qs]).to(device)
    correction_factor = torch.FloatTensor([[-1, -1, 1, 1]]).to(device)

    for i in range(n):
        if not pred_boxes[i].numel():
            new_boxes.append(torch.tensor([]).float().to(device))
            continue
        if mode == "additive":
            new_box = pred_boxes[i] + torch.mul(
                correction_factor,
                Qst,
            )
        elif mode == "multiplicative":
            w = pred_boxes[i][:, 2] - pred_boxes[i][:, 0]
            h = pred_boxes[i][:, 3] - pred_boxes[i][:, 1]
            margin = torch.mul(
                torch.stack(
                    (-w, -h, w, h),
                    dim=-1,
                ),
                Qst,
            )
            new_box = pred_boxes[i] + margin
        # TODO: implement
        elif mode == "adaptive":
            raise NotImplementedError("adaptive mode not implemented yet")
        new_boxes.append(new_box)
    return new_boxes


def compute_risk_object_level(
    conformalized_predictions,
    predictions,
    loss,
    return_list: bool = False,
) -> torch.Tensor:
    """Input : conformal and true boxes of a all images"""
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
        matched_conf_boxes_i = list(
            [
                (
                    torch.stack([conf_boxes_i[m] for m in matching_i[j]])
                    if len(matching_i[j]) > 0
                    else torch.tensor([]).float().to(device)
                )
                for j in range(len(true_boxes_i))
            ],
        )
        matched_conf_cls_i = list(
            [
                (
                    torch.stack([conf_cls_i[m] for m in matching_i[j]])
                    if len(matching_i[j]) > 0
                    else torch.tensor([]).float().to(device)
                )
                for j in range(len(true_boxes_i))
            ],
        )
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
        matched_conf_cls_i = list(
            [
                (
                    torch.stack([conf_cls_i[m] for m in matching_i[j]])
                    if len(matching_i[j]) > 0
                    else torch.tensor([]).float().to(device)
                )
                for j in range(len(true_boxes_i))
            ],
        )
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
                torch.stack([conf_loss_value_i] + other_losses_i)
            )

        # loss_value_i = aggregator_func(losses_i)
        losses.append(loss_value_i)
    losses = torch.stack(losses).ravel()
    return losses if return_list else torch.mean(losses)
