from logging import getLogger
from typing import List

logger = getLogger("cods")

import numpy as np
import torch
from tqdm import tqdm


# def get_classif_preds_from_od_preds(
#     preds: ODPredictions,
# ) -> ClassificationPredictions:
#     """Convert object detection predictions to classification predictions.

#     Args:
#     ----
#         preds (ODPredictions): Object detection predictions.

#     Returns:
#     -------
#         ClassificationPredictions: Classification predictions.

#     """
#     logger.error("Currently only handling object level guarantees")
#     dataset_name = preds.dataset_name
#     split_name = preds.split_name
#     image_paths = preds.image_paths
#     idx_to_cls = None

#     if preds.matching is None:
#         raise ValueError("Warning: preds.matching is None")
#     matching = preds.matching

#     true_cls = []
#     pred_cls = []
#     for i, true_cls_img in enumerate(preds.true_cls):
#         pred_cls_img = []
#         for j, true_cls_box in enumerate(true_cls_img):
#             pred_cls_box = preds.pred_cls[i][matching[i][j]]
#             pred_cls_img.append(pred_cls_box)
#         # TODO: fix this behavior
#         if len(pred_cls_img) == 0 or torch.stack(pred_cls_img).shape[1] == 0:
#             # TODO: Document this choice
#             # pred_cls_img = torch.zeros((0)).cuda()
#             if len(true_cls_img) == 0:
#                 continue
#             # raise ValueError("Warning: len(pred_cls_img) == 0")
#             pred_cls_img = torch.zeros((len(true_cls_img), 91)).cuda()
#         else:
#             pred_cls_img = torch.stack(pred_cls_img)
#         true_cls.append(true_cls_img)
#         pred_cls.append(pred_cls_img)
#     # pred_cls = list([x.squeeze() for x in pred_cls])
#     # pred_cls = list([x if len(x.shape) >= 2 else x.unsqueeze(0) for x in pred_cls])
#     # print([x.shape for x in pred_cls if len(x.shape) > 2])
#     pred_cls = pred_cls
#     true_cls = torch.cat(true_cls).cuda()

#     obj = ClassificationPredictions(
#         dataset_name=dataset_name,
#         split_name=split_name,
#         image_paths=image_paths,
#         idx_to_cls=idx_to_cls,
#         true_cls=true_cls,
#         pred_cls=pred_cls,
#     )
#     preds.preds_cls = obj
#     return obj


# def flatten_conf_cls(conf_cls: List[List[torch.Tensor]]) -> List[torch.Tensor]:
#     """Flatten nested arrays into a single list.

#     Args:
#     ----
#         conf_cls (List[List[torch.Tensor]]): Nested arrays.

#     Returns:
#     -------
#         List[torch.Tensor]: Flattened list.

#     """
#     conf_cls = [item for sublist in conf_cls for item in sublist]
#     return conf_cls


# def get_conf_cls_for_od(
#     od_preds: ODPredictions,
#     conformalizer: Union[
#         ClassificationConformalizer,
#         ClassificationToleranceRegion,
#     ],
# ) -> List[List[torch.Tensor]]:
#     """Get confidence scores for object detection predictions.

#     Args:
#     ----
#         od_preds (ODPredictions): Object detection predictions.
#         conformalizer (Union[ClassificationConformalizer, ClassificationToleranceRegion]): Conformalizer object.

#     Returns:
#     -------
#         List[List[torch.Tensor]]: Confidence scores for each object detection prediction.

#     """
#     logger.error("Currently only handling object level guarantees")
#     if od_preds.matching is None:
#         raise ValueError("Warning: od_preds.matching is None")
#     matching = od_preds.matching
#     conf_cls = []
#     for i, true_cls_img in enumerate(od_preds.true_cls):
#         pre_pred_cls_img = od_preds.pred_cls[i]
#         pred_cls_img = []
#         for j, true_cls_box in enumerate(true_cls_img):
#             pred_cls_box = pre_pred_cls_img[matching[i][j]]
#             pred_cls_img.append(pred_cls_box)
#         if len(pred_cls_img) != len(true_cls_img):
#             raise ValueError(
#                 "Warning: len(pred_cls_img) != len(true_cls_img), "
#                 + str(len(pred_cls_img))
#                 + " != "
#                 + str(len(true_cls_img)),
#             )
#         if len(true_cls_img) == 0:
#             conf_cls.append([])
#             continue
#         conf_cls_img = conformalizer.conformalize(
#             ClassificationPredictions(
#                 dataset_name=od_preds.dataset_name,
#                 split_name=od_preds.split_name,
#                 image_paths=od_preds.image_paths,
#                 idx_to_cls=None,
#                 true_cls=true_cls_img,
#                 pred_cls=pred_cls_img,
#             ),
#         )
#         conf_cls.append(conf_cls_img)
#     return conf_cls


# def evaluate_cls_conformalizer(
#     od_preds: ODPredictions,
#     conf_cls: List[List[torch.Tensor]],
#     conformalizer: Union[
#         ClassificationConformalizer,
#         ClassificationToleranceRegion,
#     ],
#     verbose: bool = False,
# ) -> Tuple[torch.Tensor, torch.Tensor]:
#     """Evaluate the performance of a classification conformalizer.

#     Args:
#     ----
#         od_preds (ODPredictions): Object detection predictions.
#         conf_cls (List[List[torch.Tensor]]): Confidence scores for each object detection prediction.
#         conformalizer (Union[ClassificationConformalizer, ClassificationToleranceRegion]): Conformalizer object.
#         verbose (bool, optional): Whether to print verbose output. Defaults to False.

#     Returns:
#     -------
#         torch.Tensor: Coverage and set size for each object detection prediction.

#     """
#     logger.error("Currently only handling object level guarantees")
#     if od_preds.matching is None:
#         raise ValueError("Warning: od_preds.matching is None")
#     covs = []
#     set_sizes = []
#     for i, true_cls_img in enumerate(od_preds.true_cls):
#         pre_pred_cls_img = od_preds.pred_cls[i]
#         pred_cls_img = []
#         for j, true_cls_box in enumerate(true_cls_img):
#             # if len(od_preds.matching[i]) > j:
#             pred_cls_box = pre_pred_cls_img[od_preds.matching[i][j]]
#             pred_cls_img.append(pred_cls_box)
#             if len(conf_cls[i][j]) == 0:
#                 print(  # raise ValueError(
#                     f"Warning: len(conf_cls[i][j]) == 0, conf_cls[i][j] = {conf_cls[i][j]}",
#                 )
#             # else:
#             #    pred_cls_img.append(torch.empty(0).cuda())
#         if len(pred_cls_img) != len(true_cls_img):
#             raise ValueError(
#                 "Warning: len(pred_cls_img) != len(true_cls_img), "
#                 + str(len(pred_cls_img))
#                 + " != "
#                 + str(len(true_cls_img)),
#             )
#         if len(true_cls_img) == 0:
#             continue
#         coverage_cls, set_size_cls = conformalizer.evaluate(
#             ClassificationPredictions(
#                 dataset_name=od_preds.dataset_name,
#                 split_name=od_preds.split_name,
#                 image_paths=od_preds.image_paths,
#                 idx_to_cls=None,
#                 true_cls=true_cls_img,
#                 pred_cls=pred_cls_img,
#             ),
#             conf_cls[i],
#             verbose=verbose,
#         )
#         covs.append(coverage_cls)
#         set_sizes.append(set_size_cls)
#     covs = torch.cat(covs)
#     set_sizes = torch.cat(set_sizes)
#     return covs, set_sizes


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
    xx, yy = torch.meshgrid(
        torch.linspace(x1, x2, x2 - x1 + 1).cuda(),
        torch.linspace(y1, y2, y2 - y1 + 1).cuda(),
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
    """Compute the covered areas of ground truth bounding boxes using union.

    Args:
    ----
        pred_boxes (List[List[int]]): List of predicted bounding boxes.
        true_boxes (List[List[int]]): List of ground truth bounding boxes.

    Returns:
    -------
        torch.Tensor: Covered areas of ground truth bounding boxes.

    """
    areas = []
    for tb, pbs in zip(true_boxes, pred_boxes):
        if len(pbs) == 0:
            areas.append(torch.tensor(0).float().cuda())
            continue
        x1, y1, x2, y2 = tb
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        Z = mesh_func(x1, y1, x2, y2, pbs)

        area = Z.sum() / ((x2 - x1 + 1) * (y2 - y1 + 1))
        areas.append(area)
    areas = torch.stack(areas)
    return areas


# Deprecated
# def get_covered_areas_of_gt_max(pred_boxes, true_boxes):
#     """Compute the covered areas of ground truth bounding boxes using maximum.

#     Args:
#     ----
#         pred_boxes (List[List[int]]): List of predicted bounding boxes.
#         true_boxes (List[List[int]]): List of ground truth bounding boxes.

#     Returns:
#     -------
#         torch.Tensor: Covered areas of ground truth bounding boxes.

#     """
#     areas = []
#     for tb in true_boxes:
#         x1, y1, x2, y2 = tb
#         x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

#         p_areas = []
#         for pb in pred_boxes:
#             Z = mesh_func(x1, y1, x2, y2, pb[None, ...])

#             p_area = Z.sum() / ((x2 - x1 + 1) * (y2 - y1 + 1))
#             p_areas.append(p_area)
#         area = torch.max(p_areas)
#         areas.append(area)
#     areas = torch.stack(areas)
#     return areas


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
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou


def assymetric_hausdorff_distance(true_box, pred_box):
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


def match_predictions_to_true_boxes(
    preds,
    distance_function,
    overload_confidence_threshold=None,
    verbose=False,
):
    """ """
    dist_iou = lambda x, y: -f_iou(x, y)
    DISTANCE_FUNCTIONS = {
        "iou": dist_iou,
        "hausdorff": assymetric_hausdorff_distance,
    }

    if verbose and distance_function is None:
        # defaulting to assymetric hausdorff distance
        print("Using assymetric hausdorff distance")

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
    # filter pred_boxes with low objectness
    preds_boxes = [
        x[
            y >= conf_thr
        ]  # if len(x[y >= conf_thr]) > 0 else x[None, y.argmax()]
        for x, y in zip(preds.pred_boxes, preds.confidence)
    ]
    for pred_boxes, true_boxes in tqdm(
        zip(preds_boxes, preds.true_boxes),
        disable=not verbose,
    ):
        matching = []
        for true_box in true_boxes:
            distances = []
            # print("new gt")
            for pred_box in pred_boxes:
                # TODO replace with hausdorff distance ?
                # iou = f_iou(true_box, pred_box)
                # TODO: test
                # dist = -f_iou(true_box, pred_box)
                dist = f_dist(true_box, pred_box)
                dist = (
                    dist.cpu().numpy()
                    if isinstance(dist, torch.Tensor)
                    else dist
                )
                # print(dist)
                distances.append(dist)  # .cpu().numpy())
            if len(pred_boxes) == 0:
                matching.append([])
                continue
            # TODO: replace this by possible a set of matches
            # TODO: must always be an array
            matching.append([np.argmin(distances)])  # np.argmax(ious))
            # print(matching[-1])
        all_matching.append(matching)
        # break
    preds.matching = all_matching
    return all_matching


def apply_margins(pred_boxes: List[torch.Tensor], Qs, mode="additive"):
    n = len(pred_boxes)
    new_boxes = []
    Qst = torch.FloatTensor([Qs]).cuda()
    for i in range(n):
        if len(pred_boxes[i]) == 0:
            new_boxes.append(torch.tensor([]).float().cuda())
            continue
        if mode == "additive":
            new_box = pred_boxes[i] + torch.mul(
                torch.FloatTensor([[-1, -1, 1, 1]]).cuda(),
                Qst,
            )
        elif mode == "multiplicative":
            w = pred_boxes[i][:, 2] - pred_boxes[i][:, 0]
            h = pred_boxes[i][:, 3] - pred_boxes[i][:, 1]
            new_box = pred_boxes[i] + torch.mul(
                torch.FloatTensor([[-w, -h, w, h]]).cuda(),
                Qst,
            )
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
    for i in range(len(true_boxes)):
        true_boxes_i = true_boxes[i]
        true_cls_i = true_cls[
            i
        ].cuda()  # TODO: why the cuda for cls and not boxes
        conf_boxes_i = conf_boxes[i]
        conf_cls_i = conf_cls[i]
        matching_i = predictions.matching[i]
        for j in range(len(true_boxes_i)):
            matching_i_j = matching_i[j]
            if len(matching_i_j) == 0:
                matched_conf_boxes_i_j = torch.tensor([]).float().cuda()
                matched_conf_cls_i_j = torch.tensor([]).float().cuda()
                # Unsure above
            else:
                matched_conf_boxes_i_j = torch.stack(
                    [conf_boxes_i[m] for m in matching_i[j]],
                )
                matched_conf_cls_i_j = torch.stack(
                    [conf_cls_i[m] for m in matching_i[j]],
                )
            loss_value = loss(
                [true_boxes_i[j]],
                [true_cls_i[j].cuda()],
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

    for i in range(len(true_boxes)):
        true_boxes_i = true_boxes[i]
        true_cls_i = true_cls[
            i
        ].cuda()  # TODO: why the cuda for cls and not boxes
        conf_boxes_i = conf_boxes[i]
        conf_cls_i = conf_cls[i]
        # for j in range(len(true_boxes_i)):
        matching_i = predictions.matching[i]
        matched_conf_boxes_i = list(
            [
                torch.stack([conf_boxes_i[m] for m in matching_i[j]])
                if len(matching_i[j]) > 0
                else torch.tensor([]).float().cuda()
                for j in range(len(true_boxes_i))
            ],
        )
        matched_conf_cls_i = list(
            [
                torch.stack([conf_cls_i[m] for m in matching_i[j]])
                if len(matching_i[j]) > 0
                else torch.tensor([]).float().cuda()
                for j in range(len(true_boxes_i))
            ],
        )
        loss_value = loss(
            true_boxes_i,
            true_cls_i,  # .cuda(),  # TODO: why the cuda for cls and not boxes
            matched_conf_boxes_i,
            matched_conf_cls_i,
        )
        # loss_value_i = aggregator_func(losses_i)
        losses.append(loss_value)
    losses = torch.stack(losses).ravel()
    return losses if return_list else torch.mean(losses)
