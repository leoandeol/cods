from typing import Dict, List, Tuple, Union

import numpy as np
import torch
from tqdm import tqdm

from cods.classif.cp import ClassificationConformalizer
from cods.classif.data import ClassificationPredictions
from cods.classif.tr import ClassificationToleranceRegion
from cods.od.data import ODPredictions


def get_classif_preds_from_od_preds(preds: ODPredictions) -> ClassificationPredictions:
    """
    Convert object detection predictions to classification predictions.

    Args:
        preds (ODPredictions): Object detection predictions.

    Returns:
        ClassificationPredictions: Classification predictions.
    """
    dataset_name = preds.dataset_name
    split_name = preds.split_name
    image_paths = preds.image_paths
    idx_to_cls = None

    matching = matching_by_iou(preds)

    true_cls = []
    pred_cls = []
    for i, true_cls_img in enumerate(preds.true_cls):
        pred_cls_img = []
        for j, true_cls_box in enumerate(true_cls_img):
            pred_cls_box = preds.pred_cls[i][matching[i][j]]
            pred_cls_img.append(pred_cls_box)
        if len(pred_cls_img) == 0:
            pred_cls_img = torch.zeros((0)).cuda()
        else:
            pred_cls_img = torch.stack(pred_cls_img)
        true_cls.append(true_cls_img)
        pred_cls.append(pred_cls_img)

    pred_cls = torch.cat(pred_cls)
    true_cls = torch.cat(true_cls).cuda()

    obj = ClassificationPredictions(
        dataset_name=dataset_name,
        split_name=split_name,
        image_paths=image_paths,
        idx_to_cls=idx_to_cls,
        true_cls=true_cls,
        pred_cls=pred_cls,
    )
    preds.preds_cls = obj
    return obj


def flatten_conf_cls(conf_cls: List[List[torch.Tensor]]) -> List[torch.Tensor]:
    """
    Flatten nested arrays into a single list.

    Args:
        conf_cls (List[List[torch.Tensor]]): Nested arrays.

    Returns:
        List[torch.Tensor]: Flattened list.
    """
    conf_cls = [item for sublist in conf_cls for item in sublist]
    return conf_cls


def get_conf_cls_for_od(
    od_preds: ODPredictions,
    conformalizer: Union[ClassificationConformalizer, ClassificationToleranceRegion],
) -> List[List[torch.Tensor]]:
    """
    Get confidence scores for object detection predictions.

    Args:
        od_preds (ODPredictions): Object detection predictions.
        conformalizer (Union[ClassificationConformalizer, ClassificationToleranceRegion]): Conformalizer object.

    Returns:
        List[List[torch.Tensor]]: Confidence scores for each object detection prediction.
    """
    if od_preds.matching is None:
        matching = matching_by_iou(od_preds)
    else:
        matching = od_preds.matching
    conf_cls = []
    for i, true_cls_img in enumerate(od_preds.true_cls):
        pre_pred_cls_img = od_preds.pred_cls[i]
        pred_cls_img = []
        for j, true_cls_box in enumerate(true_cls_img):
            pred_cls_box = pre_pred_cls_img[matching[i][j]]
            pred_cls_img.append(pred_cls_box)
        if len(pred_cls_img) != len(true_cls_img):
            raise ValueError(
                "Warning: len(pred_cls_img) != len(true_cls_img), "
                + str(len(pred_cls_img))
                + " != "
                + str(len(true_cls_img))
            )
        if len(true_cls_img) == 0:
            conf_cls.append([])
            continue
        conf_cls_img = conformalizer.conformalize(
            ClassificationPredictions(
                dataset_name=od_preds.dataset_name,
                split_name=od_preds.split_name,
                image_paths=od_preds.image_paths,
                idx_to_cls=None,
                true_cls=true_cls_img,
                pred_cls=pred_cls_img,
            )
        )
        conf_cls.append(conf_cls_img)
    return conf_cls


def evaluate_cls_conformalizer(
    od_preds: ODPredictions,
    conf_cls: List[List[torch.Tensor]],
    conformalizer: Union[ClassificationConformalizer, ClassificationToleranceRegion],
    verbose: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Evaluate the performance of a classification conformalizer.

    Args:
        od_preds (ODPredictions): Object detection predictions.
        conf_cls (List[List[torch.Tensor]]): Confidence scores for each object detection prediction.
        conformalizer (Union[ClassificationConformalizer, ClassificationToleranceRegion]): Conformalizer object.
        verbose (bool, optional): Whether to print verbose output. Defaults to False.

    Returns:
        torch.Tensor: Coverage and set size for each object detection prediction.
    """
    covs = []
    set_sizes = []
    for i, true_cls_img in enumerate(od_preds.true_cls):
        pre_pred_cls_img = od_preds.pred_cls[i]
        pred_cls_img = []
        for j, true_cls_box in enumerate(true_cls_img):
            pred_cls_box = pre_pred_cls_img[od_preds.matching[i][j]]
            pred_cls_img.append(pred_cls_box)
            if len(conf_cls[i][j]) == 0:
                raise ValueError(
                    f"Warning: len(conf_cls[i][j]) == 0, conf_cls[i][j] = {conf_cls[i][j]}"
                )
        if len(pred_cls_img) != len(true_cls_img):
            raise ValueError(
                "Warning: len(pred_cls_img) != len(true_cls_img), "
                + str(len(pred_cls_img))
                + " != "
                + str(len(true_cls_img))
            )
        if len(true_cls_img) == 0:
            continue
        coverage_cls, set_size_cls = conformalizer.evaluate(
            ClassificationPredictions(
                dataset_name=od_preds.dataset_name,
                split_name=od_preds.split_name,
                image_paths=od_preds.image_paths,
                idx_to_cls=None,
                true_cls=true_cls_img,
                pred_cls=pred_cls_img,
            ),
            conf_cls[i],
            verbose=verbose,
        )
        covs.append(coverage_cls)
        set_sizes.append(set_size_cls)
    covs = torch.cat(covs)
    set_sizes = torch.cat(set_sizes)
    return covs, set_sizes


def mesh_func(x1, y1, x2, y2, pbs):
    """
    Compute mesh function.

    Args:
        x1 (int): x-coordinate of the top-left corner of the bounding box.
        y1 (int): y-coordinate of the top-left corner of the bounding box.
        x2 (int): x-coordinate of the bottom-right corner of the bounding box.
        y2 (int): y-coordinate of the bottom-right corner of the bounding box.
        pbs (List[List[int]]): List of predicted bounding boxes.

    Returns:
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
    """
    Compute the covered areas of ground truth bounding boxes using union.

    Args:
        pred_boxes (List[List[int]]): List of predicted bounding boxes.
        true_boxes (List[List[int]]): List of ground truth bounding boxes.

    Returns:
        torch.Tensor: Covered areas of ground truth bounding boxes.
    """
    areas = []
    for tb in true_boxes:
        x1, y1, x2, y2 = tb
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        Z = mesh_func(x1, y1, x2, y2, pred_boxes)

        area = Z.sum() / ((x2 - x1 + 1) * (y2 - y1 + 1))
        areas.append(area)
    areas = torch.stack(areas)
    return areas


def get_covered_areas_of_gt_max(pred_boxes, true_boxes):
    """
    Compute the covered areas of ground truth bounding boxes using maximum.

    Args:
        pred_boxes (List[List[int]]): List of predicted bounding boxes.
        true_boxes (List[List[int]]): List of ground truth bounding boxes.

    Returns:
        torch.Tensor: Covered areas of ground truth bounding boxes.
    """
    areas = []
    for tb in true_boxes:
        x1, y1, x2, y2 = tb
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        p_areas = []
        for pb in pred_boxes:
            Z = mesh_func(x1, y1, x2, y2, [pb])

            p_area = Z.sum() / ((x2 - x1 + 1) * (y2 - y1 + 1))
            p_areas.append(p_area)
        p_areas = torch.stack(p_areas)
        area = torch.max(p_areas)
        areas.append(area)
    areas = torch.stack(areas)
    return areas


def contained(tb, pb):
    """
    Compute the intersection over union (IoU) between two bounding boxes.

    Args:
        tb (List[int]): Ground truth bounding box.
        pb (List[int]): Predicted bounding box.

    Returns:
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
    """
    Compute the intersection over union (IoU) between two bounding boxes.

    Args:
        boxA (List[int]): First bounding box.
        boxB (List[int]): Second bounding box.

    Returns:
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


def matching_by_iou(preds, verbose=False):
    """
    Perform matching between ground truth and predicted bounding boxes based on IoU.

    Args:
        preds: Object detection predictions.
        verbose (bool, optional): Whether to print verbose output. Defaults to False.

    Returns:
        List[List[int]]: Matching indices.
    """

    def assymetric_hausdorff_distance(true_box, pred_box):
        up_distance = pred_box[1] - true_box[1]
        down_distance = true_box[3] - pred_box[3]
        left_distance = pred_box[0] - true_box[0]

        # ... rest of the code ...
        right_distance = true_box[2] - pred_box[2]

        # Return the maximum signed distance
        max_distance = max(up_distance, down_distance, left_distance, right_distance)
        return max_distance

    all_matching = []
    if preds.confidence_threshold is not None:
        conf_thr = preds.confidence_threshold
    else:
        conf_thr = 0
    for pred_boxes, true_boxes, confs in tqdm(
        zip(preds.pred_boxes, preds.true_boxes, preds.confidence), disable=not verbose
    ):
        matching = []
        for true_box in true_boxes:
            ious = []
            for pred_box, conf in zip(pred_boxes, confs):
                if conf < conf_thr:
                    ious.append(-np.inf)
                    continue
                # TODO replace with hausdorff distance ?
                # iou = f_iou(true_box, pred_box)
                # TODO: test
                iou = assymetric_hausdorff_distance(true_box, pred_box)
                iou = iou.cpu().numpy() if isinstance(iou, torch.Tensor) else iou
                ious.append(iou)  # .cpu().numpy())
            matching.append(np.argmin(ious))  # np.argmax(ious))
        all_matching.append(matching)
    preds.matching = all_matching
    return all_matching


def apply_margins(pred_boxes, Qs, mode="additive"):
    n = len(pred_boxes)
    new_boxes = [None] * n
    # print(Qs)
    Qst = torch.FloatTensor([Qs]).cuda()
    for i in range(n):
        # print(pred_boxes[i].shape, Qst.shape)
        if mode == "additive":
            new_boxes[i] = pred_boxes[i] + torch.mul(
                torch.FloatTensor([[-1, -1, 1, 1]]).cuda(), Qst
            )
        elif mode == "multiplicative":
            w = pred_boxes[i][:, 2] - pred_boxes[i][:, 0]
            h = pred_boxes[i][:, 3] - pred_boxes[i][:, 1]
            new_boxes[i] = pred_boxes[i] + torch.mul(
                torch.stack((-w, -h, w, h), dim=-1), Qst
            )
    return new_boxes


def compute_risk_box_level(conf_boxes, true_boxes, loss):
    """
    Input : conformal and true boxes of a all images
    """
    # filter out boxes with low objectness
    losses = []
    for i in range(len(true_boxes)):
        tbs = true_boxes[i]
        cbs = conf_boxes[i]
        for j in range(len(tbs)):
            loss_value = loss(cbs, [tbs[j]])
            losses.append(loss_value)
    losses = torch.stack(losses).ravel()
    return torch.mean(losses)


def compute_risk_image_level(
    conf_boxes,
    true_boxes,
    loss,
):
    losses = torch.zeros(len(true_boxes))
    for i in range(len(true_boxes)):
        tbs = true_boxes[i]
        cbs = conf_boxes[i]
        loss_value = loss(cbs, tbs)
        losses[i] = loss_value
    return torch.mean(losses)
