from logging import getLogger
from typing import Any, Callable, Optional, Union

logger = getLogger("cods")

import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm
from numba import jit

from cods.od.data import ODConformalizedPredictions, ODPredictions
from cods.od.utils import f_iou


def compute_global_coverage(
    predictions: ODPredictions,
    conformalized_predictiond: ODConformalizedPredictions,
    confidence: bool = True,
    cls: bool = True,
    localization: bool = True,
    loss: Optional[Callable] = None,
) -> torch.Tensor:
    """
    Compute the global coverage for object detection predictions. BOXWISE/IMAGEWISE #TODO

    Args:
        predictions (ODPredictions): Object detection predictions.
        conformalized_predictiond (ODConformalizedPredictions): Conformalized object detection predictions.
        confidence (bool, optional): Whether to consider confidence coverage. Defaults to True.
        cls (bool, optional): Whether to consider class coverage. Defaults to True.
        localization (bool, optional): Whether to consider localization coverage. Defaults to True.
        loss (function, optional): Loss function. Defaults to None.

    Returns:
        torch.Tensor: Global coverage tensor.
    """

    conf_boxes = conformalized_predictiond.conf_boxes
    if conf_boxes is None and (confidence is True or localization is True):
        localization = False
        confidence = False
        logger.warning(
            "No conformal boxes provided, skipping confidence and localization"
        )
    conf_cls = conformalized_predictiond.conf_cls
    if conf_cls is None and cls is True:
        cls = False
        logger.warning(
            "No conformal classes provided, skipping classification"
        )

    covs = []
    for i in tqdm.tqdm(range(len(predictions))):
        if confidence:
            conf_loss = (
                0
                if (
                    predictions.confidence[i]
                    >= predictions.confidence_threshold
                ).sum()
                >= len(predictions.true_boxes[i])
                else 1
            )
        else:
            conf_loss = 0
        for j in range(len(predictions.true_cls[i])):
            if cls:
                true_cls = predictions.true_cls[i][j].item()
                conf_cls_i_k = conf_cls[i][j]
                cls_loss = 0 if true_cls in conf_cls_i_k else 1
            else:
                cls_loss = 0
            if localization:
                # conf_boxes_i = [
                #     box
                #     for k, box in enumerate(conf_boxes[i])
                #     if predictions.confidence[i][k] >= predictions.confidence_threshold
                # ]
                # Tensor style
                conf_boxes_i = conf_boxes[i][
                    predictions.confidence[i]
                    >= predictions.confidence_threshold
                ]
                if loss is None:
                    true_box = predictions.true_boxes[i][j]
                    loc_loss = 1
                    for conf_box in conf_boxes_i:
                        if (
                            true_box[0] >= conf_box[0]
                            and true_box[1] >= conf_box[1]
                            and true_box[2] <= conf_box[2]
                            and true_box[3] <= conf_box[3]
                        ):
                            loc_loss = 0
                            break
                else:
                    loc_loss = loss(
                        conf_boxes_i, [predictions.true_boxes[i][j]]
                    ).item()
            else:
                loc_loss = 0

            # Formule incorrect = 1 - somme des loss
            # coverage = conf_coverage * cls_coverage * loc_coverage
            # coverage = 1 - (1 - conf_coverage) * (1 - cls_coverage) * (1 - loc_loss)
            coverage = 1 - max(max(loc_loss, conf_loss), cls_loss)
            coverage = torch.tensor(coverage, dtype=torch.float)
            covs.append(coverage)
    covs = torch.stack(covs)
    return covs


def getStretch(
    od_predictions: ODPredictions, conf_boxes: list
) -> torch.Tensor:
    """
    Get the stretch of object detection predictions.

    Args:
        od_predictions (ODPredictions): Object detection predictions.
        conf_boxes (list): List of confidence boxes.

    Returns:
        torch.Tensor: Stretch tensor.
    """
    stretches = []
    area = lambda x: (x[:, 2] - x[:, 0] + 1) * (x[:, 3] - x[:, 1] + 1)
    pred_boxes = od_predictions.pred_boxes
    for i in range(len(pred_boxes)):
        stretches.append(area(conf_boxes[i]) / area(pred_boxes[i]))
    return torch.cat(stretches).mean()


def get_recall_precision(
    od_predictions: ODPredictions,
    pred_boxes,  # =None,
    IOU_THRESHOLD=0.5,
    SCORE_THRESHOLD=0.5,
    verbose=True,
    replace_iou=None,
) -> tuple:
    """
    Get the recall and precision for object detection predictions.

    Args:
        od_predictions (ODPredictions): Object detection predictions.
        pred_boxes (list): List of predicted boxes. Defaults to None.
        IOU_THRESHOLD (float, optional): IoU threshold. Defaults to 0.5.
        SCORE_THRESHOLD (float, optional): Score threshold. Defaults to 0.5.
        verbose (bool, optional): Whether to display progress. Defaults to True.
        replace_iou (function, optional): IoU replacement function. Defaults to None.

    Returns:
        tuple: Tuple containing the recall, precision, and scores.
    """
    true_boxes = od_predictions.true_boxes
    scores = od_predictions.confidence

    recalls = []
    precisions = []
    my_scores = []
    for i in tqdm.tqdm(range(len(od_predictions)), disable=not verbose):
        tbs = true_boxes[i]
        pbs = pred_boxes[i]
        batch_scores = scores[i]
        pbs = pbs[batch_scores >= SCORE_THRESHOLD]

        already_assigned = []

        tp = 0
        for tb in tbs:
            my_score = 0
            for k, pb in enumerate(pbs):
                if k in already_assigned:
                    continue
                if replace_iou is not None:
                    iou = replace_iou(tb, pb.detach().cpu().numpy())
                else:
                    iou = f_iou(tb, pb.detach().cpu().numpy())
                if iou > IOU_THRESHOLD:
                    already_assigned.append(k)
                    tp += 1
                    my_score = iou
                    my_scores.append(my_score)
                    break

        nb_predictions = len(pbs)
        nb_true = len(tbs)
        recall = tp / nb_true if nb_true > 0 else 1
        if nb_predictions > 0:
            precision = tp / nb_predictions
        else:
            precision = 1

        recalls.append(recall)
        precisions.append(precision)

    if verbose:
        print(
            f"Average Recall = {np.mean(recalls)}, Average Precision = {np.mean(precisions)}"
        )
    return recalls, precisions, my_scores


def getAveragePrecision(
    od_predictions: ODPredictions,
    pred_boxes,
    verbose=True,
    iou_threshold=0.3,
) -> tuple:
    """
    Get the average precision for object detection predictions.

    Args:
        od_predictions (ODPredictions): Object detection predictions.
        pred_boxes (list): List of predicted boxes.
        verbose (bool, optional): Whether to display progress. Defaults to True.
        iou_threshold (float, optional): IoU threshold. Defaults to 0.3.

    Returns:
        tuple: Tuple containing the average precision, total recalls, total precisions, and objectness thresholds.
    """
    total_recalls = []
    total_precisions = []
    threshes_objectness = np.linspace(0, 1, 40)
    pbar = tqdm.tqdm(threshes_objectness, disable=not verbose)
    for thresh in pbar:
        tmp_recalls, tmp_precisions, _ = get_recall_precision(
            od_predictions,
            pred_boxes,
            IOU_THRESHOLD=iou_threshold,
            SCORE_THRESHOLD=thresh,
            verbose=False,
        )
        pbar.set_description(
            f"Average Recall = {np.mean(tmp_recalls)}, Average Precision = {np.mean(tmp_precisions)}"
        )
        total_recalls.append(np.mean(tmp_recalls))
        total_precisions.append(np.mean(tmp_precisions))

    AP = np.trapz(
        x=list(reversed(total_recalls)), y=list(reversed(total_precisions))
    )
    return AP, total_recalls, total_precisions, threshes_objectness


def plot_recall_precision(
    total_recalls: list,
    total_precisions: list,
    threshes_objectness: np.ndarray,
):
    """
    Plot the recall and precision given objectness threshold or IoU threshold.

    Args:
        total_recalls (list): List of total recalls.
        total_precisions (list): List of total precisions.
        threshes_objectness (np.ndarray): Array of objectness thresholds.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.plot(threshes_objectness, total_recalls, label="Recall")
    ax1.plot(threshes_objectness, total_precisions, label="Precision")
    ax1.xlabel("Objectness score threshold")
    ax2.plot(total_recalls, total_precisions)
    ax2.xlabel("Recall")
    ax2.ylabel("Precision")
    plt.legend()
    plt.show()


def unroll_metrics(
    od_predictions: ODPredictions,
    conf_boxes: list[Any],
    conf_cls: list[Any],
    confidence_threshold: Optional[Union[float, torch.Tensor]] = None,
    iou_threshold: float = 0.5,
    verbose: bool = True,
    # ) -> dict:
):
    """
    Unroll the metrics for object detection predictions.

    Args:
        od_predictions (ODPredictions): Object detection predictions.
        conf_boxes (list): List of confidence boxes.
        conf_cls (list): List of confidence classes.
        confidence_threshold (float, optional): Confidence threshold. Defaults to None.
        iou_threshold (float, optional): IoU threshold. Defaults to 0.5.
        verbose (bool, optional): Whether to display progress. Defaults to True.

    Returns:
        dict: Dictionary containing the metrics.
    """
    # TODO: include conf_cls for metrics
    if confidence_threshold is None:
        print("Defaulting to predictions' confidence threshold")
        confidence_threshold = od_predictions.confidence_threshold
    else:
        print(f"Using confidence threshold {confidence_threshold}")

    pred_boxes = od_predictions.pred_boxes

    (
        AP_vanilla,
        total_recalls_vanilla,
        total_precisions_vanilla,
        threshes_objectness_vanilla,
    ) = getAveragePrecision(
        od_predictions, pred_boxes, verbose=True, iou_threshold=iou_threshold
    )
    if verbose:
        print(f"Average Precision: {AP_vanilla}")
    (
        AP_conf,
        total_recalls_conf,
        total_precisions_conf,
        threshes_objectness_conf,
    ) = getAveragePrecision(
        od_predictions, conf_boxes, verbose=True, iou_threshold=iou_threshold
    )
    if verbose:
        print(f"(Conformal) Average Precision: {AP_conf}")

    return {
        "AP_vanilla": AP_vanilla,
        "total_recalls_vanilla": total_recalls_vanilla,
        "total_precisions_vanilla": total_precisions_vanilla,
        "threshes_objectness_vanilla": threshes_objectness_vanilla,
        "AP_conf": AP_conf,
        "total_recalls_conf": total_recalls_conf,
        "total_precisions_conf": total_precisions_conf,
        "threshes_objectness_conf": threshes_objectness_conf,
    }
