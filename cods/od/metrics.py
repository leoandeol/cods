"""Metrics computation and evaluation for object detection conformal prediction.

This module provides functions to compute various metrics for evaluating
object detection models with conformal prediction, including coverage metrics,
precision-recall computation, and performance visualization tools.
"""

from logging import getLogger
from typing import Callable, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm

from cods.od.data import (
    ODConformalizedPredictions,
    ODParameters,
    ODPredictions,
    ODResults,
)
from cods.od.utils import f_iou

logger = getLogger("cods")


def compute_global_coverage(  # noqa: C901
    predictions: ODPredictions,
    parameters: ODParameters,
    conformalized_predictions: ODConformalizedPredictions,
    guarantee_level: str = "object",
    confidence: bool = True,
    cls: bool = True,
    localization: bool = True,
    loss: Optional[Callable] = None,
) -> torch.Tensor:
    """Compute the global coverage for object detection predictions.

    Args:
    ----
        predictions (ODPredictions): Object detection predictions.
        parameters (ODParameters): Parameters for object detection.
        conformalized_predictions (ODConformalizedPredictions): Conformalized object detection predictions.
        guarantee_level (str, optional): Level of coverage guarantee (e.g., 'object'). Defaults to 'object'.
        confidence (bool, optional): Whether to consider confidence coverage. Defaults to True.
        cls (bool, optional): Whether to consider class coverage. Defaults to True.
        localization (bool, optional): Whether to consider localization coverage. Defaults to True.
        loss (Callable, optional): Loss function to use. Defaults to None.

    Returns:
    -------
        torch.Tensor: Global coverage tensor.

    """
    conf_boxes = conformalized_predictions.conf_boxes
    if conf_boxes is None and (confidence is True or localization is True):
        localization = False
        confidence = False
        logger.warning(
            "No conformal boxes provided, skipping confidence and localization",
        )
    conf_cls = conformalized_predictions.conf_cls
    if conf_cls is None and cls is True:
        cls = False
        logger.warning(
            "No conformal classes provided, skipping classification",
        )

    covs = []
    for i in tqdm.tqdm(range(len(predictions))):
        if confidence:
            conf_loss = (
                0
                if (
                    predictions.confidences[i]
                    >= predictions.confidence_threshold
                ).sum()
                >= len(predictions.true_boxes[i])
                else 1
            )
        else:
            conf_loss = 0
        for j in range(len(predictions.true_cls[i])):
            if cls:
                if (
                    predictions.matching[i] is None
                    or predictions.matching[i][j] is None
                    or len(predictions.matching[i][j]) == 0
                ):
                    cls_loss = 1
                else:
                    true_cls = predictions.true_cls[i][j].item()
                    conf_cls_i_k = conf_cls[i][predictions.matching[i][j][0]]
                    cls_loss = 0 if true_cls in conf_cls_i_k else 1
            else:
                cls_loss = 0
            if localization:
                try:
                    conf_boxes_i = conf_boxes[i][
                        predictions.confidences[i]
                        >= predictions.confidence_threshold
                    ]
                    true_box = predictions.true_boxes[i][j]

                    if (
                        predictions.matching[i] is None
                        or predictions.matching[i][j] is None
                        or len(predictions.matching[i][j]) == 0
                    ):
                        conf_box_i = torch.tensor([])
                    else:
                        conf_box_i = conf_boxes_i[
                            predictions.matching[i][j][0]
                        ]

                    if loss is None:
                        if (
                            true_box[0] >= conf_box_i[0]
                            and true_box[1] >= conf_box_i[1]
                            and true_box[2] <= conf_box_i[2]
                            and true_box[3] <= conf_box_i[3]
                        ):
                            loc_loss = 0
                        else:
                            loc_loss = 1
                    else:
                        # TODO: partly redundant, to be improved
                        conf_box_i = (
                            conf_box_i[None, :]
                            if conf_box_i.shape[0] == 4
                            and len(conf_box_i.shape) == 1
                            else torch.tensor([])
                        )
                        loc_loss = loss(
                            true_box[None, :],
                            None,
                            conf_box_i,
                            None,
                        ).item()
                except Exception as e:
                    print(
                        f"Number of ground truth boxes: {len(predictions.true_boxes[i])}",
                    )
                    print(predictions.pred_boxes[i].shape)
                    print(
                        predictions.pred_boxes[i][
                            predictions.confidences[i]
                            >= predictions.confidence_threshold
                        ].shape,
                    )
                    print(conf_boxes[i].shape)
                    print(conf_boxes_i.shape)
                    print(predictions.matching[i][j][0])
                    print(predictions.matching[i])
                    raise AssertionError() from e

            else:
                loc_loss = 0

            # Formule incorrect = 1 - somme des loss
            # coverage = conf_coverage * cls_coverage * loc_coverage
            # coverage = 1 - (1 - conf_coverage) * (1 - cls_coverage) * (1 - loc_loss)
            coverage = max(max(loc_loss, conf_loss), cls_loss)
            coverage = torch.tensor(coverage, dtype=torch.float)
            covs.append(coverage)
    covs = torch.stack(covs)
    return covs


def getStretch(
    od_predictions: ODPredictions,
    conf_boxes: list,
) -> torch.Tensor:
    """Get the stretch of object detection predictions.

    Args:
    ----
        od_predictions (ODPredictions): Object detection predictions.
        conf_boxes (list): List of confidence boxes.

    Returns:
    -------
        torch.Tensor: Stretch tensor.

    """
    stretches = []

    def area(x):
        return (x[:, 2] - x[:, 0] + 1) * (x[:, 3] - x[:, 1] + 1)

    pred_boxes = od_predictions.pred_boxes
    for i in range(len(pred_boxes)):
        stretches.append(area(conf_boxes[i]) / area(pred_boxes[i]))
    return torch.cat(stretches).mean()


def get_recall_precision(
    od_predictions: ODPredictions,
    IOU_THRESHOLD=0.5,
    SCORE_THRESHOLD=0.5,
    verbose=True,
    replace_iou=None,
) -> tuple:
    """Get the recall and precision for object detection predictions.

    Args:
    ----
        od_predictions (ODPredictions): Object detection predictions.
        IOU_THRESHOLD (float, optional): IoU threshold. Defaults to 0.5.
        SCORE_THRESHOLD (float, optional): Score threshold. Defaults to 0.5.
        verbose (bool, optional): Whether to display progress. Defaults to True.
        replace_iou (function, optional): IoU replacement function. Defaults to None.

    Returns:
    -------
        tuple: Tuple containing the recall, precision, and scores.

    """
    true_boxes = od_predictions.true_boxes
    scores = od_predictions.confidence
    pred_boxes = od_predictions.pred_boxes

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
            f"Average Recall = {np.mean(recalls)}, Average Precision = {np.mean(precisions)}",
        )
    return recalls, precisions, my_scores


def getAveragePrecision(
    od_predictions: ODPredictions,
    verbose=True,
    iou_threshold=0.3,
) -> tuple:
    """Get the average precision for object detection predictions.

    Args:
    ----
        od_predictions (ODPredictions): Object detection predictions.
        verbose (bool, optional): Whether to display progress. Defaults to True.
        iou_threshold (float, optional): IoU threshold. Defaults to 0.3.

    Returns:
    -------
        tuple: Tuple containing the average precision, total recalls, total precisions, and objectness thresholds.

    """
    total_recalls = []
    total_precisions = []
    threshes_objectness = np.linspace(0, 1, 40)
    pbar = tqdm.tqdm(threshes_objectness, disable=not verbose)
    for thresh in pbar:
        tmp_recalls, tmp_precisions, _ = get_recall_precision(
            od_predictions,
            IOU_THRESHOLD=iou_threshold,
            SCORE_THRESHOLD=thresh,
            verbose=False,
        )
        pbar.set_description(
            f"Average Recall = {np.mean(tmp_recalls)}, Average Precision = {np.mean(tmp_precisions)}",
        )
        total_recalls.append(np.mean(tmp_recalls))
        total_precisions.append(np.mean(tmp_precisions))

    AP = np.trapz(
        x=list(reversed(total_recalls)),
        y=list(reversed(total_precisions)),
    )
    return AP, total_recalls, total_precisions, threshes_objectness


def plot_recall_precision(
    total_recalls: list,
    total_precisions: list,
    threshes_objectness: np.ndarray,
):
    """Plot the recall and precision given objectness threshold or IoU threshold.

    Args:
    ----
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
    predictions: ODPredictions,
    conformalized_predictions: ODConformalizedPredictions,
    confidence_threshold: Optional[Union[float, torch.Tensor]] = None,
    iou_threshold: float = 0.5,
    verbose: bool = True,
) -> dict:
    """Compute and return various metrics for object detection predictions and conformalized predictions.

    Args:
    ----
        predictions (ODPredictions): Object detection predictions.
        conformalized_predictions (ODConformalizedPredictions): Conformalized object detection predictions.
        confidence_threshold (float or torch.Tensor, optional): Confidence threshold. Defaults to None.
        iou_threshold (float, optional): IoU threshold. Defaults to 0.5.
        verbose (bool, optional): Whether to display progress. Defaults to True.

    Returns:
    -------
        dict: Dictionary containing metrics such as AP, recalls, precisions, and thresholds.

    """
    # TODO: include conf_cls for metrics
    if confidence_threshold is None:
        print("Defaulting to predictions' confidence threshold")
        confidence_threshold = predictions.confidence_threshold
    else:
        print(f"Using confidence threshold {confidence_threshold}")

    pred_boxes = predictions.pred_boxes

    (
        AP_vanilla,
        total_recalls_vanilla,
        total_precisions_vanilla,
        threshes_objectness_vanilla,
    ) = getAveragePrecision(
        predictions,
        pred_boxes,
        verbose=True,
        iou_threshold=iou_threshold,
    )
    conf_boxes = conformalized_predictions.conf_boxes
    if verbose:
        print(f"Average Precision: {AP_vanilla}")
    (
        AP_conf,
        total_recalls_conf,
        total_precisions_conf,
        threshes_objectness_conf,
    ) = getAveragePrecision(
        predictions,
        conf_boxes,
        verbose=True,
        iou_threshold=iou_threshold,
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


class ODEvaluator:
    """Evaluator for object detection predictions using specified loss functions."""

    def __init__(
        self,
        confidence_loss,
        localization_loss,
        classification_loss,
    ):
        """Initialize the ODEvaluator.

        Args:
        ----
            confidence_loss (callable): Loss function for confidence.
            localization_loss (callable): Loss function for localization.
            classification_loss (callable): Loss function for classification.

        """
        self.confidence_loss = confidence_loss
        self.localization_loss = localization_loss
        self.classification_loss = classification_loss

    def evaluate(  # noqa: C901
        self,
        predictions: ODPredictions,
        parameters: ODParameters,
        conformalized_predictions: ODConformalizedPredictions,
    ):
        """Evaluate predictions using the provided loss functions and return results.

        Args:
        ----
            predictions (ODPredictions): Object detection predictions.
            parameters (ODParameters): Parameters for object detection.
            conformalized_predictions (ODConformalizedPredictions): Conformalized object detection predictions.

        Returns:
        -------
            ODResults: Results object containing computed losses and set sizes.

        """
        # TODO: handle ODParameters
        confidence_losses = []
        classification_losses = []
        localization_losses = []

        confidence_set_sizes = []
        classification_set_sizes = []
        localization_set_sizes = []

        true_boxes = predictions.true_boxes
        true_cls = predictions.true_cls
        confidences = predictions.confidences

        pred_boxes = predictions.pred_boxes
        pred_cls = predictions.pred_cls

        conf_boxes = conformalized_predictions.conf_boxes
        conf_cls = conformalized_predictions.conf_cls

        device = predictions.pred_boxes[0].device
        confidence_threshold = predictions.confidence_threshold
        print(f"Confidence threshold: {confidence_threshold}")
        try:
            # printer parameters
            print("ODParameters")
            print(f"global_alpha: {parameters.global_alpha}")
            print(f"alpha_confidence: {parameters.alpha_confidence}")
            print(f"alpha_localization: {parameters.alpha_localization}")
            print(f"alpha_classification: {parameters.alpha_classification}")
            print(
                f"lambda_confidence_plus: {parameters.lambda_confidence_plus}",
            )
            print(
                f"lambda_confidence_minus: {parameters.lambda_confidence_minus}",
            )
            print(f"lambda_localization: {parameters.lambda_localization}")
            print(f"lambda_classification: {parameters.lambda_classification}")
            print(f"confidence_threshold: {parameters.confidence_threshold}")
        except Exception as e:
            print("Error printing parameters")
            print(e)
            print("Parameters are not printed")

        for i in range(len(predictions)):
            true_boxes_i = true_boxes[i]
            pred_boxes_i = pred_boxes[i]
            conf_boxes_i = conf_boxes[i]
            confidences_i = confidences[i]
            true_cls_i = true_cls[i]
            pred_cls_i = pred_cls[i]
            conf_cls_i = conf_cls[i]

            matching_i = predictions.matching[i]

            conf_boxes_i = conf_boxes_i[confidences_i >= confidence_threshold]
            pred_boxes_i = pred_boxes_i[confidences_i >= confidence_threshold]
            pred_cls_i = pred_cls_i[confidences_i >= confidence_threshold]
            conf_cls_i = [
                x
                for x, c in zip(conf_cls_i, confidences_i)
                if c >= confidence_threshold
            ]

            if self.confidence_loss is not None:
                confidence_loss_i = self.confidence_loss(
                    true_boxes_i,
                    true_cls_i,
                    pred_boxes_i,
                    pred_cls_i,  # conf_boxes_i, conf_cls_i
                )
                confidence_set_size_i = pred_boxes_i.shape[0]

                confidence_losses.append(confidence_loss_i)
                confidence_set_sizes.append(confidence_set_size_i)

            tmp_matched_boxes_i = [
                (
                    torch.stack([conf_boxes_i[m] for m in matching_i[j]])[0]
                    if len(matching_i[j]) > 0
                    else torch.tensor([]).float().to(device)
                )
                for j in range(len(true_boxes_i))
            ]
            matched_conf_boxes_i = (
                torch.stack(tmp_matched_boxes_i)
                if len(tmp_matched_boxes_i) > 0
                else torch.tensor([]).float().to(device)
            )
            matched_conf_cls_i = [
                (
                    torch.stack([conf_cls_i[m] for m in matching_i[j]])[
                        0
                    ]  # TODO zero here ?
                    if len(matching_i[j]) > 0
                    else torch.tensor([]).float().to(device)
                )
                for j in range(len(true_boxes_i))
            ]

            # if matched_conf_boxes_i.size() == 0:
            #     matched_conf_boxes_i = torch.tensor([]).float().to(device)

            if self.localization_loss is not None:
                # try:
                localization_loss_i = self.localization_loss(
                    true_boxes_i,
                    true_cls_i,
                    matched_conf_boxes_i,
                    matched_conf_cls_i,
                )
                # except:
                #     print(len(matched_conf_boxes_i))
                #     print(matched_conf_boxes_i.shape)
                #     print(matched_conf_boxes_i)
                localization_set_size_i = []
                for conf_box_i_j, pred_box_i_j in zip(
                    conf_boxes_i,
                    pred_boxes_i,
                ):
                    set_size = (
                        (conf_box_i_j[2] - conf_box_i_j[0])
                        * (conf_box_i_j[3] - conf_box_i_j[1])
                    ) / (
                        (pred_box_i_j[2] - pred_box_i_j[0])
                        * (pred_box_i_j[3] - pred_box_i_j[1])
                    )
                    set_size = torch.sqrt(set_size)
                    localization_set_size_i.append(set_size)
                if len(localization_set_size_i) == 0:
                    localization_set_size_i = torch.tensor(
                        [0.0],
                        dtype=torch.float,
                    ).to(conf_boxes_i.device)[0]
                else:
                    localization_set_size_i = torch.mean(
                        torch.stack(localization_set_size_i),
                    )

                localization_losses.append(localization_loss_i)
                localization_set_sizes.append(localization_set_size_i)

            if self.classification_loss is not None:
                classification_loss_i = self.classification_loss(
                    true_boxes_i,
                    true_cls_i,
                    matched_conf_boxes_i,
                    matched_conf_cls_i,
                )

                classification_losses.append(classification_loss_i)

                classification_set_size_i = []
                for conf_cls_i_j in conf_cls_i:
                    classification_set_size_i.append(conf_cls_i_j.shape[0])
                if len(classification_set_size_i) == 0:
                    classification_set_size_i = torch.tensor(
                        [0.0],
                        dtype=torch.float,
                    ).to(conf_boxes_i.device)[0]
                else:
                    classification_set_size_i = torch.mean(
                        torch.tensor(
                            classification_set_size_i,
                            dtype=torch.float,
                        ),
                    )
                classification_set_sizes.append(classification_set_size_i)

        if self.localization_loss is not None:
            localization_losses = torch.stack(localization_losses)
        else:
            localization_losses = None
        if self.classification_loss is not None:
            classification_losses = torch.stack(classification_losses)
        else:
            classification_losses = None

        results = ODResults(
            predictions=predictions,
            parameters=parameters,
            conformalized_predictions=conformalized_predictions,
            confidence_coverages=torch.stack(confidence_losses)
            if len(confidence_losses) > 0
            else None,
            classification_coverages=classification_losses,
            localization_coverages=localization_losses,
            confidence_set_sizes=torch.tensor(
                confidence_set_sizes,
                dtype=torch.float,
            )
            if len(confidence_set_sizes) > 0
            else None,
            classification_set_sizes=torch.stack(classification_set_sizes)
            if len(classification_set_sizes) > 0
            else None,
            localization_set_sizes=torch.stack(localization_set_sizes)
            if len(localization_set_sizes) > 0
            else None,
            global_coverage=torch.maximum(
                localization_losses,
                classification_losses,
            )
            if self.localization_loss is not None
            and self.classification_loss is not None
            else (
                localization_losses
                if self.localization_loss is not None
                else (
                    classification_losses
                    if self.classification_loss is not None
                    else None
                )
            ),
        )
        return results
