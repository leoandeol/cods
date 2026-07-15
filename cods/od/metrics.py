from collections.abc import Callable
from logging import getLogger

import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm
from scipy.optimize import linear_sum_assignment
from torchvision.ops import box_iou

from cods.od.data import (
    ODConformalizedPredictions,
    ODParameters,
    ODPredictions,
    ODResults,
)
from cods.od.utils import f_iou

logger = getLogger("cods")


def compute_box_level_metrics(
    predictions: ODPredictions,
    confidence_threshold: float | torch.Tensor | None = None,
    iou_threshold: float = 0.5,
    verbose: bool = True,
) -> dict:
    """Box-level TP/FP/FN counts and recall/precision, over all images pooled.

    For each image, an injective matching pi: ground truth -> pred is computed by
    maximizing the sum of IoU(gt, pred) over pairs (Hungarian assignment); some true
    boxes stay unmatched when there are fewer predictions than ground truths.
    Then, at box level:
    - a predicted box is a TP if it is associated to a true box with the same class
      and IoU >= `iou_threshold`; it is a FP otherwise,
    - a true box is a FN iff it has no associated prediction, or its associated
      prediction has the wrong class or IoU < `iou_threshold`.

    Both aggregation options are returned:
    - Option 1, all classes pooled: recall = TP/(TP+FN) (denominator = total number
      of true boxes), precision = TP/(TP+FP).
    - Option 2, per class then averaged: TP_k/FP_k are restricted to predictions of
      class k and FN_k to true boxes of class k; macro averages are taken over
      classes with a nonzero denominator (classes with >= 1 true box for recall,
      classes with >= 1 prediction for precision).

    Predictions are first filtered by `confidence_threshold` (defaults to
    `predictions.confidence_threshold`, or 0 if unset). The predicted class of a box
    is the argmax of its class scores.
    """
    if confidence_threshold is None:
        confidence_threshold = (
            predictions.confidence_threshold
            if predictions.confidence_threshold is not None
            else 0.0
        )
    if isinstance(confidence_threshold, torch.Tensor):
        confidence_threshold = confidence_threshold.item()

    n_classes = predictions.n_classes
    tp_per_class = torch.zeros(n_classes)  # indexed by the (common) true = pred class
    fp_per_class = torch.zeros(n_classes)  # indexed by the predicted class
    fn_per_class = torch.zeros(n_classes)  # indexed by the true class

    for i in range(len(predictions)):
        true_boxes = predictions.true_boxes[i]
        true_cls = predictions.true_cls[i]
        mask = predictions.confidences[i] >= confidence_threshold
        pred_boxes = predictions.pred_boxes[i][mask]
        n_gt, n_pred = len(true_boxes), len(pred_boxes)
        pred_labels = predictions.pred_cls[i][mask].argmax(dim=-1) if n_pred > 0 else torch.zeros(0)

        # Injective matching gt -> pred maximizing the sum of IoU
        if n_gt > 0 and n_pred > 0:
            iou = box_iou(true_boxes.float(), pred_boxes.float())  # (n_gt, n_pred)
            gt_idx, pred_idx = linear_sum_assignment(-iou.cpu().numpy())
        else:
            iou = None
            gt_idx, pred_idx = (), ()

        is_tp_pred = np.zeros(n_pred, dtype=bool)
        gt_is_covered = np.zeros(n_gt, dtype=bool)
        for g, p in zip(gt_idx, pred_idx):
            if iou[g, p].item() >= iou_threshold and int(pred_labels[p]) == int(true_cls[g]):
                is_tp_pred[p] = True
                gt_is_covered[g] = True
                tp_per_class[int(true_cls[g])] += 1
        for p in range(n_pred):
            if not is_tp_pred[p]:
                fp_per_class[int(pred_labels[p])] += 1
        for g in range(n_gt):
            if not gt_is_covered[g]:
                fn_per_class[int(true_cls[g])] += 1

    tp = tp_per_class.sum().item()
    fp = fp_per_class.sum().item()
    fn = fn_per_class.sum().item()

    # Sanity check (option 1): TP + FN = total number of true boxes
    n_true_boxes = sum(len(tb) for tb in predictions.true_boxes)
    if tp + fn != n_true_boxes:
        logger.warning(
            f"Box-level sanity check failed: TP + FN = {tp + fn} != {n_true_boxes} true boxes",
        )

    # Option 1: all classes pooled
    recall = tp / (tp + fn) if tp + fn > 0 else float("nan")
    precision = tp / (tp + fp) if tp + fp > 0 else float("nan")

    # Option 2: per class, then macro-averaged over classes with a nonzero denominator
    recall_denom = tp_per_class + fn_per_class  # = number of true boxes of each class
    precision_denom = tp_per_class + fp_per_class  # = number of predictions of each class
    recall_per_class = torch.where(
        recall_denom > 0,
        tp_per_class / recall_denom,
        torch.full_like(recall_denom, float("nan")),
    )
    precision_per_class = torch.where(
        precision_denom > 0,
        tp_per_class / precision_denom,
        torch.full_like(precision_denom, float("nan")),
    )
    macro_recall = (
        recall_per_class[recall_denom > 0].mean().item()
        if (recall_denom > 0).any()
        else float("nan")
    )
    macro_precision = (
        precision_per_class[precision_denom > 0].mean().item()
        if (precision_denom > 0).any()
        else float("nan")
    )

    if verbose:
        logger.info(
            f"Box-level metrics (IoU >= {iou_threshold}, conf >= {confidence_threshold}): "
            f"TP={tp:.0f}, FP={fp:.0f}, FN={fn:.0f} | "
            f"recall={recall:.4f}, precision={precision:.4f} | "
            f"macro_recall={macro_recall:.4f}, macro_precision={macro_precision:.4f}",
        )

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "recall": recall,
        "precision": precision,
        "tp_per_class": tp_per_class,
        "fp_per_class": fp_per_class,
        "fn_per_class": fn_per_class,
        "recall_per_class": recall_per_class,
        "precision_per_class": precision_per_class,
        "macro_recall": macro_recall,
        "macro_precision": macro_precision,
    }


def compute_global_coverage(
    predictions: ODPredictions,
    parameters: ODParameters,
    conformalized_predictions: ODConformalizedPredictions,
    guarantee_level: str = "object",
    confidence: bool = True,
    cls: bool = True,
    localization: bool = True,
    loss: Callable | None = None,
) -> torch.Tensor:
    """Compute the global coverage for object detection predictions. BOXWISE/IMAGEWISE #TODO

    Args:
    ----
        predictions (ODPredictions): Object detection predictions.
        conformalized_predictiond (ODConformalizedPredictions): Conformalized object detection predictions.
        confidence (bool, optional): Whether to consider confidence coverage. Defaults to True.
        cls (bool, optional): Whether to consider class coverage. Defaults to True.
        localization (bool, optional): Whether to consider localization coverage. Defaults to True.
        loss (function, optional): Loss function. Defaults to None.

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
                if (predictions.confidences[i] >= predictions.confidence_threshold).sum()
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
                        predictions.confidences[i] >= predictions.confidence_threshold
                    ]
                    true_box = predictions.true_boxes[i][j]

                    if (
                        predictions.matching[i] is None
                        or predictions.matching[i][j] is None
                        or len(predictions.matching[i][j]) == 0
                    ):
                        conf_box_i = torch.tensor([])
                    else:
                        conf_box_i = conf_boxes_i[predictions.matching[i][j][0]]

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
                            if conf_box_i.shape[0] == 4 and len(conf_box_i.shape) == 1
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
                            predictions.confidences[i] >= predictions.confidence_threshold
                        ].shape,
                    )
                    print(conf_boxes[i].shape)
                    print(conf_boxes_i.shape)
                    print(predictions.matching[i][j][0])
                    print(predictions.matching[i])
                    print(e)

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


def _resolve_confidence_threshold(predictions, confidence_threshold):
    if confidence_threshold is None:
        confidence_threshold = (
            predictions.confidence_threshold
            if predictions.confidence_threshold is not None
            else 0.0
        )
    if isinstance(confidence_threshold, torch.Tensor):
        confidence_threshold = confidence_threshold.item()
    return confidence_threshold


def compute_precision_recall_lrp(
    predictions: ODPredictions,
    conformalized_predictions: ODConformalizedPredictions | None = None,
    confidence_threshold: float | torch.Tensor | None = None,
    iou_threshold: float = 0.5,
    error_loc: str = "iou",
    tp_selection: str = "confidence",
    verbose: bool = True,
) -> dict:
    """Class-independent Precision, Recall and LRP with prediction -> ground-truth
    matching (spec: "OD metrics: Precision, Recall, LRP, OCE", Section 2).

    For each kept prediction (confidence >= `confidence_threshold`), an arrow is
    drawn to the ground truth with highest IoU among those with IoU >= `iou_threshold`
    and a class match. Then, per ground truth: no incoming arrow -> FN; otherwise the
    claimant with the highest confidence (`tp_selection="confidence"`, variant 1) or
    highest IoU (`tp_selection="iou"`, variant 2) becomes the single TP (its
    localization error is recorded) and the other claimants lose their arrow.
    Predictions left without an arrow are FPs.

    Raw mode (`conformalized_predictions=None`): boxes are `pred_boxes` and the class
    match is top-1 predicted class == true class. Conformal mode: boxes are the
    margin-corrected `conf_boxes` and the class match is true class in the conformal
    label set `conf_cls` (spec item 2).

    `error_loc`:
    - "iou": errorloc = (1 - IoU) / (1 - iou_threshold), in [0, 1] for a TP;
    - "hausdorff": the CODS asymmetric signed Hausdorff distance, clamped to >= 0 and
      normalized by the image diagonal (the [0, 1]-normalization is not fixed by the
      spec; the diagonal makes it image-size independent).

    Returns a dict with tp / fp / fn counts, precision = TP/(TP+FP),
    recall = TP/(TP+FN), loc = sum of errorloc over TPs, and
    lrp = (loc + FP + FN) / (TP + FP + FN).
    """
    if error_loc not in ("iou", "hausdorff"):
        raise ValueError(f"error_loc {error_loc} not accepted, must be 'iou' or 'hausdorff'")
    if tp_selection not in ("confidence", "iou"):
        raise ValueError(
            f"tp_selection {tp_selection} not accepted, must be 'confidence' or 'iou'",
        )
    confidence_threshold = _resolve_confidence_threshold(predictions, confidence_threshold)
    conf_boxes_all = conformalized_predictions.conf_boxes if conformalized_predictions else None
    conf_cls_all = conformalized_predictions.conf_cls if conformalized_predictions else None

    tp, fp, fn = 0, 0, 0
    loc_error_sum = 0.0

    for i in range(len(predictions)):
        true_boxes = predictions.true_boxes[i]
        true_cls = predictions.true_cls[i]
        mask = predictions.confidences[i] >= confidence_threshold
        confidences = predictions.confidences[i][mask]
        if conf_boxes_all is not None:
            boxes = conf_boxes_all[i][mask]
        else:
            boxes = predictions.pred_boxes[i][mask]
        n_gt, n_pred = len(true_boxes), len(boxes)

        if conf_cls_all is not None:
            kept_cls_sets = [s for s, keep in zip(conf_cls_all[i], mask) if keep]

            def class_match(k, j, kept_cls_sets=kept_cls_sets, true_cls=true_cls):
                return bool(torch.isin(true_cls[j], kept_cls_sets[k]).item())
        else:
            pred_labels = (
                predictions.pred_cls[i][mask].argmax(dim=-1) if n_pred > 0 else torch.zeros(0)
            )

            def class_match(k, j, pred_labels=pred_labels, true_cls=true_cls):
                return int(pred_labels[k]) == int(true_cls[j])

        if n_gt > 0 and n_pred > 0:
            iou = box_iou(true_boxes.float(), boxes.float())  # (n_gt, n_pred)
        else:
            iou = torch.zeros((n_gt, n_pred))

        # Arrows prediction -> ground truth: best-IoU gt among those with
        # IoU >= threshold and a class match.
        arrows = [None] * n_pred
        for k in range(n_pred):
            best_j, best_iou = None, -1.0
            for j in range(n_gt):
                iou_jk = iou[j, k].item()
                if iou_jk >= iou_threshold and iou_jk > best_iou and class_match(k, j):
                    best_j, best_iou = j, iou_jk
            arrows[k] = best_j

        # Per ground truth: FN if unclaimed, otherwise a single TP among claimants.
        for j in range(n_gt):
            claimants = [k for k in range(n_pred) if arrows[k] == j]
            if len(claimants) == 0:
                fn += 1
                continue
            if tp_selection == "confidence":
                winner = max(claimants, key=lambda k: confidences[k].item())
            else:
                winner = max(claimants, key=lambda k: iou[j, k].item())
            for k in claimants:
                if k != winner:
                    arrows[k] = None
            tp += 1
            if error_loc == "iou":
                loc_error_sum += (1.0 - iou[j, winner].item()) / (1.0 - iou_threshold)
            else:
                from cods.od.utils import assymetric_hausdorff_distance

                d = assymetric_hausdorff_distance(
                    true_boxes[j][None, :],
                    boxes[winner][None, :],
                )[0, 0].item()
                shape = predictions.image_shapes[i]
                diag = float(np.sqrt(float(shape[0]) ** 2 + float(shape[1]) ** 2))
                loc_error_sum += min(max(d, 0.0) / diag, 1.0)

        fp += sum(1 for k in range(n_pred) if arrows[k] is None)

    n_true_boxes = sum(len(tb) for tb in predictions.true_boxes)
    if tp + fn != n_true_boxes:
        logger.warning(
            f"Sanity check failed: TP + FN = {tp + fn} != {n_true_boxes} true boxes",
        )

    precision = tp / (tp + fp) if tp + fp > 0 else float("nan")
    recall = tp / (tp + fn) if tp + fn > 0 else float("nan")
    lrp = (loc_error_sum + fp + fn) / (tp + fp + fn) if tp + fp + fn > 0 else float("nan")

    if verbose:
        logger.info(
            f"P/R/LRP (IoU >= {iou_threshold}, conf >= {confidence_threshold}, "
            f"errorloc={error_loc}, tp_selection={tp_selection}): "
            f"TP={tp}, FP={fp}, FN={fn} | precision={precision:.4f}, recall={recall:.4f}, "
            f"LOC={loc_error_sum:.4f}, LRP={lrp:.4f}",
        )

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "loc": loc_error_sum,
        "lrp": lrp,
    }


def compute_oce(
    predictions: ODPredictions,
    conformalized_predictions: ODConformalizedPredictions | None = None,
    confidence_threshold: float | torch.Tensor | None = None,
    iou_thresholds: tuple = (0.5, 0.75),
    verbose: bool = True,
) -> dict:
    """OCE metric with ground-truth -> predictions matching (spec Section 3).

    For each ground truth (i, j) and IoU threshold tau, Q = set of kept predictions
    whose box has IoU(box, gt) >= tau. The averaged foreground score vector
    p_bar(c) = mean over Q of the model's class scores (0 if Q is empty), and
    Brier_tau = sum over the K foreground classes of (1{c = true class} - p_bar(c))^2.
    OCE_tau is the mean of Brier_tau over all ground truths of the dataset, and
    OCE = sum of OCE_tau over `iou_thresholds` (default OCE_0.5 + OCE_0.75).

    The class scores must be the foreground restriction of a softmax over K+1 classes
    (K foreground + background). This is what CODS stores for DETR:
    `pred_cls = softmax(logits)[..., :-1]` (the background column is dropped, so rows
    sum to < 1 — the background class makes the softmax well defined but is not
    scored, as required).

    In conformal mode (`conformalized_predictions` given), Q is computed with the
    margin-corrected `conf_boxes` (spec item 2); the scores p_bar stay the model's
    softmax scores, since OCE evaluates probabilities, not label sets.
    """
    confidence_threshold = _resolve_confidence_threshold(predictions, confidence_threshold)
    conf_boxes_all = conformalized_predictions.conf_boxes if conformalized_predictions else None

    brier_sums = dict.fromkeys(iou_thresholds, 0.0)
    n_gt_total = 0

    for i in range(len(predictions)):
        true_boxes = predictions.true_boxes[i]
        true_cls = predictions.true_cls[i]
        n_gt = len(true_boxes)
        if n_gt == 0:
            continue
        n_gt_total += n_gt

        mask = predictions.confidences[i] >= confidence_threshold
        if conf_boxes_all is not None:
            boxes = conf_boxes_all[i][mask]
        else:
            boxes = predictions.pred_boxes[i][mask]
        scores = predictions.pred_cls[i][mask]  # (n_pred, K) foreground softmax scores
        n_pred = len(boxes)

        if n_pred > 0:
            iou = box_iou(true_boxes.float(), boxes.float())  # (n_gt, n_pred)
        for tau in iou_thresholds:
            for j in range(n_gt):
                if n_pred > 0:
                    in_q = iou[j] >= tau
                else:
                    in_q = torch.zeros(0, dtype=torch.bool)
                if in_q.any():
                    p_bar = scores[in_q].mean(dim=0)
                else:
                    p_bar = torch.zeros(predictions.n_classes)
                onehot = torch.zeros_like(p_bar)
                onehot[int(true_cls[j])] = 1.0
                brier_sums[tau] += torch.sum((onehot - p_bar) ** 2).item()

    result = {}
    for tau in iou_thresholds:
        result[f"OCE_{tau}"] = brier_sums[tau] / n_gt_total if n_gt_total > 0 else float("nan")
    result["OCE"] = sum(result[f"OCE_{tau}"] for tau in iou_thresholds)

    if verbose:
        parts = ", ".join(f"OCE_{tau}={result[f'OCE_{tau}']:.4f}" for tau in iou_thresholds)
        logger.info(f"OCE (conf >= {confidence_threshold}): {parts}, OCE={result['OCE']:.4f}")

    return result


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
        pred_boxes (list): List of predicted boxes. Defaults to None.
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
        pred_boxes (list): List of predicted boxes.
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
    _, (ax1, ax2) = plt.subplots(1, 2)
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
    confidence_threshold: float | torch.Tensor | None = None,
    iou_threshold: float = 0.5,
    verbose: bool = True,
) -> dict:
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
    def __init__(
        self,
        confidence_loss,
        localization_loss,
        classification_loss,
    ):
        self.confidence_loss = confidence_loss
        self.localization_loss = localization_loss
        self.classification_loss = classification_loss

    def evaluate(
        self,
        predictions: ODPredictions,
        parameters: ODParameters,
        conformalized_predictions: ODConformalizedPredictions,
    ):
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
            conf_cls_i = [x for x, c in zip(conf_cls_i, confidences_i) if c >= confidence_threshold]

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
                    torch.stack([conf_cls_i[m] for m in matching_i[j]])[0]  # TODO zero here ?
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
                        (conf_box_i_j[2] - conf_box_i_j[0]) * (conf_box_i_j[3] - conf_box_i_j[1])
                    ) / ((pred_box_i_j[2] - pred_box_i_j[0]) * (pred_box_i_j[3] - pred_box_i_j[1]))
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
            if self.localization_loss is not None and self.classification_loss is not None
            else (
                localization_losses
                if self.localization_loss is not None
                else (classification_losses if self.classification_loss is not None else None)
            ),
        )
        return results
