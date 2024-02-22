# Classification metrics

import numpy as np
import torch
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    auc,
)
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import classification_report
from sklearn.metrics import log_loss
from sklearn.metrics import brier_score_loss
from sklearn.metrics import hamming_loss
from sklearn.metrics import zero_one_loss
from sklearn.metrics import jaccard_score
from sklearn.metrics import hinge_loss
from sklearn.metrics import fbeta_score
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import precision_recall_curve

from cods.classif.data import ClassificationPredictions
from cods.classif.loss import ClassificationLoss, CLASSIFICATION_LOSSES


def get_coverage(
    preds: ClassificationPredictions, conf_cls: torch.Tensor, verbose: bool = True
):
    """
    Computes the coverage of the conformal prediction set.
    :param preds: predictions and ground truth of the classifier
    :param conf_cls: conformalized predictions of the classifier
    :param verbose: whether to print the coverage
    :return: coverage of the conformal prediction set
    """
    cov = torch.zeros(len(preds))
    for i in range(len(preds)):
        if torch.isin(preds.true_cls[i], conf_cls[i]):
            cov[i] = 1
    if verbose:
        print("Coverage: ", cov.mean().item())
    return cov


# same as get_coverage but  instead with a more general loss instead of the inclusion of the true label
def get_empirical_risk(
    preds: ClassificationPredictions,
    conf_cls: torch.Tensor,
    loss: ClassificationLoss,
    verbose: bool = True,
):
    """
    Computes the empirical risk of the conformal prediction set.
    :param preds: predictions and ground truth of the classifier
    :param conf_cls: conformalized predictions of the classifier
    :param verbose: whether to print the empirical risk
    :return: empirical risk of the conformal prediction set
    """
    emp_risk = torch.zeros(len(preds))
    for i in range(len(preds)):
        emp_risk[i] = loss(preds.true_cls[i], conf_cls[i])
    if verbose:
        print("Empirical risk: ", emp_risk.mean().item())
    return emp_risk


# 1 - empirical_risk if risk is bounded
def get_empirical_safety(
    preds: ClassificationPredictions,
    conf_cls: torch.Tensor,
    loss: ClassificationLoss,
    verbose: bool = True,
):
    """
    Computes the empirical safety of the conformal prediction set.
    :param preds: predictions and ground truth of the classifier
    :param conf_cls: conformalized predictions of the classifier
    :param verbose: whether to print the empirical safety
    :return: empirical safety of the conformal prediction set
    """
    ACCEPTED_LOSSES = CLASSIFICATION_LOSSES
    if isinstance(loss, str):
        loss = ACCEPTED_LOSSES[loss]()
    elif isinstance(loss, ClassificationLoss):
        loss = loss()
    else:
        raise ValueError(
            f"loss must be a string or a ClassificationLoss instance, got {loss}"
        )

    B = loss.upper_bound
    safety = B - get_empirical_risk(preds, conf_cls, loss, verbose=False)
    if verbose:
        print("Empirical safety: ", safety.mean().item())
    return safety
