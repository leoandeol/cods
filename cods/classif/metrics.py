"""Metrics for evaluating conformal classification predictions."""

import torch

from cods.classif.data import ClassificationPredictions
from cods.classif.loss import CLASSIFICATION_LOSSES, ClassificationLoss


def get_coverage(
    preds: ClassificationPredictions,
    conf_cls: torch.Tensor,
    verbose: bool = True,
):
    """Compute the coverage of the conformal prediction set.

    Args:
    ----
        preds (ClassificationPredictions): Predictions and ground truth of the classifier.
        conf_cls (torch.Tensor): Conformalized predictions of the classifier.
        verbose (bool, optional): Whether to print the coverage. Defaults to True.

    Returns:
    -------
        torch.Tensor: Coverage indicator for each sample.

    """
    cov = torch.zeros(len(preds))
    for i in range(len(preds)):
        if torch.isin(preds.true_cls[i], conf_cls[i]):
            cov[i] = 1
    if verbose:
        print("Coverage: ", cov.mean().item())
    return cov


def get_empirical_risk(
    preds: ClassificationPredictions,
    conf_cls: torch.Tensor,
    loss: ClassificationLoss,
    verbose: bool = True,
):
    """Compute the empirical risk of the conformal prediction set.

    Args:
    ----
        preds (ClassificationPredictions): Predictions and ground truth of the classifier.
        conf_cls (torch.Tensor): Conformalized predictions of the classifier.
        loss (ClassificationLoss): Loss function to use.
        verbose (bool, optional): Whether to print the empirical risk. Defaults to True.

    Returns:
    -------
        torch.Tensor: Empirical risk for each sample.

    """
    emp_risk = torch.zeros(len(preds))
    for i in range(len(preds)):
        emp_risk[i] = loss(preds.true_cls[i], conf_cls[i])
    if verbose:
        print("Empirical risk: ", emp_risk.mean().item())
    return emp_risk


def get_empirical_safety(
    preds: ClassificationPredictions,
    conf_cls: torch.Tensor,
    loss: ClassificationLoss,
    verbose: bool = True,
):
    """Compute the empirical safety of the conformal prediction set.

    Args:
    ----
        preds (ClassificationPredictions): Predictions and ground truth of the classifier.
        conf_cls (torch.Tensor): Conformalized predictions of the classifier.
        loss (ClassificationLoss): Loss function to use.
        verbose (bool, optional): Whether to print the empirical safety. Defaults to True.

    Returns:
    -------
        torch.Tensor: Empirical safety for each sample.

    """
    ACCEPTED_LOSSES = CLASSIFICATION_LOSSES
    if isinstance(loss, str):
        loss = ACCEPTED_LOSSES[loss]()
    elif isinstance(loss, ClassificationLoss):
        loss = loss
    else:
        raise ValueError(
            f"loss must be a string or a ClassificationLoss instance, got {loss}",
        )

    B = loss.upper_bound
    safety = B - get_empirical_risk(preds, conf_cls, loss, verbose=False)
    if verbose:
        print("Empirical safety: ", safety.mean().item())
    return safety
