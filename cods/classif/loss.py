"""Loss functions for conformal classification."""

import torch

from cods.base.loss import Loss


class ClassificationLoss(Loss):
    """Abstract base class for classification loss functions."""

    def __init__(self, upper_bound: float = 1, **kwargs):
        """Initialize the ClassificationLoss base class.

        Args:
        ----
            upper_bound (float, optional): Upper bound for the loss. Defaults to 1.
            **kwargs: Additional arguments.

        """
        super().__init__()
        self.upper_bound = upper_bound

    def __call__(
        self,
        true_cls: torch.Tensor,
        conf_cls: torch.Tensor,
        **kwargs,
    ):
        """Compute the classification loss.

        Args:
        ----
            true_cls (torch.Tensor): True class of the sample.
            conf_cls (torch.Tensor): Conformalized prediction of the sample.
            **kwargs: Additional arguments.

        Returns:
        -------
            torch.Tensor: Computed loss value.

        Raises:
        ------
            NotImplementedError: If not implemented in subclass.

        """
        raise NotImplementedError("ClassifLoss is an abstract class.")

    def get_set(self, pred_cls: torch.Tensor, lbd: float):
        """Get the conformal prediction set for a given threshold.

        Args:
        ----
            pred_cls (torch.Tensor): Predicted class probabilities.
            lbd (float): Threshold parameter.

        Returns:
        -------
            torch.Tensor: Indices of the conformal prediction set.

        Raises:
        ------
            NotImplementedError: If not implemented in subclass.

        """
        raise NotImplementedError("ClassifLoss is an abstract class.")


class LACLoss(ClassificationLoss):
    """Loss function for Least Ambiguous Conformal (LAC) prediction sets."""

    def __init__(self, upper_bound: float = 1, **kwargs):
        """Initialize the LACLoss class.

        Args:
        ----
            upper_bound (float, optional): Upper bound for the loss. Defaults to 1.
            **kwargs: Additional arguments.

        """
        super().__init__(upper_bound=upper_bound)

    def __call__(self, true_cls: torch.Tensor, conf_cls: torch.Tensor):
        """Compute the LAC loss.

        Args:
        ----
            true_cls (torch.Tensor): True class of the sample.
            conf_cls (torch.Tensor): Conformalized prediction of the sample.

        Returns:
        -------
            torch.Tensor: LAC loss value.

        """
        return 1 - torch.isin(true_cls, conf_cls).float()

    def get_set(self, pred_cls: torch.Tensor, lbd: float):
        """Get the conformal prediction set for a given threshold.

        Args:
        ----
            pred_cls (torch.Tensor): Predicted class probabilities.
            lbd (float): Threshold parameter.

        Returns:
        -------
            torch.Tensor: Indices of the conformal prediction set.

        """
        return torch.where(pred_cls >= 1 - lbd)[0]


CLASSIFICATION_LOSSES = {"lac": LACLoss}
