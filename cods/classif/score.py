"""Non-conformity scores for conformal classification."""

import torch

from cods.base.loss import NCScore


class ClassifNCScore(NCScore):
    """Abstract base class for classification non-conformity score functions."""

    def __init__(self, n_classes, **kwargs):
        """Initialize the ClassifNCScore base class.

        Args:
        ----
            n_classes (int): Number of classes.
            **kwargs: Additional arguments.

        """
        super().__init__()
        self.n_classes = n_classes

    def __call__(self, pred_cls: torch.Tensor, y: torch.Tensor, **kwargs):
        """Compute the non-conformity score for a given prediction and label.

        Args:
        ----
            pred_cls (torch.Tensor): Predicted class probabilities.
            y (torch.Tensor): True label.
            **kwargs: Additional arguments.

        Returns:
        -------
            float: Non-conformity score.

        Raises:
        ------
            NotImplementedError: If not implemented in subclass.

        """
        raise NotImplementedError("ClassifNCScore is an abstract class.")

    def get_set(self, pred_cls, quantile):
        """Get the conformal prediction set for a given quantile.

        Args:
        ----
            pred_cls (torch.Tensor): Predicted class probabilities.
            quantile (float): Quantile threshold.

        Returns:
        -------
            torch.Tensor: Indices of the conformal prediction set.

        """
        ys = list(range(self.n_classes))
        pred_set = list(
            [y for y in ys if self._score_function(pred_cls, y) <= quantile],
        )
        pred_set = torch.tensor(pred_set)
        return pred_set


class LACNCScore(ClassifNCScore):
    """Non-conformity score for Least Ambiguous Conformal (LAC) prediction sets."""

    def __init__(self, n_classes: int, **kwargs):
        """Initialize the LACNCScore class.

        Args:
        ----
            n_classes (int): Number of classes.
            **kwargs: Additional arguments.

        """
        super().__init__(n_classes=n_classes)

    def __call__(self, pred_cls: torch.Tensor, y: torch.Tensor, **kwargs):
        """Compute the LAC non-conformity score.

        Args:
        ----
            pred_cls (torch.Tensor): Predicted class probabilities.
            y (torch.Tensor): True label.
            **kwargs: Additional arguments.

        Returns:
        -------
            float: LAC non-conformity score.

        """
        return 1 - pred_cls[y]

    def get_set(self, pred_cls, quantile):
        """Get the conformal prediction set for a given quantile.

        Args:
        ----
            pred_cls (torch.Tensor): Predicted class probabilities.
            quantile (float): Quantile threshold.

        Returns:
        -------
            torch.Tensor: Indices of the conformal prediction set.

        """
        return torch.where(pred_cls >= 1 - quantile)[0]


class APSNCScore(ClassifNCScore):
    """Non-conformity score for Adaptive Prediction Sets (APS)."""

    def __init__(self, n_classes, **kwargs):
        """Initialize the APSNCScore class.

        Args:
        ----
            n_classes (int): Number of classes.
            **kwargs: Additional arguments.

        """
        super().__init__(n_classes=n_classes)

    def __call__(self, pred_cls: torch.Tensor, y: int, **kwargs):
        """Compute the APS non-conformity score.

        Args:
        ----
            pred_cls (torch.Tensor): Predicted class probabilities.
            y (int): True label.
            **kwargs: Additional arguments.

        Returns:
        -------
            float: APS non-conformity score.

        Raises:
        ------
            ValueError: If input is not a probability vector or label not found.

        """
        # Ensure input is a probability distribution
        if not torch.isclose(pred_cls.sum(), torch.tensor(1.0), atol=1e-3):
            raise ValueError(
                f"Input pred_cls should be a probability vector, but sums to {pred_cls.sum().item()}",
            )

        values, indices = torch.sort(pred_cls, descending=True)
        cumsum = torch.cumsum(values, dim=0)
        match = torch.where(indices == y)[0]
        if len(match) == 0:
            raise ValueError(f"Label {y} not found in indices.")

        # Randomized nonconformity score for calibration
        u = torch.rand(1).item() * values[match[0]]
        score = cumsum[match[0]] - values[match[0]] + u
        return max(score.item(), 0.0)

    def get_set(
        self,
        pred_cls: torch.Tensor,
        quantile: float,
    ):
        """Get the conformal prediction set for a given quantile.

        Args:
        ----
            pred_cls (torch.Tensor): Predicted class probabilities.
            quantile (float): Quantile threshold.

        Returns:
        -------
            torch.Tensor: Indices of the conformal prediction set.

        """
        # TODO: tmp fix
        pred_cls /= pred_cls.sum()
        # In many cases the probabilities don't sum to 1
        # This case is handled in the else : defaults to the full prediction set.
        values, indices = torch.sort(pred_cls, descending=True)
        cumsum = torch.cumsum(values, dim=0)
        idxs = torch.where(cumsum > quantile)[0]
        if len(idxs) > 0:
            k = idxs[0]
        else:
            k = len(cumsum) - 1
        return indices[: k + 1]
