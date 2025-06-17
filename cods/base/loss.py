"""Loss functions and non-conformity scores for conformal prediction."""

import torch

from cods.base.data import ConformalizedPredictions, Predictions


class NCScore:
    """Abstract base class for non-conformity scores in conformal prediction."""

    def __init__(self):
        """Initialize the NCScore base class."""
        pass


class Loss:
    """Abstract base class for loss functions in conformal prediction."""

    def __init__(self):
        """Initialize the Loss base class."""
        pass

    def __call__(
        self,
        predictions: Predictions,
        conformalized_predictions: ConformalizedPredictions,
        **kwargs,
    ) -> torch.Tensor:
        """Compute the loss between predictions and conformalized predictions.

        Args:
        ----
            predictions (Predictions): The original predictions.
            conformalized_predictions (ConformalizedPredictions): The conformalized predictions.
            **kwargs: Additional arguments.

        Returns:
        -------
            torch.Tensor: The computed loss.

        Raises:
        ------
            NotImplementedError: If not implemented in subclass.

        """
        raise NotImplementedError
