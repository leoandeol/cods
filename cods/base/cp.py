"""Base module for conformal prediction classes."""


class Conformalizer:
    """Abstract base class for conformal prediction methods."""

    def __init__(self):
        """Initialize the Conformalizer base class."""
        pass

    def calibrate(self, preds, alpha=0.1):
        """Calibrate the conformal predictor on the given predictions.

        Args:
        ----
            preds: Predictions to calibrate on.
            alpha (float): Miscoverage level (default: 0.1).

        Raises:
        ------
            NotImplementedError: If not implemented in subclass.

        """
        raise NotImplementedError(
            "Conformalizer is an abstract class, must be instantiated on a given task.",
        )

    def conformalize(self, preds):
        """Conformalize the given predictions.

        Args:
        ----
            preds: Predictions to conformalize.

        Raises:
        ------
            NotImplementedError: If not implemented in subclass.

        """
        raise NotImplementedError(
            "Conformalizer is an abstract class, must be instantiated on a given task.",
        )

    def evaluate(self, preds, verbose=True):
        """Evaluate the conformal predictor on the given predictions.

        Args:
        ----
            preds: Predictions to evaluate.
            verbose (bool): Whether to print detailed output (default: True).

        Raises:
        ------
            NotImplementedError: If not implemented in subclass.

        """
        raise NotImplementedError(
            "Conformalizer is an abstract class, must be instantiated on a given task.",
        )
