"""Tolerance region implementation for conformal classification."""

from typing import Callable

import torch

from cods.base.tr import ToleranceRegion
from cods.classif.data import ClassificationPredictions
from cods.classif.loss import CLASSIFICATION_LOSSES, ClassificationLoss


class ClassificationToleranceRegion(ToleranceRegion):
    """Tolerance region for conformal classification tasks."""

    ACCEPTED_PREPROCESS = {"softmax": torch.softmax}

    def __init__(
        self,
        loss="lac",
        inequality="binomial_inverse_cdf",
        optimizer="binary_search",
        preprocess="softmax",
        device="cpu",
        optimizer_args=None,
    ):
        """Initialize the ClassificationToleranceRegion.

        Args:
        ----
            loss (str or ClassificationLoss): Loss function or its name.
            inequality (str): Inequality function name.
            optimizer (str): Optimizer name.
            preprocess (str or Callable): Preprocessing function or its name.
            device (str): Device to use.
            optimizer_args (dict, optional): Arguments for the optimizer.

        """
        if optimizer_args is None:
            optimizer_args = {}
        super().__init__(
            inequality=inequality,
            optimizer=optimizer,
            optimizer_args=optimizer_args,
        )
        self.ACCEPTED_LOSSES = CLASSIFICATION_LOSSES
        self.lbd = None
        if isinstance(preprocess, str):
            if preprocess not in self.ACCEPTED_PREPROCESS.keys():
                raise ValueError(
                    f"preprocess '{preprocess}' not accepted, must be one of {self.ACCEPTED_PREPROCESS.keys()}"
                )
            self.preprocess = preprocess
            self.f_preprocess = self.ACCEPTED_PREPROCESS[preprocess]
        elif isinstance(preprocess, Callable):
            self.preprocess_name = preprocess.__name__
            self.f_preprocess = preprocess
        else:
            raise ValueError(
                f"Loss {loss} not supported. Choose from {self.ACCEPTED_LOSSES}.",
            )
        if preprocess not in self.ACCEPTED_PREPROCESS.keys():
            raise ValueError(
                f"preprocess '{preprocess}' not accepted, must be one of {self.ACCEPTED_PREPROCESS}",
            )
        self.device = device
        self.preprocess = preprocess
        self.f_preprocess = self.ACCEPTED_PREPROCESS[preprocess]
        if isinstance(loss, str):
            if loss not in self.ACCEPTED_LOSSES:
                raise ValueError(f"Loss {loss} not supported. Choose from {self.ACCEPTED_LOSSES}.")
            self.loss_name = loss
            self.loss = self.ACCEPTED_LOSSES[loss]()
        elif isinstance(loss, ClassificationLoss):
            self.loss_name = loss.__class__.__name__
            self.loss = loss
        else:
            raise ValueError(
                f"Loss {loss} not supported. Choose from {self.ACCEPTED_LOSSES}.",
            )
        self.confidence_threshold = None

    def calibrate(
        self,
        predictions: ClassificationPredictions,
        alpha=0.1,
        delta=0.1,
        steps=13,
        bounds=[0, 1],
        verbose=True,
        objectness_threshold=0.8,
    ):
        """Calibrate the tolerance region for conformal classification.

        Args:
        ----
            predictions (ClassificationPredictions): Predictions to calibrate on.
            alpha (float, optional): Miscoverage level. Defaults to 0.1.
            delta (float, optional): Confidence level. Defaults to 0.1.
            steps (int, optional): Number of optimization steps. Defaults to 13.
            bounds (list, optional): Search bounds. Defaults to [0, 1].
            verbose (bool, optional): Whether to print progress. Defaults to True.
            objectness_threshold (float, optional): Objectness threshold. Defaults to 0.8.

        Returns:
        -------
            float: The calibrated lambda value.

        """
        if self.lbd is not None:
            print("Replacing previously computed lambda")
        self._n_classes = predictions.n_classes
        risk_function = self._get_risk_function(
            predictions=predictions,
            alpha=alpha,
            delta=delta,
            objectness_threshold=objectness_threshold,
        )

        lbd = self.optimizer.optimize(
            objective_function=risk_function,
            alpha=alpha,
            bounds=bounds,
            steps=steps,
            verbose=verbose,
        )
        self.lbd = lbd
        self.confidence_threshold = objectness_threshold
        return lbd

    def _get_risk_function(
        self,
        predictions: ClassificationPredictions,
        alpha: float,
        delta: float,
        objectness_threshold: float,
        **kwargs,
    ) -> Callable:
        """Return a risk function for calibration.

        Args:
        ----
            predictions (ClassificationPredictions): Predictions to use.
            alpha (float): Miscoverage level.
            delta (float): Confidence level.
            objectness_threshold (float): Objectness threshold.
            **kwargs: Additional arguments.

        Returns:
        -------
            Callable: Risk function for calibration.

        """
        n = len(predictions.true_cls)

        def risk_function(lbd):
            risk = []
            for i, true_cls in enumerate(predictions.true_cls):
                pred_cls = self.f_preprocess(predictions.pred_cls[i], -1)
                conf_set = self.loss.get_set(pred_cls=pred_cls, lbd=lbd)
                score = self.loss(true_cls=true_cls, conf_cls=conf_set)
                risk.append(score)
            risk = torch.stack(risk)
            risk = torch.mean(risk)
            corrected_risk = self._correct_risk(
                risk=risk,
                n=n,
                delta=delta,
            )
            return corrected_risk

        return risk_function

    def _correct_risk(
        self,
        risk: torch.Tensor,
        n: int,
        delta: float,
    ) -> torch.Tensor:
        """Apply the selected inequality to correct the risk estimate.

        Args:
        ----
            risk (torch.Tensor): Empirical risk.
            n (int): Number of samples.
            delta (float): Confidence level.

        Returns:
        -------
            torch.Tensor: Corrected risk.

        """
        return self.f_inequality(
            Rhat=risk,
            n=torch.tensor(n, dtype=torch.float).to(self.device),
            delta=torch.tensor(delta, dtype=torch.float).to(self.device),
        )

    def conformalize(
        self,
        predictions: ClassificationPredictions,
        verbose: bool = True,
        **kwargs,
    ) -> list:
        """Conformalize the predictions using the calibrated lambda.

        Args:
        ----
            predictions (ClassificationPredictions): Predictions to conformalize.
            verbose (bool, optional): Whether to print progress. Defaults to True.
            **kwargs: Additional arguments.

        Returns:
        -------
            list: List of conformalized prediction sets.

        """
        if self.lbd is None:
            raise ValueError(
                "Conformalizer must be calibrated before conformalizing.",
            )
        conf_cls = []
        for pred_cls in predictions.pred_cls:
            pred_cls = self.f_preprocess(pred_cls, -1)
            ys = self.loss.get_set(pred_cls=pred_cls, lbd=self.lbd)
            conf_cls.append(ys)
        return conf_cls

    def evaluate(
        self,
        preds: ClassificationPredictions,
        conf_cls: list,
        verbose=True,
        **kwargs,
    ):
        """Evaluate the conformalized predictions.

        Args:
        ----
            preds (ClassificationPredictions): Predictions to evaluate.
            conf_cls (list): Conformalized prediction sets.
            verbose (bool, optional): Whether to print progress. Defaults to True.
            **kwargs: Additional arguments.

        Returns:
        -------
            tuple: Tuple of (coverage, average set size).

        """
        if self.lbd is None:
            raise ValueError(
                "Conformalizer must be calibrated before evaluating.",
            )
        losses = []
        set_sizes = []
        for i, true_cls in enumerate(preds.true_cls):
            conf_cls_i = conf_cls[i]
            loss = torch.tensor(float(true_cls in conf_cls_i))
            losses.append(loss)
            set_size = torch.tensor(float(len(conf_cls_i)))
            set_sizes.append(set_size)
        losses = torch.stack(losses)
        set_sizes = torch.stack(set_sizes)
        if verbose:
            print(
                f"Coverage: {torch.mean(losses)}, Avg. set size: {torch.mean(set_sizes)}",
            )
        return losses, set_sizes
