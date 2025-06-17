"""Base training and tolerance region utilities for conformal prediction."""

from typing import Callable, Union

import torch
from scipy.optimize import brentq
from scipy.stats import binom

from cods.base.data import Predictions
from cods.base.optim import BinarySearchOptimizer, GaussianProcessOptimizer


def hoeffding(Rhat, n, delta):
    """Hoeffding's inequality bound for binomial proportion confidence intervals.

    Args:
    ----
        Rhat: Empirical risk estimate.
        n: Number of samples.
        delta: Confidence level.

    Returns:
    -------
        Bound on the risk with probability at least 1 - delta.

    """
    return Rhat + torch.sqrt(1 / (2 * n) * torch.log(1 / delta))


def bernstein_emp(Rhat, n, delta):
    """Empirical Bernstein bound for binomial proportion confidence intervals.

    Args:
    ----
        Rhat: Empirical risk estimate.
        n: Number of samples.
        delta: Confidence level.

    Returns:
    -------
        Bound on the risk with probability at least 1 - delta.

    """
    part1 = torch.sqrt((2 * torch.log(2 / delta)) / n)
    part2 = (7 * torch.log(2 / delta)) / (3 * (n - 1))
    sigma = torch.sqrt(Rhat * (1 - Rhat))
    return Rhat + sigma * part1 + part2


def bernstein(Rhat, n, delta):
    """Bernstein's inequality bound for binomial proportion confidence intervals.

    Args:
    ----
        Rhat: Empirical risk estimate.
        n: Number of samples.
        delta: Confidence level.

    Returns:
    -------
        Bound on the risk with probability at least 1 - delta.

    """
    part1 = torch.sqrt((2 * Rhat * torch.log(1 / delta)) / n)
    part2 = 2 * torch.log(1 / delta) / n
    return Rhat + part1 + part2


def bernstein_uni(Rhat, n, delta):
    """Uniform Bernstein bound for binomial proportion confidence intervals.

    Args:
    ----
        Rhat: Empirical risk estimate.
        n: Number of samples.
        delta: Confidence level.

    Returns:
    -------
        Bound on the risk with probability at least 1 - delta.

    """
    part1 = torch.sqrt((2 * Rhat * torch.log(n / delta)) / n)
    part2 = (2 * torch.log(n / delta) / n) + 1.0 / n
    return Rhat + part1 + part2


def bernstein_uni_lim(Rhat, n, delta):
    """Limited uniform Bernstein bound for binomial proportion confidence intervals.

    Args:
    ----
        Rhat: Empirical risk estimate.
        n: Number of samples.
        delta: Confidence level.

    Returns:
    -------
        Bound on the risk with probability at least 1 - delta.

    """
    part1 = torch.sqrt((2 * Rhat * torch.log(n / delta)) / n)
    part2 = (2 * torch.log((1000 * n) / (delta)) / n) + 1.0 / (1000 * n)
    return Rhat + part1 + part2


def binom_inv_cdf(Rhat, n, delta, device="cpu"):
    """Inverse binomial CDF for confidence interval calculation.

    Args:
    ----
        Rhat: Empirical risk estimate.
        n: Number of samples.
        delta: Confidence level.
        device: Torch device string.

    Returns:
    -------
        Upper bound on the risk with probability at least 1 - delta.

    """
    k = int(Rhat * n)

    n = n.detach().cpu().numpy()
    delta = delta.detach().cpu().numpy()

    def f_bin(p):
        return binom.cdf(k, n, p) - delta - 1e-5

    if f_bin(1) > 0:
        return 1.0

    return torch.tensor(
        brentq(f_bin, 1e-10, 1 - 1e-10, maxiter=1000, xtol=1e-4),
        dtype=torch.float,
    ).to(device)


class ToleranceRegion:
    """Abstract base class for tolerance region conformal predictors.

    Provides interface for calibrating, conformalizing, and evaluating predictions using different inequalities and optimizers.
    """

    AVAILABLE_INEQUALITIES = {
        "hoeffding": hoeffding,
        "bernstein_emp": bernstein_emp,
        "bernstein": bernstein,
        "bernstein_uni": bernstein_uni,
        "bernstein_uni_lim": bernstein_uni_lim,
        "binomial_inverse_cdf": binom_inv_cdf,
    }
    ACCEPTED_OPTIMIZERS = {
        "binary_search": BinarySearchOptimizer,
        "gpr": GaussianProcessOptimizer,
        "gaussianprocess": GaussianProcessOptimizer,
        "kriging": GaussianProcessOptimizer,
    }

    def __init__(
        self,
        inequality: Union[str, Callable] = "binomial_inverse_cdf",
        optimizer: str = "binary_search",
        optimizer_args: dict = {},
    ):
        """Initialize a ToleranceRegion instance.

        Args:
        ----
            inequality (Union[str, Callable]): Inequality function or name.
            optimizer (str): Optimizer name.
            optimizer_args (dict): Arguments for the optimizer.

        Raises:
        ------
            ValueError: If the inequality or optimizer is not recognized.

        """
        if isinstance(inequality, str):
            if inequality not in self.AVAILABLE_INEQUALITIES.keys():
                raise ValueError(f"Available inequalities are {self.AVAILABLE_INEQUALITIES.keys()}")
            self.inequality_name = inequality
            self.f_inequality = self.AVAILABLE_INEQUALITIES[inequality]
        elif isinstance(inequality, Callable):
            self.f_inequality = inequality
            self.inequality_name = inequality.__name__
        else:
            raise ValueError(
                f"Available inequalities are {self.AVAILABLE_INEQUALITIES.keys()}",
            )
        if optimizer not in self.ACCEPTED_OPTIMIZERS.keys():
            raise ValueError(
                f"Available optimizers are {self.ACCEPTED_OPTIMIZERS.keys()}",
            )
        self.optimizer_name = optimizer
        self.optimizer = self.ACCEPTED_OPTIMIZERS[optimizer](**optimizer_args)

    def calibrate(
        self,
        preds: Predictions,
        alpha: float = 0.1,
        delta: float = 0.1,
        verbose: bool = True,
        **kwargs,
    ):
        """Calibrate the tolerance region on the given predictions.

        Args:
        ----
            preds (Predictions): Predictions to calibrate on.
            alpha (float): Miscoverage level.
            delta (float): Confidence level.
            verbose (bool): Whether to print detailed output.
            **kwargs: Additional arguments.

        Raises:
        ------
            NotImplementedError: If not implemented in subclass.

        """
        raise NotImplementedError(
            "ToleranceRegion is an abstract class, must be instantiated on a given task.",
        )

    def conformalize(self, preds: Predictions, verbose: bool = True, **kwargs):
        """Conformalize the given predictions.

        Args:
        ----
            preds (Predictions): Predictions to conformalize.
            verbose (bool): Whether to print detailed output.
            **kwargs: Additional arguments.

        Raises:
        ------
            NotImplementedError: If not implemented in subclass.

        """
        raise NotImplementedError(
            "ToleranceRegion is an abstract class, must be instantiated on a given task.",
        )

    def evaluate(self, preds: Predictions, verbose: bool = True, **kwargs):
        """Evaluate the tolerance region on the given predictions.

        Args:
        ----
            preds (Predictions): Predictions to evaluate.
            verbose (bool): Whether to print detailed output.
            **kwargs: Additional arguments.

        Raises:
        ------
            NotImplementedError: If not implemented in subclass.

        """
        raise NotImplementedError(
            "ToleranceRegion is an abstract class, must be instantiated on a given task.",
        )


class CombiningToleranceRegions(ToleranceRegion):
    """Combine multiple tolerance regions, e.g., using Bonferroni correction."""

    def __init__(self, *tregions: ToleranceRegion, mode: str = "bonferroni"):
        """Initialize a CombiningToleranceRegions instance.

        Args:
        ----
            *tregions (ToleranceRegion): Tolerance region instances to combine.
            mode (str): Combination mode (default: 'bonferroni').

        """
        self.tregions = tregions
        self.mode = mode

    def calibrate(
        self,
        preds: Predictions,
        alpha: float = 0.1,
        delta: float = 0.1,
        parameters: Union[None, list] = None,
    ):
        """Calibrate all tolerance regions with appropriate correction.

        Args:
        ----
            preds (Predictions): Predictions to calibrate on.
            alpha (float): Miscoverage level.
            delta (float): Confidence level.
            parameters (Optional[list]): List of parameter dicts for each region.

        Returns:
        -------
            list: Calibration results for each region.

        """
        if parameters is None:
            parameters = [{} for _ in range(len(self.tregions))]
        if self.mode == "bonferroni":
            return list(
                [
                    conformalizer.calibrate(
                        preds,
                        alpha=alpha / len(self.tregions),
                        delta=delta / len(self.tregions),
                        **parameters[i],
                    )
                    for i, conformalizer in enumerate(self.tregions)
                ],
            )

    def conformalize(self, preds):
        """Conformalize predictions using all tolerance regions.

        Args:
        ----
            preds: Predictions to conformalize.

        Returns:
        -------
            list: Conformalized predictions for each region.

        """
        return list([tregion.conformalize(preds) for tregion in self.tregions])

    def evaluate(self, preds, verbose=True):
        """Evaluate all tolerance regions on the given predictions.

        Args:
        ----
            preds: Predictions to evaluate.
            verbose (bool): Whether to print detailed output.

        Returns:
        -------
            list: Evaluation results for each region.

        """
        return list(
            [tregion.evaluate(preds, verbose=verbose) for tregion in self.tregions],
        )
