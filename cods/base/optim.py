"""Optimizers for conformal prediction and related search procedures."""

import logging
from typing import Callable, List, Tuple, Union

import numpy as np
from skopt import gp_minimize
from tqdm import tqdm


class Optimizer:
    """Abstract base class for optimizers used in conformal prediction calibration."""

    def optimize(
        self,
        objective_function: Callable,
        alpha: float,
        **kwargs,
    ) -> float:
        """Optimize the objective function to satisfy the risk constraint.

        Args:
        ----
            objective_function (Callable): The function to optimize.
            alpha (float): The risk threshold.
            **kwargs: Additional arguments for the optimizer.

        Returns:
        -------
            float: The optimal parameter value.

        Raises:
        ------
            NotImplementedError: If not implemented in subclass.

        """
        raise NotImplementedError("Optimizer is an abstract class")


class BinarySearchOptimizer(Optimizer):
    """Optimizer using binary search in 1D (or multi-D) for risk calibration."""

    def __init__(self):
        """Initialize the BinarySearchOptimizer."""
        pass

    def optimize(  # noqa: C901
        self,
        objective_function: Callable,
        alpha: float,
        bounds: Union[Tuple, List, List[Tuple]],
        steps: int,
        epsilon=1e-5,
        verbose=True,
    ) -> float:
        """Perform binary search to find the optimal parameter.

        Args:
        ----
            objective_function (Callable): Function of one parameter lbd (use partials), which includes the correction part.
            alpha (float): Risk threshold.
            bounds (Union[Tuple, List, List[Tuple]]): Search bounds.
            steps (int): Number of search steps.
            epsilon (float): Precision threshold.
            verbose (bool): Whether to print progress.

        Returns:
        -------
            float: The optimal parameter value.

        """
        if not isinstance(bounds[0], list) and not isinstance(
            bounds[0],
            tuple,
        ):
            bounds = [bounds]

        lowers = []
        uppers = []
        for bound in bounds:
            lower, upper = bound
            lowers.append(lower)
            uppers.append(upper)
        good_lbds = []
        current_lbds = [
            (upper - lower) / 2 for lower, upper in zip(lowers, uppers)
        ]

        pbar = tqdm(range(steps), disable=not verbose)

        for step in pbar:
            for id, (lower, upper) in enumerate(zip(lowers, uppers)):
                # Stop based on lambda
                if upper - lower < epsilon:
                    break
                lbd = (lower + upper) / 2
                current_lbds[id] = lbd

                risk = objective_function(*current_lbds)

                pbar.set_description(
                    f"[{lower:.2f}, {upper:.2f}] -> {current_lbds}. Corrected Risk = {risk:.2f}",
                )

                if risk <= alpha:
                    good_lbds.append(current_lbds.copy())
                    if risk >= alpha - epsilon:
                        break
                # TODO: find better approach
                if step < steps - 1:
                    if risk <= alpha:
                        upper = lbd
                        uppers[id] = upper
                    elif risk > alpha:
                        lower = lbd
                        lowers[id] = lower
                elif len(good_lbds) == 0:
                    logging.error(
                        "No satisfactory solution of binary search found.",
                    )
                    return None
        return good_lbds[-1] if len(current_lbds) > 1 else good_lbds[-1][0]


class GaussianProcessOptimizer(Optimizer):
    """Optimizer using Gaussian Process Regression (Bayesian optimization) for risk calibration."""

    def __init__(self):
        """Initialize the GaussianProcessOptimizer."""
        pass

    def optimize(
        self,
        objective_function: Callable,
        alpha: float,
        bounds: Union[Tuple, List, List[Tuple]],
        steps: int,
        epsilon=1e-5,
        verbose=True,
    ) -> Union[float, np.ndarray, None]:
        """Optimize using Gaussian Process Regression (Bayesian optimization).

        Args:
        ----
            objective_function (Callable): The function to optimize.
            alpha (float): The risk threshold.
            bounds (Union[Tuple, List, List[Tuple]]): Search bounds.
            steps (int): Number of optimization steps.
            epsilon (float): Precision threshold.
            verbose (bool): Whether to print progress.

        Returns:
        -------
            float or np.ndarray or None: The optimal parameter value, or None if not found.

        """

        def fun_opti(params):
            corr_risk = objective_function(*params)
            return (
                5 * (corr_risk - alpha + 1e-3)
                if alpha < corr_risk
                else alpha - corr_risk
            )

        res = gp_minimize(
            fun_opti,
            (
                [bounds]
                if not isinstance(bounds[0], list)
                and not isinstance(bounds[0], tuple)
                else bounds
            ),
            n_calls=steps,
            n_random_starts=20,
            random_state=1234,
            verbose=verbose,
        )
        if res is None:
            logging.info("No satisfactory solution of GPR found.")
            return None
        logging.info(f"Ideal parameter after GPR is {res.x}")
        return res.x


class MonteCarloOptimizer(Optimizer):
    """Optimizer using Monte Carlo random search for risk calibration."""

    def __init__(self):
        """Initialize the MonteCarloOptimizer."""
        pass

    def optimize(
        self,
        objective_function: Callable,
        alpha: float,
        bounds: Union[Tuple, List[Tuple]],
        steps: int,
        epsilon=1e-4,
        verbose=True,
    ) -> Union[float, np.ndarray, None]:
        """Optimize using Monte Carlo random search.

        Args:
        ----
            objective_function (Callable): The function to optimize.
            alpha (float): The risk threshold.
            bounds (Union[Tuple, List[Tuple]]): Search bounds.
            steps (int): Number of optimization steps.
            epsilon (float): Precision threshold.
            verbose (bool): Whether to print progress.

        Returns:
        -------
            float or np.ndarray or None: The optimal parameter value, or None if not found.

        """
        good_lbds = []
        lbds_risks = []

        pbar = tqdm(range(steps), disable=not verbose)
        for _it in pbar:
            lbd = np.random.uniform(size=len(bounds))
            for i, bound in enumerate(bounds):
                lower, upper = bound
                lbd[i] = lower + lbd[i] * (upper - lower)
            corr_risk = objective_function(*lbd)
            if corr_risk <= alpha:
                if corr_risk >= alpha - epsilon:
                    logging.info(
                        "Found satisfactory solution of Monte Carlo. Early Stopping.",
                    )
                    return lbd
                good_lbds.append(lbd)
                lbds_risks.append(corr_risk)
                pbar.set_description(f"best_risk = {np.max(lbds_risks)}")
            pbar.update(1)
        if len(good_lbds) == 0:
            logging.warning("No satisfactory solution of Monte Carlo found.")
            return None
        logging.info(
            f"Found {len(good_lbds)} satisfactory solutions of Monte Carlo. Best risk = {np.max(lbds_risks)} for Lambda={good_lbds[np.argmax(lbds_risks)]}",
        )
        return good_lbds[np.argmax(lbds_risks)]
