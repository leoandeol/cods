import logging
from typing import Callable

import numpy as np

from tqdm import tqdm
from skopt import gp_minimize


class Optimizer:
    def optimize(self, objective_function: Callable, alpha: int, **kwargs) -> float:
        raise NotImplementedError("Optimizer is an abstract class")


# Binary search in 1D only
class BinarySearchOptimizer(Optimizer):
    def __init__(self):
        pass

    def optimize(
        self,
        objective_function: Callable,
        alpha: int,
        bounds: list,
        steps: int,
        epsilon=1e-5,
        verbose=True,
    ) -> float:
        """
        params:
        epsilon:
        objective_function: function of one parameter lbd (use partials), which includes the correction part
        """
        if not isinstance(bounds[0], list) and not isinstance(bounds[0], tuple):
            bounds = [bounds]

        lowers = []
        uppers = []
        for bound in bounds:
            lower, upper = bound
            lowers.append(lower)
            uppers.append(upper)
        good_lbds = list([])
        current_lbds = list(
            [(upper - lower) / 2 for lower, upper in zip(lowers, uppers)]
        )

        pbar = tqdm(range(steps), disable=not verbose)

        for step in pbar:
            for id, (lower, upper) in enumerate(zip(lowers, uppers)):
                if upper - lower < epsilon:
                    break
                lbd = (lower + upper) / 2
                current_lbds[id] = lbd
                risk = objective_function(*current_lbds)

                pbar.set_description(
                    f"[{lower:.2f}, {upper:.2f}] -> {current_lbds}. Corrected Risk = {risk:.2f}"
                )

                if risk <= alpha:
                    good_lbds.append(current_lbds.copy())
                if risk <= alpha and risk >= alpha - epsilon:
                    break
                # TODO: find better approach
                if step < steps - 1:
                    if risk <= alpha:
                        upper = lbd
                        uppers[id] = upper
                    elif risk > alpha:
                        lower = lbd
                        lowers[id] = lower
                else:
                    if len(good_lbds) == 0:
                        logging.warning(
                            "No satisfactory solution of binary search found."
                        )
                        return None
        return good_lbds[-1] if len(current_lbds) > 1 else good_lbds[-1][0]


class GaussianProcessOptimizer(Optimizer):
    def __init__(self):
        pass

    def optimize(
        self,
        objective_function: Callable,
        alpha: int,
        bounds: list,
        steps: int,
        epsilon=1e-5,
        verbose=True,
    ) -> float:
        # TODO: experimental
        def fun_opti(params):
            corr_risk = objective_function(*params)
            return (
                5 * (corr_risk - alpha + 1e-3)
                if alpha < corr_risk
                else alpha - corr_risk
            )

        # TODO: put hyperparameters in kwargs
        res = gp_minimize(
            fun_opti,
            (
                [bounds]
                if not isinstance(bounds[0], list) and not isinstance(bounds[0], tuple)
                else bounds
            ),
            n_calls=steps,
            n_random_starts=20,
            random_state=1234,
            verbose=verbose,
        )
        logging.info(f"Ideal parameter after GPR is {res.x}")
        return res.x


class MonteCarloOptimizer(Optimizer):
    def __init__(self):
        pass

    def optimize(
        self,
        objective_function: Callable,
        alpha: int,
        bounds: list,
        steps: int,
        epsilon=1e-4,
        verbose=True,
    ) -> float:
        good_lbds = []
        lbds_risks = []

        pbar = tqdm(range(steps), disable=not verbose)
        for i in pbar:
            lbd = np.random.uniform(size=len(bounds))
            for i, bound in enumerate(bounds):
                lower, upper = bound
                lbd[i] = lower + lbd[i] * (upper - lower)
            corr_risk = objective_function(*lbd)
            if corr_risk <= alpha:
                if corr_risk >= alpha - epsilon:
                    logging.info(
                        "Found satisfactory solution of Monte Carlo. Early Stopping."
                    )
                    return lbd
                good_lbds.append(lbd)
                lbds_risks.append(corr_risk)
                pbar.set_description(f"best_risk = {np.max(lbds_risks)}")
            pbar.update(1)
        if len(good_lbds) == 0:
            logging.warning("No satisfactory solution of Monte Carlo found.")
            return None
        else:
            logging.info(
                f"Found {len(good_lbds)} satisfactory solutions of Monte Carlo. Best risk = {np.max(lbds_risks)} for Lambda={good_lbds[np.argmax(lbds_risks)]}"
            )
            return good_lbds[np.argmax(lbds_risks)]
