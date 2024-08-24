import logging
from concurrent.futures import thread
from typing import Any, Callable, List, Optional, Sequence, Tuple, Union

from cods.classif.data.predictions import ClassificationPredictions

logger = logging.getLogger("cods")
FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
logging.basicConfig(format=FORMAT)

import torch
from sympy import N
from tqdm import tqdm

from cods.base.cp import Conformalizer
from cods.base.optim import (
    BinarySearchOptimizer,
    GaussianProcessOptimizer,
    MonteCarloOptimizer,
    Optimizer,
)
from cods.classif.cp import ClassificationConformalizer
from cods.od.data import (
    ODConformalizedPredictions,
    ODParameters,
    ODPredictions,
    ODResults,
)
from cods.od.loss import (
    BoxWiseRecallLoss,
    # ClassBoxWiseRecallLoss,
    ClassificationLossWrapper,
    ConfidenceLoss,
    LACLoss,
    # MaximumLoss,
    ODLoss,
    PixelWiseRecallLoss,
)
from cods.od.metrics import compute_global_coverage
from cods.od.score import (
    MinAdditiveSignedAssymetricHausdorffNCScore,
    MinMultiplicativeSignedAssymetricHausdorffNCScore,
    ObjectnessNCScore,
    UnionAdditiveSignedAssymetricHausdorffNCScore,
    UnionMultiplicativeSignedAssymetricHausdorffNCScore,
)
from cods.od.utils import (
    apply_margins,
    # compute_risk_cls_box_level,
    # compute_risk_cls_image_level,
    compute_risk_image_level,
    compute_risk_object_level,
    # evaluate_cls_conformalizer,
    # get_classif_predictions_from_od_predictions,
    # get_conf_cls_for_od,
    match_predictions_to_true_boxes,
)

"""
TODO:
* rewrite the matching to be more efficient
* check nms is done correctly and add bayesod
"""

################ BASIC BRICS #####################


class LocalizationConformalizer(Conformalizer):
    """A class for performing localization conformalization. Should be used within an ODConformalizer.

    Attributes
    ----------
    - BACKENDS (list): A list of supported backends.
    - accepted_methods (dict): A dictionary mapping accepted method names to their corresponding score functions.
    - PREDICTION_SETS (list): A list of supported prediction sets.
    - LOSSES (dict): A dictionary mapping loss names to their corresponding loss classes.
    - OPTIMIZERS (dict): A dictionary mapping optimizer names to their corresponding optimizer classes.

    Methods
    -------
    - __init__: Initialize the LocalizationConformalizer class.
    - _get_risk_function: Get the risk function for risk conformalization.

    """

    BACKENDS = ["auto", "cp", "crc"]

    accepted_methods = {
        "min-hausdorff-additive": MinAdditiveSignedAssymetricHausdorffNCScore,
        "min-hausdorff-multiplicative": MinMultiplicativeSignedAssymetricHausdorffNCScore,
        "union-hausdorff-additive": UnionAdditiveSignedAssymetricHausdorffNCScore,
        "union-hausdorff-multiplicative": UnionMultiplicativeSignedAssymetricHausdorffNCScore,
    }

    PREDICTION_SETS = ["additive", "multiplicative", "adaptive"]
    LOSSES = {
        "pixelwise": PixelWiseRecallLoss,
        "boxwise": BoxWiseRecallLoss,
        "thresholded": None,
    }
    OPTIMIZERS = {
        "binary_search": BinarySearchOptimizer,
        "gaussian_process": GaussianProcessOptimizer,
    }
    GUARANTEE_LEVELS = ["image", "object"]

    def __init__(
        self,
        loss: Union[str, ODLoss],
        prediction_set: str,
        guarantee_level: str,
        number_of_margins: int = 1,  # where to compute 1, 2 or 4 margins with bonferroni corrections
        optimizer: Optional[Union[str, Optimizer]] = None,
        backend: str = "auto",
        **kwargs,
    ):
        """Initialize the CP class.

        Parameters
        ----------
        - loss (Union[str, ODLoss]): The loss function to be used. It can be either a string representing a predefined loss function or an instance of the ODLoss class.
        - prediction_set (str): The prediction set to be used. Must be one of ["additive", "multiplicative", "adaptive"].
        - guarantee_level (str): The guarantee level to be used. Must be one of ["image", "object"].
        - number_of_margins (int, optional): The number of margins to compute. Default is 1.
        - optimizer (Optional[Union[str, Optimizer]], optional): The optimizer to be used. It can be either a string representing a predefined optimizer or an instance of the Optimizer class. Default is None.
        - backend (str, optional): The backend to be used. Default is "auto".
        - **kwargs: Additional keyword arguments.

        Raises
        ------
        - ValueError: If the loss is not accepted, it must be one of the predefined losses or an instance of ODLoss.
        - ValueError: If the prediction set is not accepted, it must be one of the predefined prediction sets.
        - ValueError: If the number of margins is not 1, 2, or 4.
        - NotImplementedError: If the number of margins is greater than 1 (only 1 margin is supported for now).
        - ValueError: If the backend is not accepted, it must be one of the predefined backends.
        - ValueError: If the optimizer is not accepted, it must be one of the predefined optimizers or an instance of Optimizer.

        """
        super().__init__()
        if isinstance(loss, str) and loss in self.LOSSES:
            self.loss = self.LOSSES[loss]()
            self.loss_name = loss
        elif isinstance(loss, ODLoss):
            self.loss = loss
            self.loss_name = loss.__class__.__name__
        else:
            raise ValueError(
                f"loss {loss} not accepted, must be one of {self.LOSSES.keys()} or an instance of ODLoss",
            )

        if prediction_set not in self.PREDICTION_SETS:
            raise ValueError(
                f"prediction_set {prediction_set} not accepted, must be one of {self.PREDICTION_SETS}",
            )
        self.prediction_set = prediction_set

        if guarantee_level not in self.GUARANTEE_LEVELS:
            raise ValueError(
                f"guarantee_level {guarantee_level} not accepted, must be one of {self.GUARANTEE_LEVELS}",
            )
        self.guarantee_level = guarantee_level

        if self.guarantee_level == "object":
            self.risk_function = compute_risk_object_level
        elif self.guarantee_level == "image":
            self.risk_function = compute_risk_image_level

        if number_of_margins not in [1, 2, 4]:
            raise ValueError("number_of_margins must be 1, 2 or 4")
        if number_of_margins > 1:
            raise NotImplementedError("Only 1 margin is supported for now")
        self.number_of_margins = number_of_margins

        if backend not in self.BACKENDS:
            raise ValueError(
                f"backend {backend} not accepted, must be one of {self.BACKENDS}",
            )
        if backend == "auto":
            self.backend = "crc"
            logger.info("Defaulting to CRC backend")
            # TODO: make it adaptive based on the loss (binary or not)
        if backend == "cp":
            raise NotImplementedError("CP backend is not supported yet")
        self.backend = backend

        if optimizer is None and self.backend == "crc":
            self.optimizer = BinarySearchOptimizer()
            logger.warning(
                "Defaulting to BinarySearchOptimizer for CRC backend",
            )
        elif isinstance(optimizer, str) and optimizer in self.OPTIMIZERS:
            self.optimizer = self.OPTIMIZERS[optimizer]()
        elif isinstance(optimizer, Optimizer):
            self.optimizer = optimizer
        else:
            raise ValueError(
                f"optimizer {optimizer} not accepted, must be one of {self.OPTIMIZERS.keys()} or an instance of Optimizer",
            )

    # def calibrate(
    #     self,
    #     predictions: ODPredictions,
    #     alpha: float = 0.1,
    #     confidence_threshold: Union[float, None] = None,
    #     verbose: bool = True,
    # ) -> list:
    #     """
    #     Calibrates the conformalizer using the given predictions.

    #     Args:
    #         predictions (ODPredictions): The object detection predictions.
    #         alpha (float): The significance level for the calibration.
    #         confidence_threshold (float, optional): The confidence threshold for the predictions. If not provided, it must be set in the predictions or in the conformalizer.
    #         verbose (bool): Whether to display progress information.

    #     Returns:
    #         list: The computed quantiles for each margin.
    #     """
    #     if self._score_function is None:
    #         self._score_function = self.accepted_methods[self.method]()
    #     if predictions.confidence_threshold is None:
    #         if confidence_threshold is None:
    #             raise ValueError(
    #                 "confidence_threshold must be set in the predictions or in the conformalizer"
    #             )
    #         else:
    #             predictions.confidence_threshold = confidence_threshold
    #     self.confidence_threshold = predictions.confidence_threshold

    #     if self.scores is None:
    #         # compute all non-conformity scores for each four axes
    #         scores = []
    #         for i, true_box_img in tqdm(
    #             enumerate(predictions.true_boxes), disable=not verbose
    #         ):
    #             for j, true_box in enumerate(true_box_img):
    #                 confidences = predictions.confidence[i]
    #                 pred_boxes = (
    #                     predictions.pred_boxes[i][confidences >= predictions.confidence_threshold]
    #                     if len(
    #                         predictions.pred_boxes[i][
    #                             confidences >= predictions.confidence_threshold
    #                         ]
    #                     )
    #                     > 0
    #                     else predictions.pred_boxes[i][confidences.argmax()]
    #                 )
    #                 score = self._score_function(pred_boxes, true_box)
    #                 scores.append(score)
    #         scores = torch.stack(scores).squeeze()
    #         self.scores = scores
    #         n = len(scores)
    #     else:
    #         scores = torch.clone(self.scores.detach())
    #         n = len(scores)

    #     # 1 margin: take max over all four scores
    #     if self.margins == 1:
    #         scores_1, _ = torch.max(scores, dim=-1)
    #         q = torch.quantile(
    #             scores_1,
    #             (1 - alpha) * (n + 1) / n,
    #             interpolation="higher",
    #         )
    #         quantiles = [q] * 4
    #     # 2 margins: take maximum on x [0, 2] and y [1, 3] axes
    #     elif self.margins == 2:
    #         scores_1, _ = torch.max(scores[:, [0, 2]], dim=-1)
    #         scores_2, _ = torch.max(scores[:, [1, 3]], dim=-1)
    #         # must apply statistical correction (bonferroni correction)
    #         q1 = torch.quantile(
    #             scores_1,
    #             (1 - alpha / 2) * (n + 1) / n,
    #             interpolation="higher",
    #         )
    #         q2 = torch.quantile(
    #             scores_2,
    #             (1 - alpha / 2) * (n + 1) / n,
    #             interpolation="higher",
    #         )
    #         quantiles = [q1, q2, q1, q2]
    #     # 4 margins: take quantile on each axis
    #     elif self.margins == 4:
    #         q1 = torch.quantile(
    #             scores[:, 0],
    #             (1 - alpha / 4) * (n + 1) / n,
    #             interpolation="higher",
    #         )
    #         q2 = torch.quantile(
    #             scores[:, 1],
    #             (1 - alpha / 4) * (n + 1) / n,
    #             interpolation="higher",
    #         )
    #         q3 = torch.quantile(
    #             scores[:, 2],
    #             (1 - alpha / 4) * (n + 1) / n,
    #             interpolation="higher",
    #         )
    #         q4 = torch.quantile(
    #             scores[:, 3],
    #             (1 - alpha / 4) * (n + 1) / n,
    #             interpolation="higher",
    #         )
    #         quantiles = [q1, q2, q3, q4]

    #     self.quantiles = quantiles
    #     return quantiles

    # TODO: clean it, only crc backend, just create a new optimizer when we have binary losses "QuantileOptimizer"

    def _get_objective_function(
        self,
        predictions: ODPredictions,
        alpha: float,
        confidence_threshold: float,
        **kwargs,
    ) -> Callable[[float], torch.Tensor]:
        """TODO: Add docstring"""
        pred_boxes_filtered = list(
            [
                (
                    x[y >= confidence_threshold]
                    # rip to my little trick
                    # if len(x[y >= confidence_threshold]) > 0
                    # else x[None, y.argmax()]
                )
                for x, y in zip(predictions.pred_boxes, predictions.confidence)
            ],
        )

        def objective_function(lbd: float) -> torch.Tensor:
            """Compute the risk given a lambda value.

            Parameters
            ----------
            lbd (float): The lambda value.

            Returns
            -------
            corrected_risk (float): The corrected risk.

            """
            conf_boxes = apply_margins(
                pred_boxes_filtered,
                [lbd, lbd, lbd, lbd],
                mode=self.prediction_set,
            )
            tmp_parameters = ODParameters(
                global_alpha=alpha,
                confidence_threshold=confidence_threshold,
                predictions_id=predictions.unique_id,
            )
            tmp_conformalized_predictions = ODConformalizedPredictions(
                predictions=predictions,
                parameters=tmp_parameters,
                conf_boxes=conf_boxes,
                conf_cls=None,
            )
            # TODO(leoandeol): classwise risk ????
            risk = self.risk_function(
                tmp_conformalized_predictions,
                predictions,
                loss=self.loss,
            )

            n = len(predictions)
            corrected_risk = self._correct_risk(
                risk=risk,
                n=n,
                B=self.loss.upper_bound,
            )
            return corrected_risk

        return objective_function

    def _correct_risk(
        self,
        risk: torch.Tensor,
        n: int,
        B: float,
    ) -> torch.Tensor:
        """Correct the risk using the number of predictions and the upper bound.

        Parameters
        ----------
        - risk (torch.Tensor): The risk tensor.
        - n (int): The number of predictions.
        - B (float): The upper bound.

        Returns
        -------
        - corrected_risk (torch.Tensor): The corrected risk tensor.

        """
        return (n / (n + 1)) * torch.mean(risk) + B / (n + 1)

    def calibrate(
        self,
        predictions: ODPredictions,
        alpha: float,
        steps: int = 13,
        bounds: List[float] = [0, 1000],
        verbose: bool = True,
        overload_confidence_threshold: Optional[float] = None,
    ) -> float:
        """Calibrate the conformalizer.

        Parameters
        ----------
        - predictions (ODPredictions): The object detection predictions.
        - alpha (float): The significance level.
        - steps (int): The number of steps for optimization.
        - bounds (List[float]): The bounds for optimization.
        - verbose (bool): Whether to print the optimization progress.
        - confidence_threshold (float): The threshold for objectness confidence.

        Returns
        -------
        - lbd (float): The calibrated lambda value.

        """
        if self.lambda_localization is not None:
            logger.info("Replacing previously computed λ")
        if overload_confidence_threshold is None:
            if predictions.confidence_threshold is None:
                raise ValueError(
                    "confidence_threshold must be set in the predictions or in the conformalizer",
                )
            confidence_threshold = predictions.confidence_threshold
            if isinstance(confidence_threshold, torch.Tensor):
                confidence_threshold = confidence_threshold.item()
            logger.info(
                f"Using predictions' confidence threshold: {confidence_threshold:.4f}",
            )
        else:
            logger.info(
                f"Using overload confidence threshold: {overload_confidence_threshold:.4f}",
            )
            confidence_threshold = overload_confidence_threshold

        objective_function = self._get_objective_function(
            predictions=predictions,
            alpha=alpha,
            confidence_threshold=confidence_threshold,
        )

        lambda_localization = self.optimizer.optimize(
            objective_function=objective_function,
            alpha=alpha,
            bounds=bounds,
            steps=steps,
            verbose=verbose,
        )

        if verbose:
            logger.info(
                f"Calibrated λ for localization: {lambda_localization}",
            )

        self.lambda_localization = lambda_localization
        return lambda_localization

    def conformalize(
        self,
        predictions: ODPredictions,
        parameters: Optional[ODParameters] = None,
        verbose: bool = True,
    ) -> List[torch.Tensor]:
        """Conformalizes the predictions using the specified lambda values for localization.

        Args:
        ----
            predictions (ODPredictions): The predictions to be conformalized.
            parameters (Optional[ODParameters], optional): The optional parameters containing the lambda value for localization. Defaults to None.
            verbose (bool, optional): Whether to display verbose information. Defaults to True.

        Returns:
        -------
            List[torch.Tensor]: The conformalized bounding boxes.

        Raises:
        ------
            ValueError: If the conformalizer is not calibrated before conformalizing.

        """
        if self.lambda_localization is None and (
            parameters is None or parameters.lambda_localization is None
        ):
            raise ValueError(
                "Conformalizer must be calibrated, or parameters provided, before conformalizing.",
            )
        if (
            parameters is not None
            and parameters.lambda_localization is not None
        ):
            if verbose:
                logger.info("Using lambda for localization from parameters")
            lambda_localization = parameters.lambda_localization
        else:
            if verbose:
                logger.info("loggingUsing previous λ for localization")
            lambda_localization = self.lambda_localization
        if isinstance(lambda_localization, list):
            if len(lambda_localization) == 4:
                lambdas = lambda_localization
            elif len(lambda_localization) == 2:
                lambdas = [lambda_localization[0], lambda_localization[1]] * 2
            elif len(lambda_localization) == 1 or isinstance(
                self.lambda_localization,
                float,
            ):
                lambdas = [lambda_localization[0]] * 4
        elif isinstance(lambda_localization, float):
            lambdas = [lambda_localization] * 4
        if verbose:
            logger.info("Conformalizing Localization with λ")
        conf_boxes = apply_margins(
            predictions.pred_boxes,
            lambdas,
            mode=self.prediction_set,
        )
        return conf_boxes

    def evaluate(
        self,
        predictions: ODPredictions,
        parameters: ODParameters,
        conformalized_predictions: ODConformalizedPredictions,
        verbose: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """ """
        conf_boxes = conformalized_predictions.conf_boxes

        if conf_boxes is None:
            raise ValueError(
                "Conformalized predictions must be provided for evaluation.",
            )

        conf_boxes = list(
            [
                (
                    x[y >= predictions.confidence_threshold]
                    if len(x[y >= predictions.confidence_threshold]) > 0
                    else x[None, y.argmax()]
                )
                for x, y in zip(conf_boxes, predictions.confidence)
            ],
        )

        losses = self.risk_function(
            conformalized_predictions,
            predictions,
            loss=self.loss,
            return_list=True,
        )

        def compute_set_size(boxes: List[torch.Tensor]) -> torch.Tensor:
            set_sizes = []
            for image_boxes in boxes:
                for box in image_boxes:
                    set_size = (box[2] - box[0]) * (box[3] - box[1])
                    set_size = set_size**0.5
                    set_sizes.append(set_size)
            set_sizes = torch.stack(set_sizes).squeeze()
            return set_sizes

        set_sizes = compute_set_size(conf_boxes)
        if verbose:
            logger.info(f"Risk = {torch.mean(losses)}")
            logger.info(f"Average set size = {torch.mean(set_sizes)}")
        return losses, set_sizes


class ConfidenceConformalizer(Conformalizer):
    """ """

    ACCEPTED_LOSSES = {"nb_boxes": ConfidenceLoss}

    def __init__(
        self,
        guarantee_level: str,
        loss: str = "nb_boxes",
        other_losses: Optional[List] = None,
        optimizer: str = "binary_search",
    ):
        """ """
        super().__init__()
        if loss not in self.ACCEPTED_LOSSES:
            raise ValueError(
                f"loss {loss} not accepted, must be one of {self.ACCEPTED_LOSSES.keys()}",
            )
        self.loss_name = loss
        self.other_losses = other_losses
        self.loss = self.ACCEPTED_LOSSES[loss](other_losses=other_losses)
        self.guarantee_level = guarantee_level

        if guarantee_level == "object":
            self.risk_function = compute_risk_object_level
        elif guarantee_level == "image":
            self.risk_function = compute_risk_image_level

        self.lbd = None
        if optimizer == "binary_search":
            self.optimizer = BinarySearchOptimizer()
        elif optimizer in ["gaussianprocess", "gpr", "kriging"]:
            self.optimizer = GaussianProcessOptimizer()
        else:
            raise ValueError(f"optimizer {optimizer} not accepted")

    def _get_objective_function(
        self,
        predictions: ODPredictions,
        alpha: float,
        overload_B=None,
        **kwargs,
    ) -> Callable[[float], torch.Tensor]:
        """Get the risk function for risk conformalization.

        Parameters
        ----------
        - predictions (ODPredictions): The object detection predictions.
        - alpha (float): The significance level.
        - objectness_threshold (float): The threshold for objectness confidence.

        Returns
        -------
        - risk_function (Callable[[float], float]): The risk function.

        """

        def objective_function(lbd: float) -> torch.Tensor:
            """Compute the risk given a lambda value.

            Parameters
            ----------
            - lbd (float): The lambda value.

            Returns
            -------
            - corrected_risk (float): The corrected risk.

            """
            # TODO(leoandeol): super costly and probably redundant
            # URGENT: fix this : store values of distances in matching so it's instantaneous to redo
            match_predictions_to_true_boxes(
                predictions,
                verbose=True,
                overload_confidence_threshold=1 - lbd,
            )
            conf_boxes = list(
                [
                    x[y >= 1 - lbd]
                    # rippity rip to my trick
                    # if len(x[y >= 1 - lbd]) > 0
                    # else x[None, y.argmax()]
                    for x, y in zip(
                        predictions.pred_boxes,
                        predictions.confidence,
                    )
                ],
            )
            # TODO(leoandeol): cleanify this
            # First enlarge bounding boxes to the max size
            # TODO(leoandeol): this is hardcoded, we should get input image size somewhere
            conf_boxes = apply_margins(
                conf_boxes,
                [500, 500, 500, 500],
                mode="additive",
            )
            # Second, prediction sets for classification with always everything
            n_classes = len(predictions.pred_cls[0][0].squeeze())
            conf_cls = [
                [[torch.arange(n_classes)] for true_cls_i_j in true_cls_i]
                for true_cls_i in predictions.true_cls
            ]

            tmp_parameters = ODParameters(
                global_alpha=alpha,
                confidence_threshold=1 - lbd,
                predictions_id=predictions.unique_id,
            )
            tmp_conformalized_predictions = ODConformalizedPredictions(
                predictions=predictions,
                parameters=tmp_parameters,
                conf_boxes=conf_boxes,
                conf_cls=conf_cls,
            )
            # TODO(leoandeol): classwise risk ????
            risk = self.risk_function(
                tmp_conformalized_predictions,
                predictions,
                loss=self.loss,
            )
            n = len(predictions)
            B = overload_B if overload_B is not None else self.loss.upper_bound
            corrected_risk = self._correct_risk(
                risk=risk,
                n=n,
                B=B,
            )
            return corrected_risk

        return objective_function

    def _correct_risk(
        self,
        risk: torch.Tensor,
        n: int,
        B: float,
    ) -> torch.Tensor:
        """Correct the risk using the number of predictions and the upper bound.

        Parameters
        ----------
        - risk (torch.Tensor): The risk tensor.
        - n (int): The number of predictions.
        - B (float): The upper bound.

        Returns
        -------
        - corrected_risk (torch.Tensor): The corrected risk tensor.

        """
        return (n / (n + 1)) * torch.mean(risk) + B / (n + 1)

    def calibrate(
        self,
        predictions: ODPredictions,
        alpha: float = 0.1,
        steps: int = 13,
        bounds: List[float] = [0, 1000],
        verbose: bool = True,
    ) -> Tuple[float, float]:
        """ """
        if self.lbd is not None:
            logger.info("Replacing previously computed λ")

        objective_function = self._get_objective_function(
            predictions=predictions,
            alpha=alpha,
            objectness_threshold=predictions.confidence_threshold,
            overload_B=0,
        )

        lambda_minus = self.optimizer.optimize(
            objective_function=objective_function,
            alpha=alpha,
            bounds=bounds,
            steps=steps,
            verbose=verbose,
        )

        objective_function = self._get_objective_function(
            predictions=predictions,
            alpha=alpha,
            objectness_threshold=predictions.confidence_threshold,
        )

        lambda_plus = self.optimizer.optimize(
            objective_function=objective_function,
            alpha=alpha,
            bounds=bounds,
            steps=steps,
            verbose=verbose,
        )
        self.lambda_plus = lambda_plus
        self.lambda_minus = lambda_minus
        return lambda_minus, lambda_plus

    def conformalize(self, predictions: ODPredictions) -> float:
        """Conformalize the object detection predictions.

        Parameters
        ----------
        - predictions (ODPredictions): The object detection predictions.

        Returns
        -------
        - conf_boxes (List[List[float]]): The conformalized bounding boxes.

        """
        if self.lbd is None:
            raise ValueError(
                "Conformalizer must be calibrated before conformalizing.",
            )
        predictions.confidence_threshold = 1 - self.lbd
        return 1 - self.lbd

    def evaluate(
        self,
        predictions: ODPredictions,
        parameters: ODParameters,
        conformalized_predictions: ODConformalizedPredictions,
        verbose: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Evaluate the conformalized predictions.

        Parameters
        ----------
        - predictions (ODPredictions): The object detection predictions.
        - conf_boxes (List[List[float]]): The conformalized bounding boxes.
        - verbose (bool): Whether to print the evaluation results.

        Returns
        -------
        - safety (torch.Tensor): The safety scores.
        - set_sizes (torch.Tensor): The set sizes.

        """
        if self.lbd is None:
            raise ValueError(
                "Conformalizer must be calibrated before evaluating.",
            )

        losses = self.risk_function(
            conformalized_predictions,
            predictions,
            loss=self.loss,
            return_list=True,
        )

        def compute_set_size(boxes: List[List[float]]) -> torch.Tensor:
            set_sizes = []
            for image_boxes in boxes:
                set_sizes.append(len(image_boxes))
            set_sizes = torch.stack(set_sizes).ravel()
            return set_sizes

        set_sizes = compute_set_size(conformalized_predictions.conf_boxes)
        if verbose:
            logger.info(f"Risk = {torch.mean(losses)}")
            logger.info(f"Average set size = {torch.mean(set_sizes)}")
        return losses, set_sizes


class ODClassificationConformalizer(ClassificationConformalizer):
    """ """

    BACKENDS = ["auto", "cp", "crc"]
    GUARANTEE_LEVELS = ["image", "object"]

    def __init__(
        self,
        method="lac",
        preprocess="softmax",
        backend="auto",
        guarantee_level="image",
        **kwargs,
    ):
        """ """
        super().__init__(method=method, preprocess=preprocess)
        if guarantee_level not in self.GUARANTEE_LEVELS:
            raise ValueError(
                f"guarantee_level {guarantee_level} not accepted, must be one of {self.GUARANTEE_LEVELS}",
            )
        self.guarantee_level = guarantee_level
        if self.guarantee_level == "object":
            self.risk_function = compute_risk_object_level
        elif self.guarantee_level == "image":
            self.risk_function = compute_risk_image_level

        if backend not in self.BACKENDS:
            raise ValueError(
                f"backend {backend} not accepted, must be one of {self.BACKENDS}",
            )
        if backend == "auto":
            self.backend = "cp"
            logger.info("Defaulting to CP backend")
        if backend == "crc":
            raise NotImplementedError("CRC backend is not supported yet")

        self.backend = backend
        self._backend_loss = LACLoss()
        self.loss = ClassificationLossWrapper(self._backend_loss)

    def _get_objective_function(
        self,
        predictions: ODPredictions,
        alpha: float,
        confidence_threshold: float,
        **kwargs,
    ) -> Callable[[float], torch.Tensor]:
        """TODO: Add docstring"""

        def objective_function(lbd: float) -> torch.Tensor:
            """Compute the risk given a lambda value.

            Parameters
            ----------
            lbd (float): The lambda value.

            Returns
            -------
            corrected_risk (float): The corrected risk.

            """
            logger.warning(
                "Currently considering that there is only one matching prediction to each true box for classification pruposes. To add later how to aggregate if multiple preidctions matched."
            )

            def get_conf_cls():
                conf_cls = []
                for i, true_cls in enumerate(predictions.true_cls):
                    conf_cls_i = []
                    for j, true_cls_i in enumerate(true_cls):
                        matched = predictions.matching[i][j][0]
                        conf_cls_i_j = predictions.pred_cls[i][matched]
                        conf_cls_i_j = conf_cls_i_j[conf_cls_i_j >= 1 - lbd]
                        conf_cls_i.append(conf_cls_i_j)
                    conf_cls_i = torch.stack(conf_cls_i)
                    conf_cls.append(conf_cls_i)
                return conf_cls

            conf_cls = get_conf_cls()
            tmp_parameters = ODParameters(
                global_alpha=alpha,
                confidence_threshold=confidence_threshold,
                predictions_id=predictions.unique_id,
            )
            tmp_conformalized_predictions = ODConformalizedPredictions(
                predictions=predictions,
                parameters=tmp_parameters,
                conf_boxes=None,
                conf_cls=conf_cls,
            )
            # TODO(leoandeol): classwise risk ????
            risk = self.risk_function(
                tmp_conformalized_predictions,
                predictions,
                loss=self.loss,
            )

            n = len(predictions)
            corrected_risk = self._correct_risk(
                risk=risk,
                n=n,
                B=self.loss.upper_bound,
            )
            return corrected_risk

        return objective_function

    def _correct_risk(
        self,
        risk: torch.Tensor,
        n: int,
        B: float,
    ) -> torch.Tensor:
        """Correct the risk using the number of predictions and the upper bound.

        Parameters
        ----------
        - risk (torch.Tensor): The risk tensor.
        - n (int): The number of predictions.
        - B (float): The upper bound.

        Returns
        -------
        - corrected_risk (torch.Tensor): The corrected risk tensor.

        """
        return (n / (n + 1)) * torch.mean(risk) + B / (n + 1)

    def calibrate(
        self,
        predictions: ODPredictions,
        alpha: float,
        bounds: List[float] = [0, 1],
        steps: int = 13,
        verbose: bool = True,
        overload_confidence_threshold: Optional[float] = None,
    ) -> torch.Tensor:
        if overload_confidence_threshold is None:
            if predictions.confidence_threshold is None:
                raise ValueError(
                    "confidence_threshold must be set in the predictions or in the conformalizer",
                )
            confidence_threshold = predictions.confidence_threshold
            if isinstance(confidence_threshold, torch.Tensor):
                confidence_threshold = confidence_threshold.item()
            logger.info(
                f"Using predictions' confidence threshold: {confidence_threshold:.4f}",
            )
        else:
            logger.info(
                f"Using overload confidence threshold: {overload_confidence_threshold:.4f}",
            )
            confidence_threshold = overload_confidence_threshold

        objective_function = self._get_objective_function(
            predictions=predictions,
            alpha=alpha,
            confidence_threshold=confidence_threshold,
        )

        lambda_classification = self.optimizer.optimize(
            objective_function=objective_function,
            alpha=alpha,
            bounds=bounds,
            steps=steps,
            verbose=verbose,
        )

        if verbose:
            logger.info(
                f"Calibrated λ for localization: {lambda_classification}",
            )

        self.lambda_classification = lambda_classification
        return lambda_classification

    def conformalize(self, predictions: ODPredictions) -> List:
        # TODO: add od parameters
        def get_conf_cls():
            conf_cls = []
            for i, true_cls in enumerate(predictions.true_cls):
                conf_cls_i = []
                for j, true_cls_i in enumerate(true_cls):
                    matched = predictions.matching[i][j][0]
                    conf_cls_i_j = predictions.pred_cls[i][matched]
                    conf_cls_i_j = conf_cls_i_j[
                        conf_cls_i_j >= 1 - self.lambda_classification
                    ]
                    conf_cls_i.append(conf_cls_i_j)
                conf_cls_i = torch.stack(conf_cls_i)
                conf_cls.append(conf_cls_i)
            return conf_cls

        return get_conf_cls()

    def evaluate(
        self,
        predictions: ODPredictions,
        parameters: Optional[ODParameters],
        conformalized_predictions: ODConformalizedPredictions,
        verbose: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        logger.warning("Evaluating classification conformalizer")

        losses = self.risk_function(
            conformalized_predictions,
            predictions,
            loss=self.loss,
            return_list=True,
        )

        # TODO(leoandeol): this should vary based on object or image level
        def compute_set_size(conf_cls: List[torch.Tensor]) -> torch.Tensor:
            set_sizes = []
            for conf_cls_i in conf_cls:
                for conf_cls_i_j in conf_cls_i:
                    set_sizes.append(len(conf_cls_i_j))
            set_sizes = torch.stack(set_sizes).squeeze()
            return set_sizes

        set_sizes = compute_set_size(predictions.conf_cls)
        if verbose:
            logger.info(f"Risk = {torch.mean(losses)}")
            logger.info(f"Average set size = {torch.mean(set_sizes)}")
        return losses, set_sizes


class ODConformalizer(Conformalizer):
    """Class representing conformalizers for object detection tasks.

    Attributes:
    ----------
        MULTIPLE_TESTING_CORRECTIONS (List[str]): List of supported multiple testing correction methods.
        BACKENDS (List[str]): List of supported backends.
        GUARANTEE_LEVELS (List[str]): List of supported guarantee levels.

    Args:
    ----
        backend (str): The backend used for the conformalization. Only 'auto' is supported currently.
        guarantee_level (str): The guarantee level for the conformalization. Must be one of ["image", "object"].
        confidence_threshold (Optional[float]): The confidence threshold used for objectness conformalization. Mutually exclusive with 'confidence_method', if set, then confidence_method must be None.
        multiple_testing_correction (Optional[str]): The method used for multiple testing correction. Must be one of ["bonferroni"] or None. None implies no correction is applied, and that a List of Alphas is expected for calibration instead.
        confidence_method (Union[ConfidenceConformalizer, str, None]): The method used for confidence conformalization. Mutually exclusive with 'confidence_threshold', if set, then confidence_threshold must be None. Either pass a ConfidenceConformalizer instance, a string representing the method (loss) name, or None.
        localization_method (Union[LocalizationConformalizer, str, None]): The method used for localization conformalization. Either pass a LocalizationConformalizer instance, a string representing the method (loss) name, or None.
        classification_method (Union[ClassificationConformalizer, str, None]): The method used for classification conformalization. Either pass a ClassificationConformalizer instance, a string representing the method (loss) name, or None.
        **kwargs: Additional keyword arguments.

    Raises:
    ------
        ValueError: If the provided backend is not supported.
        ValueError: If the provided guarantee level is not supported.
        ValueError: If both confidence_threshold and confidence_method are provided.
        ValueError: If neither confidence_threshold nor confidence_method are provided.
        ValueError: If the provided multiple_testing_correction is not supported.

    Methods:
    -------
        calibrate(predictions, global_alpha, alpha_confidence, alpha_localization, alpha_classification, verbose=True)
            Calibrates the conformalizers and returns the calibration results.

    Args:
    ----
                predictions (ODPredictions): The predictions to be calibrated.
                global_alpha (Optional[float]): The global alpha value for calibration. If multiple_testing_correction is None, individual alpha values will be used for each conformalizer.
                alpha_confidence (Optional[float]): The alpha value for the confidence conformalizer.
                alpha_localization (Optional[float]): The alpha value for the localization conformalizer.
                alpha_classification (Optional[float]): The alpha value for the classification conformalizer.
                verbose (bool, optional): Whether to print calibration information. Defaults to True.

    Returns:
    -------
                dict[str, Any]: A dictionary containing the calibration results, including target alpha values and estimated lambda values for each conformalizer.

    Raises:
    ------
                ValueError: If the multiple_testing_correction is not provided or is not valid.
                ValueError: If the global_alpha is not provided when using the Bonferroni multiple_testing_correction.
                ValueError: If explicit alpha values are provided when using the Bonferroni multiple_testing_correction.

    Note:
    ----
                - The multiple_testing_correction attribute of the class must be set before calling this method.
                - The conformalizers must be initialized before calling this method.

    """

    MULTIPLE_TESTING_CORRECTIONS = ["bonferroni"]
    BACKENDS = ["auto"]
    GUARANTEE_LEVELS = ["image", "object"]
    MATCHINGS = ["assymetric_hausdorff"]

    def __init__(
        self,
        backend: str = "auto",
        guarantee_level: str = "image",
        matching: str = "assymetric_hausdorff",
        confidence_threshold: Optional[float] = None,
        multiple_testing_correction: Optional[str] = None,
        confidence_method: Union[ConfidenceConformalizer, str, None] = None,
        localization_method: Union[
            LocalizationConformalizer,
            str,
            None,
        ] = None,
        localization_prediction_set: str = "additive",  # Fix where we type check
        classification_method: Union[
            ClassificationConformalizer,
            str,
            None,
        ] = None,
        **kwargs,
    ):
        """Initialize the ODClassificationConformalizer object.

        Parameters
        ----------
        - backend (str): The backend used for the conformalization. Only 'auto' is supported currently.
        - guarantee_level (str): The guarantee level for the conformalization. Must be one of ["image", "object"].
        - confidence_threshold (Optional[float]): The confidence threshold used for objectness conformalization.  Mutually exclusive with 'confidence_method', if set, then confidence_method must be None.
        - multiple_testing_correction (str): The method used for multiple testing correction. Must be one of ["bonferroni"] or None. None implies no correction is applied, and that a List of Alphas is expected for calibration instead.
        - confidence_method (Union[ConfidenceConformalizer, str, None]): The method used for confidence conformalization. Mutually exclusive with 'confidence_threshold', if set, then confidence_threshold must be None. Either pass a ConfidenceConformalizer instance, a string representing the method (loss) name, or None.
        - localization_method (Union[LocalizationConformalizer, str, None]): The method used for localization conformalization. Either pass a LocalizationConformalizer instance, a string representing the method (loss) name, or None.
        - classification_method (Union[ClassificationConformalizer, str, None]): The method used for classification conformalization. Either pass a ClassificationConformalizer instance, a string representing the method (loss) name, or None.
        - kwargs: Additional keyword arguments.

        """
        if backend not in self.BACKENDS:
            raise ValueError(
                f"backend {backend} not accepted, must be one of {self.BACKENDS}",
            )
        self.backend = backend

        if guarantee_level not in self.GUARANTEE_LEVELS:
            raise ValueError(
                f"guarantee_level {guarantee_level} not accepted, must be one of {self.GUARANTEE_LEVELS}",
            )
        self.guarantee_level = guarantee_level

        if matching not in self.MATCHINGS:
            raise ValueError(
                f"matching {matching} not accepted, must be one of {self.MATCHINGS}",
            )
        self.matching = matching

        if multiple_testing_correction is None:
            logger.warning(
                "No multiple_testing_correction provided, assuming no correction is needed. The explicit list of alphas is expected for calibration.",
            )
            self.multiple_testing_correction = multiple_testing_correction
        elif (
            multiple_testing_correction
            not in self.MULTIPLE_TESTING_CORRECTIONS
        ):
            raise ValueError(
                f"multiple_testing_correction {multiple_testing_correction} not accepted, must be one of {self.MULTIPLE_TESTING_CORRECTIONS}",
            )
        else:
            logger.warning(
                "Multiple testing correction is set to Bonferroni, expecting a global alpha for calibration.",
            )
            self.multiple_testing_correction = multiple_testing_correction

        if confidence_threshold is None and confidence_method is None:
            raise ValueError(
                "Either confidence_threshold or confidence_method must be set",
            )
        if confidence_threshold is not None and confidence_method is not None:
            raise ValueError(
                "confidence_threshold and confidence_method are mutually exclusive",
            )

        # Localization

        if isinstance(localization_method, str):
            self.localization_method = localization_method
            self.localization_prediction_set = localization_prediction_set
            self.localization_conformalizer = LocalizationConformalizer(
                loss=localization_method,
                guarantee_level=guarantee_level,
                prediction_set=localization_prediction_set,
                **kwargs,
            )
        elif isinstance(localization_method, LocalizationConformalizer):
            self.localization_conformalizer = localization_method
            self.localization_method = localization_method.loss_name
            self.localization_prediction_set = (
                localization_method.prediction_set
            )
        else:
            self.localization_conformalizer = None
            self.localization_method = None

        # Classification

        if isinstance(classification_method, str):
            self.classification_method = classification_method
            self.classification_conformalizer = ODClassificationConformalizer(
                loss=classification_method,
            )
        elif isinstance(classification_method, ODClassificationConformalizer):
            self.classification_conformalizer = classification_method
            self.classification_method = classification_method.method
        else:
            self.classification_conformalizer = None
            self.classification_method = None

        # Confidence

        if isinstance(confidence_threshold, float):
            self.confidence_threshold = confidence_threshold
            self.confidence_conformalizer = None
            self.confidence_method = None
        elif isinstance(confidence_method, str):
            self.confidence_threshold = None
            self.confidence_conformalizer = ConfidenceConformalizer(
                loss=confidence_method,
                guarantee_level=guarantee_level,
                other_losses=[
                    conf.loss
                    for conf in [
                        self.localization_conformalizer,
                        self.classification_conformalizer,
                    ]
                    if conf is not None
                ],
                **kwargs,
            )
            self.confidence_method = confidence_method
        elif isinstance(confidence_method, ConfidenceConformalizer):
            self.confidence_threshold = None
            self.confidence_conformalizer = confidence_method
            self.confidence_method = confidence_method.loss_name

    def calibrate(
        self,
        predictions: ODPredictions,
        global_alpha: Optional[float] = None,
        alpha_confidence: Optional[float] = None,
        alpha_localization: Optional[float] = None,
        alpha_classification: Optional[float] = None,
        verbose: bool = True,
    ) -> ODParameters:
        """Calibrates the conformalizers and returns the calibration results.

        Args:
        ----
            predictions (ODPredictions): The predictions to be calibrated.
            global_alpha (Optional[float]): The global alpha value for calibration. If multiple_testing_correction is None, individual alpha values will be used for each conformalizer.
            alpha_confidence (Optional[float]): The alpha value for the confidence conformalizer.
            alpha_localization (Optional[float]): The alpha value for the localization conformalizer.
            alpha_classification (Optional[float]): The alpha value for the classification conformalizer.
            verbose (bool, optional): Whether to print calibration information. Defaults to True.

        Returns:
        -------
            dict[str, Any]: A dictionary containing the calibration results, including target alpha values and estimated lambda values for each conformalizer.

        Raises:
        ------
            ValueError: If the multiple_testing_correction is not provided or is not valid.
            ValueError: If the global_alpha is not provided when using the Bonferroni multiple_testing_correction.
            ValueError: If explicit alpha values are provided when using the Bonferroni multiple_testing_correction.

        Note:
        ----
            - The multiple_testing_correction attribute of the class must be set before calling this method.
            - The conformalizers must be initialized before calling this method.

        """
        # Checking Multiple Testing Correction
        n_conformalizers = sum(
            x is not None
            for x in [
                self.confidence_conformalizer,
                self.localization_conformalizer,
                self.classification_conformalizer,
            ]
        )

        if self.multiple_testing_correction is None:
            if global_alpha is not None:
                raise ValueError(
                    "No multiple_testing_correction provided, expecting an explicit alpha for each conformalizer. 'global_alpha' should be 'None'.",
                )
            if (
                (
                    alpha_confidence
                    is None
                    != self.confidence_conformalizer
                    is None
                )
                or (
                    alpha_localization
                    is None
                    != self.localization_conformalizer
                    is None
                )
                or (
                    alpha_classification
                    is None
                    != self.classification_conformalizer
                    is None
                )
            ):
                raise ValueError(
                    "No multiple_testing_correction provided, expecting an explicit alpha for each conformalizer. Explicity alphas should be set only if there's a corresponding conformalizer.",
                )
            # sum only the ones that are not none
            global_alpha = sum(
                [
                    x
                    for x in [
                        alpha_confidence,
                        alpha_localization,
                        alpha_classification,
                    ]
                    if x is not None
                ],
            )
        elif self.multiple_testing_correction == "bonferroni":
            if global_alpha is None:
                raise ValueError(
                    "Bonferroni multiple_testing_correction provided, expecting a global alpha for calibration.",
                )
            if (
                alpha_classification is not None
                or alpha_localization is not None
                or alpha_confidence is not None
            ):
                raise ValueError(
                    "Bonferroni multiple_testing_correction provided, expecting a global alpha for calibration. Explicit alphas should be set to None.",
                )
            alpha_confidence = (
                global_alpha / n_conformalizers
                if self.confidence_conformalizer is not None
                else None
            )
            alpha_localization = (
                global_alpha / n_conformalizers
                if self.localization_conformalizer is not None
                else None
            )
            alpha_classification = (
                global_alpha / n_conformalizers
                if self.classification_conformalizer is not None
                else None
            )
        else:
            raise ValueError(
                f"multiple_testing_correction {self.multiple_testing_correction} not accepted, must be one of {self.MULTIPLE_TESTING_CORRECTIONS}. This check should have failed in the __init__ method.",
            )

        # Confidence

        if self.confidence_conformalizer is not None:
            if verbose:
                logger.info("Calibrating Confidence Conformalizer")

            lambda_confidence_minus, lambda_confidence_plus = (
                self.confidence_conformalizer.calibrate(
                    predictions,
                    alpha=alpha_confidence,
                    verbose=verbose,
                )
            )
            # Unique to Confidence due to dependence
            logger.info("Setting Confidence Threshold of Predictions")
            self.confidence_conformalizer.conformalize(predictions)
            self.confidence_threshold = predictions.confidence_threshold

            optimistic_confidence_threshold = 1 - lambda_confidence_minus

            if verbose:
                logger.info(
                    f"Calibrated Confidence λ : {lambda_confidence_plus:.4f}\n\t and associated Confidence Threshold : {predictions.confidence_threshold}",
                )
        else:
            predictions.confidence_threshold = self.confidence_threshold

        # Now that we fixed the confidence threshold, we need to do the matching before moving on to the next steps

        if predictions.matching is not None:
            logger.warning("Overwriting previous matching")
        if verbose:
            logger.info("Matching Predictions to True Boxes")

        match_predictions_to_true_boxes(
            predictions,
            verbose=True,
            overload_confidence_threshold=optimistic_confidence_threshold,
        )

        # Localization

        if self.localization_conformalizer is not None:
            if verbose:
                logger.info("Calibrating Localization Conformalizer")

            lambda_localization = self.localization_conformalizer.calibrate(
                predictions,
                alpha=alpha_localization,
                verbose=verbose,
                overload_confidence_threshold=optimistic_confidence_threshold,
            )

            if verbose:
                logger.info(
                    "Calibrated Localization λ : {lambda_localization}",
                )

        # Classification

        if self.classification_conformalizer is not None:
            if verbose:
                logger.info("Calibrating Classification Conformalizer")

            lambda_classification = self.classification_conformalizer.calibrate(
                predictions,
                alpha=alpha_classification,
                verbose=verbose,
                overload_confidence_threshold=optimistic_confidence_threshold,
            )

            if verbose:
                logger.info(
                    "Calibrated Classification λ : {lambda_classification}",
                )

        result = ODParameters(
            predictions_id=predictions.unique_id,
            global_alpha=global_alpha,
            alpha_confidence=alpha_confidence,
            alpha_localization=alpha_localization,
            alpha_classification=alpha_classification,
            lambda_confidence_plus=lambda_confidence_plus,
            lambda_confidence_minus=lambda_confidence_minus,
            lambda_localization=lambda_localization,
            lambda_classification=lambda_classification,
            confidence_threshold=predictions.confidence_threshold,
        )

        # Saving the last parameters id to conformalize the predictions
        self._last_parameters_id = result.unique_id

        return result

    def conformalize(
        self,
        predictions: ODPredictions,
        parameters: Optional[ODParameters] = None,
        verbose: bool = True,
    ) -> ODConformalizedPredictions:
        """Conformalize the given predictions.

        Args:
        ----
            predictions (ODPredictions): The predictions to be conformalized.
            parameters (Optional[ODParameters]): The parameters to be used for conformalization. If None, the last parameters will be used.
            verbose (bool): Whether to print conformalization information.

        Returns:
        -------
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing the conformalized predictions.

        """
        if verbose:
            logger.info("Conformalizing Predictions")
        if parameters is not None:
            if verbose:
                logger.info("Using provided parameters for conformalization")
            if parameters.predictions_id != predictions.unique_id:
                raise ValueError(
                    "The parameters have been computed on another set of predictions.",
                )
            parameters_unique_id = parameters.unique_id
        else:
            if verbose:
                logger.info("Using last parameters for conformalization")
            parameters_unique_id = self._last_parameters_id

        # TODO: later check this is correct behavior
        if self.confidence_conformalizer is not None:
            if verbose:
                logger.info("Conformalizing Confidence")
            self.confidence_conformalizer.conformalize(
                predictions,
                parameters=parameters,
            )
        elif parameters is not None:
            if verbose:
                logger.info("Using provided confidence threshold")
            predictions.confidence_threshold = parameters.confidence_threshold
        else:
            if verbose:
                logger.info("Using last confidence threshold")
            predictions.confidence_threshold = self.confidence_threshold

        if self.localization_conformalizer is not None:
            if verbose:
                logger.info("Conformalizing Localization")
            conf_boxes = self.localization_conformalizer.conformalize(
                predictions,
                parameters=parameters,
            )
        else:
            conf_boxes = None

        if self.classification_conformalizer is not None:
            if verbose:
                logger.info("Conformalizing Classification")
            conf_cls = self.classification_conformalizer.conformalize(
                predictions,
                parameters=parameters,
            )
        else:
            conf_cls = None

        results = ODConformalizedPredictions(
            predictions_id=predictions.unique_id,
            parameters_id=parameters_unique_id,
            conf_boxes=conf_boxes,
            conf_cls=conf_cls,
        )

        return results

    def evaluate(
        self,
        predictions: ODPredictions,
        parameters: ODParameters,
        conformalized_predictions: ODConformalizedPredictions,
        include_confidence_in_global: bool,
        verbose: bool = True,
    ) -> ODResults:
        if self.localization_conformalizer is not None:
            if verbose:
                logger.info("Evaluating Localization Conformalizer")
            coverage_loc, set_size_loc = (
                self.localization_conformalizer.evaluate(
                    predictions,
                    parameters,
                    conformalized_predictions,
                    verbose=False,
                )
            )
        else:
            coverage_loc, set_size_loc = None, None
        if self.confidence_conformalizer is not None:
            if verbose:
                logger.info("Evaluating Confidence Conformalizer")
            coverage_obj, set_size_obj = (
                self.confidence_conformalizer.evaluate(
                    predictions,
                    parameters,
                    conformalized_predictions,
                    verbose=False,
                )
            )
        else:
            coverage_obj, set_size_obj = None, None
        if self.classification_conformalizer is not None:
            if verbose:
                logger.info("Evaluating Classification Conformalizer")
            coverage_cls, set_size_cls = (
                self.classification_conformalizer.evaluate(
                    predictions,
                    parameters,
                    conformalized_predictions,
                    verbose=False,
                )
            )
        else:
            coverage_cls, set_size_cls = None, None

        global_coverage = compute_global_coverage(
            predictions=predictions,
            parameters=parameters,
            conformalized_predictions=conformalized_predictions,
            confidence=(
                self.obj_conformalizer is not None
                if include_confidence_in_global
                else False
            ),
            cls=self.cls_conformalizer is not None,
            localization=self.loc_conformalizer is not None,
        )

        # TODO: Use parameters to compare distance to ideal coverage and other things

        if verbose:
            # log results
            logger.info("Evaluation Results:")
            if self.confidence_conformalizer is not None:
                logger.info("\t Confidence:")
                logger.info(f"\t\t Coverage: {torch.mean(coverage_obj):.2f}")
                logger.info(
                    f"\t\t Mean Set Size: {torch.mean(set_size_obj):.2f}",
                )
            if self.localization_conformalizer is not None:
                logger.info("\t Localization:")
                logger.info(f"\t\t Coverage: {torch.mean(coverage_loc):.2f}")
                logger.info(
                    f"\t\t Mean Set Size: {torch.mean(set_size_loc):.2f}",
                )
            if self.classification_conformalizer is not None:
                logger.info("\t Classification:")
                logger.info(f"\t\t Coverage: {torch.mean(coverage_cls):.2f}")
                logger.info(
                    f"\t\t Mean Set Size: {torch.mean(set_size_cls):.2f}",
                )
            if global_coverage is not None:
                logger.info("\t Global:")
                logger.info(
                    f"\t\t Coverage: {torch.mean(global_coverage):.2f}",
                )

        results = ODResults(
            predictions=predictions,
            parameters=parameters,
            conformalized_predictions=conformalized_predictions,
            coverage_cls=coverage_cls,
            set_size_cls=set_size_cls,
            coverage_obj=coverage_obj,
            set_size_obj=set_size_obj,
            coverage_loc=coverage_loc,
            set_size_loc=set_size_loc,
            global_coverage=global_coverage,
        )

        return results


###################################################################
##################### SEQ CRC #####################################
###################################################################


class SeqGlobalODConformalizer(ODConformalizer):
    def __init__(
        self,
        localization_method: str,
        objectness_method: str,
        classification_method: str,
        confidence_threshold: Optional[float] = None,
        fix_cls=False,
        **kwargs,
    ):
        self.objectness_method = objectness_method
        self.obj_conformalizer = ConfidenceConformalizer(
            method=objectness_method,
        )

        self.classification_method = classification_method

        self.cls_conformalizer = ClassificationConformalizer(
            method=self.classification_method,
        )

        self.loc_conformalizer = LocalizationConformalizer(
            prediction_set="additive",
            loss="classboxwise",
        )
        self.fix_cls = fix_cls
        self.multiple_testing_correction = "none"
        self.confidence_threshold = confidence_threshold
        if (
            self.confidence_threshold is not None
            and self.obj_conformalizer is not None
        ):
            # TODO: replace by warnings
            logger.info(
                "Warning: confidence_threshold is ignored if objectness_method is not None",
            )

    def calibrate(
        self,
        predictions: ODPredictions,
        alpha: float = 0.1,
        verbose: bool = True,
    ) -> Tuple[Sequence[float], float, float]:
        if self.multiple_testing_correction == "none":
            # real_alpha = alpha / sum(
            #     x is not None
            #     for x in [
            #         self.loc_conformalizer,
            #         self.obj_conformalizer,
            #         self.cls_conformalizer,
            #     ]
            # )

            real_alpha = alpha
        else:
            raise ValueError(
                f"multiple_testing_correction {self.multiple_testing_correction} not accepted, should be one of {self.MULTIPLE_TESTING_CORRECTIONS}",
            )

        # Confidence

        quantile_obj_confidence_minus = self.obj_conformalizer.calibrate(
            predictions,
            alpha=real_alpha * 0.5,
            verbose=verbose,
            override_B=True,
        )
        minus_conf_threshold = 1 - quantile_obj_confidence_minus
        logger.info(
            "Minus confidence threshold",
            minus_conf_threshold,
            quantile_obj_confidence_minus,
        )
        quantile_obj_confidence = self.obj_conformalizer.calibrate(
            predictions,
            alpha=real_alpha * 0.5,
            verbose=verbose,
        )
        confidence_threshold = 1 - quantile_obj_confidence
        predictions.confidence_threshold = minus_conf_threshold
        logger.info(
            "Plus confidence threshold",
            confidence_threshold,
            quantile_obj_confidence,
        )

        # Classif
        if self.fix_cls:
            # lbd minus
            cls_predictions = get_classif_predictions_from_od_predictions(
                predictions
            )
            quantile_classif, score_cls_min = self.cls_conformalizer.calibrate(
                cls_predictions,
                alpha=real_alpha * 0.5,
                verbose=verbose,
                lbd_minus=True,
            )
            conf_cls = get_conf_cls_for_od(predictions, self.cls_conformalizer)
            # for the real lbd+
            quantile_classif, score_cls_plus = (
                self.cls_conformalizer.calibrate(
                    cls_predictions,
                    alpha=real_alpha * 0.5,
                    verbose=verbose,
                )
            )
        else:
            cls_predictions = get_classif_predictions_from_od_predictions(
                predictions
            )
            quantile_classif, score_cls = self.cls_conformalizer.calibrate(
                cls_predictions,
                alpha=real_alpha * 0.5,
                verbose=verbose,
            )
            conf_cls = get_conf_cls_for_od(predictions, self.cls_conformalizer)
            score_cls_min, score_cls_plus = score_cls, score_cls

        # Localization

        quantile_localization = self.loc_conformalizer.calibrate(
            predictions,
            conf_cls=conf_cls,
            alpha=real_alpha,
            verbose=verbose,
        )

        predictions.confidence_threshold = confidence_threshold

        if verbose:
            logger.info("Quantiles")
            if self.obj_conformalizer is not None:
                logger.info(f"Confidence: {quantile_obj_confidence}")
            if self.loc_conformalizer is not None:
                logger.info(f"Localization: {quantile_localization}")
            if self.cls_conformalizer is not None:
                logger.info(f"Classification: {quantile_classif}")

        # TODO: future move to dictionary for better handling
        return (
            quantile_localization,
            quantile_obj_confidence,
            quantile_classif,
            score_cls_min,
            score_cls_plus,
        )

    def evaluate(
        self,
        predictions: ODPredictions,
        conf_boxes: list,
        conf_cls: list,
        verbose: bool = True,
    ):
        """Evaluate the conformalizers.

        Parameters
        ----------
        - predictions: The ODPredictions object containing the predictions.
        - conf_boxes: The conformalized bounding boxes.
        - conf_cls: The conformalized classification scores.
        - verbose: Whether to print the evaluation results.

        """
        if self.loc_conformalizer is not None:
            coverage_loc, set_size_loc = self.loc_conformalizer.evaluate(
                predictions,
                conf_boxes,
                conf_cls=conf_cls,
                verbose=False,
            )
        else:
            coverage_loc, set_size_loc = None, None
        if self.obj_conformalizer is not None:
            coverage_obj, set_size_obj = self.obj_conformalizer.evaluate(
                predictions,
                conf_boxes,
                verbose=False,
            )
        else:
            coverage_obj, set_size_obj = None, None
        if self.cls_conformalizer is not None:
            coverage_cls, set_size_cls = evaluate_cls_conformalizer(
                predictions,
                conf_cls,
                self.cls_conformalizer,
                verbose=False,
            )
        else:
            coverage_cls, set_size_cls = None, None

        if self.loc_conformalizer.loss_name == "classboxwise":
            loss = BoxWiseRecallLoss()
        global_coverage = compute_global_coverage(
            predictions=predictions,
            conf_boxes=conf_boxes,
            conf_cls=conf_cls,
            confidence=self.obj_conformalizer is not None,
            cls=self.cls_conformalizer is not None,
            localization=self.loc_conformalizer is not None,
            loss=loss,
        )
        if verbose:
            logger.info("Confidence:")
            logger.info(f"\t Coverage: {torch.mean(coverage_obj):.2f}")
            logger.info(f"\t Mean Set Size: {torch.mean(set_size_obj):.2f}")
            logger.info("Classification:")
            logger.info(f"\t Coverage: {torch.mean(coverage_cls):.2f}")
            logger.info(f"\t Mean Set Size: {torch.mean(set_size_cls):.2f}")
            logger.info("Localization:")
            logger.info(f"\t Coverage: {torch.mean(coverage_loc):.2f}")
            logger.info(f"\t Mean Set Size: {torch.mean(set_size_loc):.2f}")
            logger.info("Global:")
            logger.info(f"\t Coverage: {torch.mean(global_coverage):.2f}")

        return (
            coverage_obj,
            coverage_loc,
            coverage_cls,
            set_size_obj,
            set_size_loc,
            set_size_cls,
            global_coverage,
        )


####################################################################################################


class AsymptoticLocalizationObjectnessConformalizer(Conformalizer):
    """A class that performs risk conformalization for object detection predictions with asymptotic localization and objectness.

    Args:
    ----
        prediction_set (str): The type of prediction set to use. Must be one of "additive", "multiplicative", or "adaptative".
        localization_loss (str): The type of localization loss to use. Must be one of "pixelwise" or "boxwise".
        optimizer (str): The type of optimizer to use. Must be one of "gaussianprocess", "gpr", "kriging", "mc", or "montecarlo".

    Attributes:
    ----------
        ACCEPTED_LOSSES (dict): A dictionary mapping accepted localization losses to their corresponding classes.
        loss_name (str): The name of the localization loss.
        loss (Loss): An instance of the localization loss class.
        prediction_set (str): The type of prediction set.
        lbd (tuple): The calibrated lambda values.

    Methods:
    -------
        _get_risk_function: Returns the risk function for optimization.
        _correct_risk: Corrects the risk using the number of predictions and the upper bound of the loss.
        calibrate: Calibrates the conformalizer using the given predictions.
        conformalize: Conformalizes the predictions using the calibrated lambda values.
        evaluate: Evaluates the conformalized predictions.

    """

    ACCEPTED_LOSSES = {
        "pixelwise": PixelWiseRecallLoss,
        "boxwise": BoxWiseRecallLoss,
    }

    def __init__(
        self,
        prediction_set: str = "additive",
        localization_loss: str = "boxwise",
        optimizer: str = "gpr",
    ):
        super().__init__()
        if localization_loss not in self.ACCEPTED_LOSSES:
            raise ValueError(
                f"loss {localization_loss} not accepted, must be one of {self.ACCEPTED_LOSSES.keys()}",
            )
        self.loss_name = localization_loss
        self.loss = self.ACCEPTED_LOSSES[localization_loss]()

        if prediction_set not in ["additive", "multiplicative", "adaptative"]:
            raise ValueError(f"prediction_set {prediction_set} not accepted")
        self.prediction_set = prediction_set
        self.lbd = None
        if optimizer in ["gaussianprocess", "gpr", "kriging"]:
            self.optimizer = GaussianProcessOptimizer()
        elif optimizer in ["mc", "montecarlo"]:
            self.optimizer = MonteCarloOptimizer()
        else:
            raise ValueError(
                f"optimizer {optimizer} not accepted in multidim, currently only gpr and mc",
            )

    def _get_risk_function(self, predictions, alpha, **kwargs):
        """Returns the risk function for optimization.

        Args:
        ----
            predictions (ODPredictions): The object detection predictions.
            alpha (float): The significance level.

        Returns:
        -------
            function: The risk function.

        """

        def risk_function(*lbd):
            lbd_loc, lbd_obj = lbd
            pred_boxes_filtered = list(
                [
                    (
                        x[y >= 1 - lbd_obj]
                        if len(x[y >= 1 - lbd_obj]) > 0
                        else x[None, y.argmax()]
                    )
                    for x, y in zip(
                        predictions.pred_boxes, predictions.confidence
                    )
                ],
            )
            conf_boxes = apply_margins(
                pred_boxes_filtered,
                [lbd_loc, lbd_loc, lbd_loc, lbd_loc],
                mode=self.prediction_set,
            )
            risk = compute_risk_object_level(
                conf_boxes,
                predictions.true_boxes,
                loss=self.loss,
            )
            n = len(predictions)
            corrected_risk = self._correct_risk(
                risk=risk,
                n=n,
                B=self.loss.upper_bound,
            )
            return corrected_risk

        return risk_function

    def _correct_risk(self, risk, n, B):
        """Corrects the risk using the number of predictions and the upper bound of the loss.

        Args:
        ----
            risk (torch.Tensor): The risk values.
            n (int): The number of predictions.
            B (float): The upper bound of the loss.

        Returns:
        -------
            torch.Tensor: The corrected risk values.

        """
        return (n / (n + 1)) * torch.mean(risk) + B / (n + 1)

    def calibrate(
        self,
        predictions: ODPredictions,
        alpha: float = 0.1,
        steps: int = 13,
        bounds: list = [(0, 500), (0.0, 1.0)],
        verbose: bool = True,
    ):
        """Calibrates the conformalizer using the given predictions.

        Args:
        ----
            predictions (ODPredictions): The object detection predictions.
            alpha (float): The significance level.
            steps (int): The number of optimization steps.
            bounds (list): The bounds for the optimization variables.
            verbose (bool): Whether to print verbose output.

        Returns:
        -------
            tuple: The calibrated lambda values.

        Raises:
        ------
            ValueError: If the conformalizer has already been calibrated.

        """
        if self.lbd is not None:
            logger.info("Replacing previously computed lambda")
        risk_function = self._get_risk_function(
            predictions=predictions,
            alpha=alpha,
        )

        lbd = self.optimizer.optimize(
            objective_function=risk_function,
            alpha=alpha,
            bounds=bounds,
            steps=steps,
            verbose=verbose,
        )
        self.lbd = lbd
        return lbd

    def conformalize(self, predictions: ODPredictions):
        """Conformalizes the predictions using the calibrated lambda values.

        Args:
        ----
            predictions (ODPredictions): The object detection predictions.

        Returns:
        -------
            list: The conformalized bounding boxes.

        Raises:
        ------
            ValueError: If the conformalizer has not been calibrated.

        """
        if self.lbd is None:
            raise ValueError(
                "Conformalizer must be calibrated before conformalizing.",
            )
        conf_boxes = apply_margins(
            predictions.pred_boxes,
            [self.lbd[0]] * 4,
            mode=self.prediction_set,
        )
        predictions.confidence_threshold = 1 - self.lbd[1]
        predictions.conf_boxes = conf_boxes
        return conf_boxes

    def evaluate(
        self,
        predictions: ODPredictions,
        conf_boxes: list,
        verbose: bool = True,
    ):
        """Evaluates the conformalized predictions.

        Args:
        ----
            predictions (ODPredictions): The object detection predictions.
            conf_boxes (list): The conformalized bounding boxes.
            verbose (bool): Whether to print verbose output.

        Returns:
        -------
            tuple: The evaluation results.

        Raises:
        ------
            ValueError: If the conformalizer has not been calibrated or the predictions have not been conformalized.

        """
        if self.lbd is None:
            raise ValueError(
                "Conformalizer must be calibrated before evaluating.",
            )
        if predictions.conf_boxes is None:
            raise ValueError(
                "Predictions must be conformalized before evaluating.",
            )
        coverage_obj = []
        set_size_obj = []
        for true_boxes, confidence in zip(
            predictions.true_boxes, predictions.confidence
        ):
            cov = (
                1
                if len(true_boxes)
                <= (confidence >= predictions.confidence_threshold).sum()
                else 0
            )
            set_size = (confidence >= predictions.confidence_threshold).sum()
            set_size_obj.append(set_size)
            coverage_obj.append(cov)
        if verbose:
            logger.info(
                f"Confidence Treshold {predictions.confidence_threshold}, Coverage = {torch.mean(coverage_obj)}, Median set size = {torch.mean(set_size_obj)}",
            )

        coverage_loc = []

        def compute_set_size(boxes):
            set_sizes = []
            for image_boxes in boxes:
                for box in image_boxes:
                    set_size = (box[2] - box[0]) * (box[3] - box[1])
                    set_size = set_size.item()
                    set_size = torch.sqrt(set_size)
                    set_sizes.append(set_size)
            return set_sizes

        conf_boxes = conf_boxes
        true_boxes = predictions.true_boxes
        conf_boxes = list(
            [
                (
                    x[y >= predictions.confidence_threshold]
                    if len(x[y >= predictions.confidence_threshold]) > 0
                    else x[None, y.argmax()]
                )
                for x, y in zip(conf_boxes, predictions.confidence)
            ],
        )
        set_size_loc = compute_set_size(conf_boxes)
        risk = compute_risk_object_level(
            conf_boxes,
            true_boxes,
            loss=self.loss,
        )
        safety = 1 - risk
        if verbose:
            logger.info(f"Safety = {safety}")
            logger.info(f"Average set size = {torch.mean(set_size_loc)}")
        global_coverage = compute_global_coverage(
            predictions=predictions,
            also_conf=True,
            also_cls=False,
            loss=self.loss,
        )
        return (
            coverage_obj,
            safety,
            set_size_obj,
            set_size_loc,
            global_coverage,
        )
