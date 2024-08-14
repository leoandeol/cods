import logging
from concurrent.futures import thread
from typing import Any, Callable, List, Optional, Sequence, Tuple, Union

logger = logging.getLogger("cods")

import torch
from tqdm import tqdm

from cods.base.cp import Conformalizer, RiskConformalizer
from cods.base.optim import (
    BinarySearchOptimizer,
    GaussianProcessOptimizer,
    MonteCarloOptimizer,
)
from cods.classif.cp import ClassificationConformalizer
from cods.od.data import ODParameters, ODPredictions, ODResults, ODConformalizedPredictions
from cods.od.loss import (
    BoxWiseRecallLoss,
    ClassBoxWiseRecallLoss,
    ClassificationLossWrapper,
    MaximumLoss,
    ObjectnessLoss,
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
    compute_risk_box_level,
    compute_risk_cls_box_level,
    compute_risk_cls_image_level,
    compute_risk_image_level,
    evaluate_cls_conformalizer,
    flatten_conf_cls,
    get_classif_preds_from_od_preds,
    get_conf_cls_for_od,
)

################ BASIC BRICS ####################################################


class LocalizationConformalizer(Conformalizer):
    def __init__(
        self,
        method: str = "min-hausdorff-additive",
        margins: int = 1,  # where to compute 1, 2 or 4 margins with bonferroni corrections
        **kwargs,
    ):
        """
        Conformalizer for object localization tasks.

        Args:
            method (str): The method to compute non-conformity scores. Must be one of ["min-hausdorff-additive", "min-hausdorff-multiplicative", "union-hausdorff-additive", "union-hausdorff-multiplicative"].
            margins (int): The number of margins to compute. Must be one of [1, 2, 4].
            **kwargs: Additional keyword arguments.
        """
        self.accepted_methods = {
            "min-hausdorff-additive": MinAdditiveSignedAssymetricHausdorffNCScore,
            "min-hausdorff-multiplicative": MinMultiplicativeSignedAssymetricHausdorffNCScore,
            "union-hausdorff-additive": UnionAdditiveSignedAssymetricHausdorffNCScore,
            "union-hausdorff-multiplicative": UnionMultiplicativeSignedAssymetricHausdorffNCScore,
        }
        if method not in self.accepted_methods:
            raise ValueError(
                f"method {method} not accepted, must be one of {self.accepted_methods.keys()}"
            )
        self.method = method
        if margins not in [1, 2, 4]:
            raise ValueError(
                f"margins {margins} not accepted, must be one of [1, 2, 4]"
            )
        self.margins = margins
        self._score_function = None
        self.quantiles = None
        self.scores = None

    def calibrate(
        self,
        preds: ODPredictions,
        alpha: float = 0.1,
        confidence_threshold: Union[float, None] = None,
        verbose: bool = True,
    ) -> list:
        """
        Calibrates the conformalizer using the given predictions.

        Args:
            preds (ODPredictions): The object detection predictions.
            alpha (float): The significance level for the calibration.
            confidence_threshold (float, optional): The confidence threshold for the predictions. If not provided, it must be set in the predictions or in the conformalizer.
            verbose (bool): Whether to display progress information.

        Returns:
            list: The computed quantiles for each margin.
        """
        if self._score_function is None:
            self._score_function = self.accepted_methods[self.method]()
        if preds.confidence_threshold is None:
            if confidence_threshold is None:
                raise ValueError(
                    "confidence_threshold must be set in the predictions or in the conformalizer"
                )
            else:
                preds.confidence_threshold = confidence_threshold
        self.confidence_threshold = preds.confidence_threshold

        if self.scores is None:
            # compute all non-conformity scores for each four axes
            scores = []
            for i, true_box_img in tqdm(
                enumerate(preds.true_boxes), disable=not verbose
            ):
                for j, true_box in enumerate(true_box_img):
                    confidences = preds.confidence[i]
                    pred_boxes = (
                        preds.pred_boxes[i][confidences >= preds.confidence_threshold]
                        if len(
                            preds.pred_boxes[i][
                                confidences >= preds.confidence_threshold
                            ]
                        )
                        > 0
                        else preds.pred_boxes[i][confidences.argmax()]
                    )
                    score = self._score_function(pred_boxes, true_box)
                    scores.append(score)
            scores = torch.stack(scores).squeeze()
            self.scores = scores
            n = len(scores)
        else:
            scores = torch.clone(self.scores.detach())
            n = len(scores)

        # 1 margin: take max over all four scores
        if self.margins == 1:
            scores_1, _ = torch.max(scores, dim=-1)
            q = torch.quantile(
                scores_1,
                (1 - alpha) * (n + 1) / n,
                interpolation="higher",
            )
            quantiles = [q] * 4
        # 2 margins: take maximum on x [0, 2] and y [1, 3] axes
        elif self.margins == 2:
            scores_1, _ = torch.max(scores[:, [0, 2]], dim=-1)
            scores_2, _ = torch.max(scores[:, [1, 3]], dim=-1)
            # must apply statistical correction (bonferroni correction)
            q1 = torch.quantile(
                scores_1,
                (1 - alpha / 2) * (n + 1) / n,
                interpolation="higher",
            )
            q2 = torch.quantile(
                scores_2,
                (1 - alpha / 2) * (n + 1) / n,
                interpolation="higher",
            )
            quantiles = [q1, q2, q1, q2]
        # 4 margins: take quantile on each axis
        elif self.margins == 4:
            q1 = torch.quantile(
                scores[:, 0],
                (1 - alpha / 4) * (n + 1) / n,
                interpolation="higher",
            )
            q2 = torch.quantile(
                scores[:, 1],
                (1 - alpha / 4) * (n + 1) / n,
                interpolation="higher",
            )
            q3 = torch.quantile(
                scores[:, 2],
                (1 - alpha / 4) * (n + 1) / n,
                interpolation="higher",
            )
            q4 = torch.quantile(
                scores[:, 3],
                (1 - alpha / 4) * (n + 1) / n,
                interpolation="higher",
            )
            quantiles = [q1, q2, q3, q4]

        self.quantiles = quantiles
        return quantiles

    def conformalize(self, preds: ODPredictions) -> list:
        """
        Conformalizes the object detection predictions using the calibrated quantiles.

        Args:
            preds (ODPredictions): The object detection predictions.

        Returns:
            list: The conformalized bounding boxes.
        """
        if self.quantiles is None:
            raise ValueError("Conformalizer must be calibrated before conformalizing.")

        conf_boxes = self._score_function.apply_margins(
            preds.pred_boxes, self.quantiles
        )
        preds.confidence_threshold = self.confidence_threshold
        return conf_boxes

    def evaluate(
        self, preds: ODPredictions, conf_boxes: list, verbose: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluates the conformalized predictions.

        Args:
            preds (ODPredictions): The object detection predictions.
            conf_boxes (list): The conformalized bounding boxes.
            verbose (bool): Whether to display evaluation results.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The computed coverage and set sizes.
        """
        if self.quantiles is None:
            raise ValueError("Conformalizer must be calibrated before evaluating.")
        if conf_boxes is None:
            raise ValueError("Predictions must be conformalized before evaluating.")
        # TODO fix Height and width
        H, W = 800, 600
        coverage = []
        set_sizes = []
        for i, true_box_img in enumerate(preds.true_boxes):
            for j, true_box in enumerate(true_box_img):
                acc = torch.zeros(1).float()
                set_size = torch.tensor(H * W, dtype=torch.float).cuda() ** 0.5
                for conf_box in conf_boxes[i]:
                    sizes = []
                    if (
                        true_box[0] >= conf_box[0]
                        and true_box[1] >= conf_box[1]
                        and true_box[2] <= conf_box[2]
                        and true_box[3] <= conf_box[3]
                    ):
                        acc = torch.ones(1).float()
                        # area of the conformal box
                        set_size = (conf_box[2] - conf_box[0]) * (
                            conf_box[3] - conf_box[1]
                        )
                        set_size = set_size**0.5
                        sizes.append(set_size)
                if len(sizes) > 0:
                    set_size = torch.stack(sizes).min()
                coverage.append(acc)
                set_sizes.append(
                    set_size  # .detach().cpu().numpy()
                    # if isinstance(set_size, torch.Tensor)
                    # else set_size
                )
        coverage = torch.stack(coverage).squeeze()
        set_sizes = torch.stack(set_sizes).squeeze()
        if verbose:
            print(f"Coverage = {torch.mean(coverage)}")
            print(f"Average set size = {torch.mean(set_sizes)}")
        return coverage, set_sizes


class ObjectnessConformalizer(Conformalizer):
    def __init__(self, method: str = "box_number", **kwargs):
        """
        Conformalizer for objectness tasks.

        Args:
            method (str): The method to compute non-conformity scores. Must be "box_number".
            **kwargs: Additional keyword arguments.
        """
        self.accepted_methods = {"box_number": ObjectnessNCScore}
        self.method = method

        self._score_function = None
        self._quantile = None

    def calibrate(
        self,
        preds: ODPredictions,
        alpha: float = 0.1,
        verbose: bool = True,
        override_B=None,
    ) -> torch.Tensor:
        """
        Calibrates the conformalizer using the given predictions.

        Args:
            preds (ODPredictions): The object detection predictions.
            alpha (float): The significance level for the calibration.
            verbose (bool): Whether to display progress information.

        Returns:
            float: The computed quantile.
        """
        if self._score_function is None:
            self._score_function = self.accepted_methods[self.method]()

        scores = []
        for true_boxes, confidence in zip(preds.true_boxes, preds.confidence):
            score = self._score_function(len(true_boxes), confidence)
            scores.append(score)
        scores = torch.stack(scores).squeeze()

        n = len(scores)
        if override_B is not None:
            quantile = torch.quantile(scores, (1 - alpha), interpolation="higher")
        else:
            quantile = torch.quantile(
                scores, (1 - alpha) * ((n + 1) / n), interpolation="higher"
            )
        self._quantile = quantile
        return quantile

    def conformalize(self, preds: ODPredictions) -> torch.Tensor:
        """
        Conformalizes the object detection predictions using the calibrated quantile.

        Args:
            preds (ODPredictions): The object detection predictions.

        Returns:
            float: The conformalized confidence threshold.
        """
        if self._quantile is None:
            raise ValueError("Conformalizer must be calibrated before conformalizing.")
        preds.confidence_threshold = 1 - self._quantile
        return preds.confidence_threshold

    def evaluate(
        self, preds: ODPredictions, conf_boxes: list, verbose: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluates the conformalized predictions.

        Args:
            preds (ODPredictions): The object detection predictions.
            conf_boxes (list): The conformalized bounding boxes.
            verbose (bool): Whether to display evaluation results.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The computed coverage and set sizes.
        """
        if self._quantile is None:
            raise ValueError("Conformalizer must be calibrated before evaluating.")
        if preds.confidence_threshold is None:
            raise ValueError("Predictions must be conformalized before evaluating.")
        coverage = []
        set_sizes = []
        for true_boxes, confidence in zip(preds.true_boxes, preds.confidence):
            cov = (
                torch.ones(1).float()
                if len(true_boxes) <= (confidence >= preds.confidence_threshold).sum()
                else torch.zeros(1).float()
            )
            set_size = (confidence >= preds.confidence_threshold).sum().float()
            set_sizes.append(set_size)
            coverage.append(cov)
        coverage = torch.stack(coverage).squeeze()
        set_sizes = torch.stack(set_sizes).squeeze()
        if verbose:
            print(
                f"Confidence Treshold {preds.confidence_threshold}, Coverage = {torch.mean(coverage)}, Median set size = {torch.mean(set_sizes)}"
            )
        return coverage, set_sizes


class LocalizationRiskConformalizer(RiskConformalizer):
    """
    A class that performs risk conformalization for localization tasks.
    """

    ACCEPTED_LOSSES = {"pixelwise": PixelWiseRecallLoss, "boxwise": BoxWiseRecallLoss}

    def __init__(
        self,
        prediction_set: str = "additive",
        loss: str = None,
        optimizer: str = "binary_search",
    ):
        """
        Initialize the LocalizationRiskConformalizer.

        Parameters:
        - prediction_set (str): The type of prediction set to use. Must be one of ["additive", "multiplicative", "adaptative"].
        - loss (str): The type of loss to use. Must be one of ["pixelwise", "boxwise"].
        - optimizer (str): The type of optimizer to use. Must be one of ["binary_search", "gaussianprocess", "gpr", "kriging"].
        """
        super().__init__()
        if loss not in self.ACCEPTED_LOSSES:
            raise ValueError(
                f"loss {loss} not accepted, must be one of {self.ACCEPTED_LOSSES.keys()}"
            )
        self.loss_name = loss
        self.loss = self.ACCEPTED_LOSSES[loss]()

        if prediction_set not in ["additive", "multiplicative", "adaptative"]:
            raise ValueError(f"prediction_set {prediction_set} not accepted")
        self.prediction_set = prediction_set
        self.lbd = None
        if optimizer == "binary_search":
            self.optimizer = BinarySearchOptimizer()
        elif optimizer in ["gaussianprocess", "gpr", "kriging"]:
            self.optimizer = GaussianProcessOptimizer()
        else:
            raise ValueError(f"optimizer {optimizer} not accepted")

    def _get_risk_function(
        self, preds: ODPredictions, alpha: float, objectness_threshold: float, **kwargs
    ) -> Callable[[float], float]:
        """
        Get the risk function for risk conformalization.

        Parameters:
        - preds (ODPredictions): The object detection predictions.
        - alpha (float): The significance level.
        - objectness_threshold (float): The threshold for objectness confidence.

        Returns:
        - risk_function (Callable[[float], float]): The risk function.
        """
        pred_boxes_filtered = list(
            [
                (
                    x[y >= objectness_threshold]
                    if len(x[y >= objectness_threshold]) > 0
                    else x[None, y.argmax()]
                )
                for x, y in zip(preds.pred_boxes, preds.confidence)
            ]
        )

        def risk_function(lbd: float) -> float:
            """
            Compute the risk given a lambda value.

            Parameters:
            - lbd (float): The lambda value.

            Returns:
            - corrected_risk (float): The corrected risk.
            """
            conf_boxes = apply_margins(
                pred_boxes_filtered, [lbd, lbd, lbd, lbd], mode=self.prediction_set
            )
            risk = compute_risk_box_level(
                conf_boxes,
                preds.true_boxes,
                loss=self.loss,
            )
            n = len(preds)
            corrected_risk = self._correct_risk(
                risk=risk,
                n=n,
                B=self.loss.upper_bound,
            )
            return corrected_risk

        return risk_function

    def _correct_risk(self, risk: torch.Tensor, n: int, B: float) -> torch.Tensor:
        """
        Correct the risk using the number of predictions and the upper bound.

        Parameters:
        - risk (torch.Tensor): The risk tensor.
        - n (int): The number of predictions.
        - B (float): The upper bound.

        Returns:
        - corrected_risk (torch.Tensor): The corrected risk tensor.
        """
        return (n / (n + 1)) * torch.mean(risk) + B / (n + 1)

    def calibrate(
        self,
        preds: ODPredictions,
        alpha: float = 0.1,
        steps: int = 13,
        bounds: List[float] = [0, 1000],
        verbose: bool = True,
        confidence_threshold: float = None,
    ) -> float:
        """
        Calibrate the conformalizer.

        Parameters:
        - preds (ODPredictions): The object detection predictions.
        - alpha (float): The significance level.
        - steps (int): The number of steps for optimization.
        - bounds (List[float]): The bounds for optimization.
        - verbose (bool): Whether to print the optimization progress.
        - confidence_threshold (float): The threshold for objectness confidence.

        Returns:
        - lbd (float): The calibrated lambda value.
        """
        if self.lbd is not None:
            print("Replacing previously computed lambda")
        if preds.confidence_threshold is None:
            if confidence_threshold is None:
                raise ValueError(
                    "confidence_threshold must be set in the predictions or in the conformalizer"
                )
            else:
                preds.confidence_threshold = confidence_threshold

        risk_function = self._get_risk_function(
            preds=preds,
            alpha=alpha,
            objectness_threshold=preds.confidence_threshold,
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

    def conformalize(self, preds: ODPredictions) -> List[List[float]]:
        """
        Conformalize the object detection predictions.

        Parameters:
        - preds (ODPredictions): The object detection predictions.

        Returns:
        - conf_boxes (List[List[float]]): The conformalized bounding boxes.
        """
        if self.lbd is None:
            raise ValueError("Conformalizer must be calibrated before conformalizing.")
        conf_boxes = apply_margins(
            preds.pred_boxes, [self.lbd] * 4, mode=self.prediction_set
        )
        preds.confidence_threshold = preds.confidence_threshold
        preds.conf_boxes = conf_boxes
        return conf_boxes

    def evaluate(
        self, preds: ODPredictions, conf_boxes: List[List[float]], verbose: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluate the conformalized predictions.

        Parameters:
        - preds (ODPredictions): The object detection predictions.
        - conf_boxes (List[List[float]]): The conformalized bounding boxes.
        - verbose (bool): Whether to print the evaluation results.

        Returns:
        - safety (torch.Tensor): The safety scores.
        - set_sizes (torch.Tensor): The set sizes.
        """
        if self.lbd is None:
            raise ValueError("Conformalizer must be calibrated before evaluating.")
        true_boxes = preds.true_boxes
        conf_boxes = list(
            [
                (
                    x[y >= preds.confidence_threshold]
                    if len(x[y >= preds.confidence_threshold]) > 0
                    else x[None, y.argmax()]
                )
                for x, y in zip(conf_boxes, preds.confidence)
            ]
        )

        risk = compute_risk_box_level(
            conf_boxes,
            true_boxes,
            loss=self.loss,
        )
        safety = 1 - risk

        def compute_set_size(boxes: List[List[float]]) -> torch.Tensor:
            set_sizes = []
            for image_boxes in boxes:
                for box in image_boxes:
                    set_size = (box[2] - box[0]) * (box[3] - box[1])
                    set_size = set_size**0.5
                    set_sizes.append(set_size)
            set_sizes = torch.stack(set_sizes).ravel()
            return set_sizes

        set_sizes = compute_set_size(conf_boxes)
        if verbose:
            print(f"Safety = {safety}")
            print(f"Average set size = {torch.mean(set_sizes)}")
        return safety, set_sizes


class ObjectnessRiskConformalizer(RiskConformalizer):
    """
    A class that performs risk conformalization for localization tasks.
    """

    ACCEPTED_LOSSES = {"nb_boxes": ObjectnessLoss}

    def __init__(
        self,
        prediction_set: str = "additive",
        loss: str = "nb_boxes",
        optimizer: str = "binary_search",
    ):
        """
        Initialize the LocalizationRiskConformalizer.

        Parameters:
        - prediction_set (str): The type of prediction set to use. Must be one of ["additive", "multiplicative", "adaptative"].
        - loss (str): The type of loss to use. Must be one of ["pixelwise", "boxwise"].
        - optimizer (str): The type of optimizer to use. Must be one of ["binary_search", "gaussianprocess", "gpr", "kriging"].
        """
        super().__init__()
        if loss not in self.ACCEPTED_LOSSES:
            raise ValueError(
                f"loss {loss} not accepted, must be one of {self.ACCEPTED_LOSSES.keys()}"
            )
        self.loss_name = loss
        self.loss = self.ACCEPTED_LOSSES[loss]()

        self.prediction_set = prediction_set
        self.lbd = None
        if optimizer == "binary_search":
            self.optimizer = BinarySearchOptimizer()
        elif optimizer in ["gaussianprocess", "gpr", "kriging"]:
            self.optimizer = GaussianProcessOptimizer()
        else:
            raise ValueError(f"optimizer {optimizer} not accepted")

    def _get_risk_function(
        self, preds: ODPredictions, alpha: float, **kwargs
    ) -> Callable[[float], float]:
        """
        Get the risk function for risk conformalization.

        Parameters:
        - preds (ODPredictions): The object detection predictions.
        - alpha (float): The significance level.
        - objectness_threshold (float): The threshold for objectness confidence.

        Returns:
        - risk_function (Callable[[float], float]): The risk function.
        """

        def risk_function(lbd: float) -> float:
            """
            Compute the risk given a lambda value.

            Parameters:
            - lbd (float): The lambda value.

            Returns:
            - corrected_risk (float): The corrected risk.
            """
            conf_boxes = list(
                [
                    x[y >= 1 - lbd] if len(x[y >= 1 - lbd]) > 0 else x[None, y.argmax()]
                    for x, y in zip(preds.pred_boxes, preds.confidence)
                ]
            )
            risk = compute_risk_box_level(
                conf_boxes,
                preds.true_boxes,
                loss=self.loss,
            )
            n = len(preds)
            corrected_risk = self._correct_risk(
                risk=risk,
                n=n,
                B=self.loss.upper_bound,
            )
            return corrected_risk

        return risk_function

    def _correct_risk(self, risk: torch.Tensor, n: int, B: float) -> torch.Tensor:
        """
        Correct the risk using the number of predictions and the upper bound.

        Parameters:
        - risk (torch.Tensor): The risk tensor.
        - n (int): The number of predictions.
        - B (float): The upper bound.

        Returns:
        - corrected_risk (torch.Tensor): The corrected risk tensor.
        """
        return (n / (n + 1)) * torch.mean(risk) + B / (n + 1)

    def calibrate(
        self,
        preds: ODPredictions,
        alpha: float = 0.1,
        steps: int = 13,
        bounds: List[float] = [0, 1000],
        verbose: bool = True,
        confidence_threshold: float = None,
    ) -> float:
        """
        Calibrate the conformalizer.

        Parameters:
        - preds (ODPredictions): The object detection predictions.
        - alpha (float): The significance level.
        - steps (int): The number of steps for optimization.
        - bounds (List[float]): The bounds for optimization.
        - verbose (bool): Whether to print the optimization progress.
        - confidence_threshold (float): The threshold for objectness confidence.

        Returns:
        - lbd (float): The calibrated lambda value.
        """
        if self.lbd is not None:
            print("Replacing previously computed lambda")
        if preds.confidence_threshold is None:
            if confidence_threshold is None:
                raise ValueError(
                    "confidence_threshold must be set in the predictions or in the conformalizer"
                )
            else:
                preds.confidence_threshold = confidence_threshold

        risk_function = self._get_risk_function(
            preds=preds,
            alpha=alpha,
            objectness_threshold=preds.confidence_threshold,
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

    def conformalize(self, preds: ODPredictions) -> float:
        """
        Conformalize the object detection predictions.

        Parameters:
        - preds (ODPredictions): The object detection predictions.

        Returns:
        - conf_boxes (List[List[float]]): The conformalized bounding boxes.
        """
        if self.lbd is None:
            raise ValueError("Conformalizer must be calibrated before conformalizing.")
        preds.confidence_threshold = 1 - self.lbd
        return 1 - self.lbd

    def evaluate(
        self, preds: ODPredictions, conf_boxes: List[List[float]], verbose: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluate the conformalized predictions.

        Parameters:
        - preds (ODPredictions): The object detection predictions.
        - conf_boxes (List[List[float]]): The conformalized bounding boxes.
        - verbose (bool): Whether to print the evaluation results.

        Returns:
        - safety (torch.Tensor): The safety scores.
        - set_sizes (torch.Tensor): The set sizes.
        """
        if self.lbd is None:
            raise ValueError("Conformalizer must be calibrated before evaluating.")
        true_boxes = preds.true_boxes
        conf_boxes = list(
            [
                (
                    x[y >= preds.confidence_threshold]
                    if len(x[y >= preds.confidence_threshold]) > 0
                    else x[None, y.argmax()]
                )
                for x, y in zip(conf_boxes, preds.confidence)
            ]
        )

        risk = compute_risk_box_level(
            conf_boxes,
            true_boxes,
            loss=self.loss,
        )
        safety = 1 - risk

        def compute_set_size(boxes: List[List[float]]) -> torch.Tensor:
            set_sizes = []
            for image_boxes in boxes:
                for box in image_boxes:
                    set_size = (box[2] - box[0]) * (box[3] - box[1])
                    set_size = set_size**0.5
                    set_sizes.append(set_size)
            set_sizes = torch.stack(set_sizes).ravel()
            return set_sizes

        set_sizes = compute_set_size(conf_boxes)
        if verbose:
            print(f"Safety = {safety}")
            print(f"Average set size = {torch.mean(set_sizes)}")
        return safety, set_sizes


class ODClassificationConformalizer(Conformalizer):

    pass


class ODConformalizer(Conformalizer):
    """_summary_

    Args:
        Conformalizer (_type_): _description_

    Raises:
        ValueError: _description_
        ValueError: _description_
        ValueError: _description_
        ValueError: _description_
        ValueError: _description_
    """

    MULTIPLE_TESTING_CORRECTIONS = ["bonferroni"]
    BACKENDS = ["auto"]
    GUARANTEE_LEVELS = ["image", "object"]

    def __init__(
        self,
        backend: str = "auto",
        guarantee_level: str = "image",
        confidence_threshold: Optional[float] = None,
        multiple_testing_correction: Optional[str] = None,
        confidence_method: Union[ConfidenceConformalizer, str, None] = None,
        localization_method: Union[LocalizationConformalizer, str, None] = None,
        classification_method: Union[ClassificationConformalizer, str, None] = None,
        **kwargs,
    ):
        """
        Initialize the ODClassificationConformalizer object.

        Parameters:
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
                f"backend {backend} not accepted, must be one of {self.BACKENDS}"
            )
        self.backend = backend

        if guarantee_level not in self.GUARANTEE_LEVELS:
            raise ValueError(
                f"guarantee_level {guarantee_level} not accepted, must be one of {self.GUARANTEE_LEVELS}"
            )
        self.guarantee_level = guarantee_level

        if multiple_testing_correction is None:
            logger.warning(
                "No multiple_testing_correction provided, assuming no correction is needed. The explicit list of alphas is expected for calibration."
            )
            self.multiple_testing_correction = multiple_testing_correction
        elif multiple_testing_correction not in self.MULTIPLE_TESTING_CORRECTIONS:
            raise ValueError(
                f"multiple_testing_correction {multiple_testing_correction} not accepted, must be one of {self.MULTIPLE_TESTING_CORRECTIONS}"
            )
        else:
            logger.warning(
                "Multiple testing correction is set to Bonferroni, expecting a global alpha for calibration."
            )
            self.multiple_testing_correction = multiple_testing_correction

        if confidence_threshold is None and confidence_method is None:
            raise ValueError(
                "Either confidence_threshold or confidence_method must be set"
            )
        elif confidence_threshold is not None and confidence_method is not None:
            raise ValueError(
                "confidence_threshold and confidence_method are mutually exclusive"
            )

        # Confidence

        if isinstance(confidence_threshold, float):
            self.confidence_threshold = confidence_threshold
            self.confidence_conformalizer = None
            self.confidence_method = None
        elif isinstance(confidence_method, str):
            self.confidence_threshold = None
            self.confidence_conformalizer = ConfidenceConformalizer(
                method=confidence_method
            )
            self.confidence_method = confidence_method
        elif isinstance(confidence_method, ConfidenceConformalizer):
            self.confidence_threshold = None
            self.confidence_conformalizer = confidence_method
            self.confidence_method = confidence_method.method

        # Localization

        if isinstance(localization_method, str):
            self.localization_method = localization_method
            self.localization_conformalizer = LocalizationConformalizer(
                method=localization_method
            )
        elif isinstance(localization_method, LocalizationConformalizer):
            self.localization_conformalizer = localization_method
            self.localization_method = localization_method.method
        else:
            self.localization_conformalizer = None
            self.localization_method = None

        # Classification

        if isinstance(classification_method, str):
            self.classification_method = classification_method
            self.classification_conformalizer = ClassificationConformalizer(
                method=classification_method
            )
        elif isinstance(classification_method, ClassificationConformalizer):
            self.classification_conformalizer = classification_method
            self.classification_method = classification_method.method
        else:
            self.classification_conformalizer = None
            self.classification_method = None

    def calibrate(
        self,
        predictions: ODPredictions,
        global_alpha: Optional[float],
        alpha_confidence: Optional[float],
        alpha_localization: Optional[float],
        alpha_classification: Optional[float],
        verbose: bool = True,
    ) -> ODParameters:
        """
        Calibrates the conformalizers and returns the calibration results.

        Args:
            predictions (ODPredictions): The predictions to be calibrated.
            global_alpha (Optional[float]): The global alpha value for calibration. If multiple_testing_correction is None, individual alpha values will be used for each conformalizer.
            alpha_confidence (Optional[float]): The alpha value for the confidence conformalizer.
            alpha_localization (Optional[float]): The alpha value for the localization conformalizer.
            alpha_classification (Optional[float]): The alpha value for the classification conformalizer.
            verbose (bool, optional): Whether to print calibration information. Defaults to True.

        Returns:
            dict[str, Any]: A dictionary containing the calibration results, including target alpha values and estimated lambda values for each conformalizer.

        Raises:
            ValueError: If the multiple_testing_correction is not provided or is not valid.
            ValueError: If the global_alpha is not provided when using the Bonferroni multiple_testing_correction.
            ValueError: If explicit alpha values are provided when using the Bonferroni multiple_testing_correction.

        Note:
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
                    "No multiple_testing_correction provided, expecting an explicit alpha for each conformalizer. 'global_alpha' should be 'None'."
                )
            elif (
                (alpha_confidence is None != self.confidence_conformalizer is None)
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
                    "No multiple_testing_correction provided, expecting an explicit alpha for each conformalizer. Explicity alphas should be set only if there's a corresponding conformalizer."
                )
            else:
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
                    ]
                )
        elif self.multiple_testing_correction == "bonferroni":
            if global_alpha is None:
                raise ValueError(
                    "Bonferroni multiple_testing_correction provided, expecting a global alpha for calibration."
                )
            elif (
                alpha_classification is not None
                or alpha_localization is not None
                or alpha_confidence is not None
            ):
                raise ValueError(
                    "Bonferroni multiple_testing_correction provided, expecting a global alpha for calibration. Explicit alphas should be set to None."
                )
            else:
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
                f"multiple_testing_correction {self.multiple_testing_correction} not accepted, must be one of {self.MULTIPLE_TESTING_CORRECTIONS}. This check should have failed in the __init__ method."
            )

        # Confidence

        if self.confidence_conformalizer is not None:
            if verbose:
                logger.info("Calibrating Confidence Conformalizer")

            lambda_confidence = self.confidence_conformalizer.calibrate(
                predictions, alpha=alpha_confidence, verbose=verbose
            )
            # Unique to Confidence due to dependence
            self.confidence_conformalizer.conformalize(predictions)  # TODO: inplace ?
            self.confidence_threshold = predictions.confidence_threshold

            if verbose:
                logger.info(
                    f"Calibrated Confidence λ : {lambda_confidence}\n\t and associated Confidence Threshold : {predictions.confidence_threshold}"
                )
        else:
            predictions.confidence_threshold = self.confidence_threshold

        # Localization

        if self.localization_conformalizer is not None:
            if verbose:
                logger.info("Calibrating Localization Conformalizer")

            lambda_localization = self.localization_conformalizer.calibrate(
                predictions, alpha=alpha_localization, verbose=verbose
            )

            if verbose:
                logger.info("Calibrated Localization λ : {lambda_localization}")

        # Classification

        if self.classification_conformalizer is not None:
            if verbose:
                logger.info("Calibrating Classification Conformalizer")

            lambda_classification = self.classification_conformalizer.calibrate(
                predictions, alpha=alpha_classification, verbose=verbose
            )

            if verbose:
                logger.info("Calibrated Classification λ : {lambda_classification}")

        result = ODParameters(
            predictions_id=predictions.unique_id,
            global_alpha=global_alpha,
            alpha_confidence=alpha_confidence,
            alpha_localization=alpha_localization,
            alpha_classification=alpha_classification,
            lambda_confidence=lambda_confidence,
            lambda_localization=lambda_localization,
            lambda_classification=lambda_classification,
        )

        # Saving the last parameters id to conformalize the predictions
        self._last_parameters_id = result.unique_id

        return result

    def conformalize(
        self, predictions: ODPredictions, parameters=Optional[ODParameters]
    ) -> ODResults:
        """
        Conformalize the given predictions.

        Args:
            preds (ODPredictions): The predictions to be conformalized.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing the conformalized predictions.
        """

        if parameters is not None:
            parameters_unique_id = parameters.unique_id
            lambda_confidence = parameters.lambda_confidence
            lambda_localization = parameters.lambda_localization
            lambda_classification = parameters.lambda_classification
        else:
            parameters_unique_id = self._last_parameters_id
            lambda_confidence = None
            lambda_localization = None
            lambda_classification = None

        if self.confidence_conformalizer is not None:
            self.confidence_conformalizer.conformalize(
                predictions, replace_lambda=lambda_confidence
            )
        else:
            # TODO what here ?
            predictions.confidence_threshold = self.confidence_threshold

        if self.localization_conformalizer is not None:
            conf_boxes = self.localization_conformalizer.conformalize(
                predictions, replace_lambda=lambda_localization
            )
        else:
            conf_boxes = None

        if self.classification_conformalizer is not None:
            conf_cls = self.classification_conformalizer.conformalize(
                predictions, replace_lambda=lambda_classification
            )
        else:
            conf_cls = None

        results = ODResults(
            predictions_id=predictions.unique_id,
            parameters_id=parameters_unique_id,
            conf_boxes=conf_boxes,
            conf_cls=conf_cls,
        )

        return results
    
    def evaluate


######## NEW OD Generalization ####################################################


# ODConformalizer, a class that can handle doing conformalization on OD predictions for Localization (or not), Objectness (or not) and Classification (or not)
# It's a rewritting of the classes below, to avoid having a class for each possible combination. Here we can either conformalize each of the three cases, or use a constant/normal approach


class OOOOOODConformalizer(Conformalizer):

    def conformalize(self, preds: ODPredictions):
        """
        Perform conformalization on the predictions.

        Parameters:
        - preds: The ODPredictions object containing the predictions.

        Returns:
        - conf_boxes: The conformalized bounding boxes.
        - conf_cls: The conformalized classification scores.
        """
        if self.obj_conformalizer is not None:
            self.obj_conformalizer.conformalize(preds)
        else:
            preds.confidence_threshold = self.confidence_threshold

        if self.loc_conformalizer is not None:
            conf_boxes = self.loc_conformalizer.conformalize(preds)
        else:
            conf_boxes = None

        if self.cls_conformalizer is not None:
            conf_cls = get_conf_cls_for_od(preds, self.cls_conformalizer)
        else:
            conf_cls = None

        return conf_boxes, conf_cls

    def evaluate(
        self,
        preds: ODPredictions,
        conf_boxes: list,
        conf_cls: list,
        verbose: bool = True,
    ):
        """
        Evaluate the conformalizers.

        Parameters:
        - preds: The ODPredictions object containing the predictions.
        - conf_boxes: The conformalized bounding boxes.
        - conf_cls: The conformalized classification scores.
        - verbose: Whether to print the evaluation results.
        """
        if self.loc_conformalizer is not None:
            coverage_loc, set_size_loc = self.loc_conformalizer.evaluate(
                preds, conf_boxes, verbose=False
            )
        else:
            coverage_loc, set_size_loc = None, None
        if self.obj_conformalizer is not None:
            coverage_obj, set_size_obj = self.obj_conformalizer.evaluate(
                preds, conf_boxes, verbose=False
            )
        else:
            coverage_obj, set_size_obj = None, None
        if self.cls_conformalizer is not None:
            coverage_cls, set_size_cls = evaluate_cls_conformalizer(
                preds, conf_cls, self.cls_conformalizer, verbose=False
            )
        else:
            coverage_cls, set_size_cls = None, None
        global_coverage = compute_global_coverage(
            preds=preds,
            conf_boxes=conf_boxes,
            conf_cls=conf_cls,
            confidence=self.obj_conformalizer is not None,
            cls=self.cls_conformalizer is not None,
            localization=self.loc_conformalizer is not None,
        )
        if verbose:
            print("Confidence:")
            print(f"\t Coverage: {torch.mean(coverage_obj):.2f}")
            print(f"\t Mean Set Size: {torch.mean(set_size_obj):.2f}")
            print("Localization:")
            print(f"\t Coverage: {torch.mean(coverage_loc):.2f}")
            print(f"\t Mean Set Size: {torch.mean(set_size_loc):.2f}")
            print("Classification:")
            print(f"\t Coverage: {torch.mean(coverage_cls):.2f}")
            print(f"\t Mean Set Size: {torch.mean(set_size_cls):.2f}")
            print("Global:")
            print(f"\t Coverage: {torch.mean(global_coverage):.2f}")

        return (
            coverage_obj,
            coverage_loc,
            coverage_cls,
            set_size_obj,
            set_size_loc,
            set_size_cls,
            global_coverage,
        )


# class ODRiskConformalizer(ODConformalizer, RiskConformalizer):
#     """
#     ODRiskConformalizer class is used for risk conformalization in object detection tasks.
#     It combines the functionalities of ODConformalizer and RiskConformalizer classes.

#     Parameters:
#         localization_method (Union[LocalizationRiskConformalizer, str, None]): The method used for localization risk conformalization.
#             It can be an instance of LocalizationRiskConformalizer class, a string representing the loss function, or None.
#         objectness_method (Union[ObjectnessConformalizer, str, None]): The method used for objectness conformalization.
#             It can be an instance of ObjectnessConformalizer class, a string representing the method, or None.
#         classification_method (Union[ClassificationConformalizer, str, None]): The method used for classification conformalization.
#             It can be an instance of ClassificationConformalizer class, a string representing the method, or None.
#         multiple_testing_correction (str): The method used for multiple testing correction.
#             It should be one of the accepted multiple testing correction methods.
#         confidence_threshold (float): The confidence threshold. It is ignored if objectness_method is not None.
#         **kwargs: Additional keyword arguments.

#     Attributes:
#         loc_conformalizer (LocalizationRiskConformalizer): The localization risk conformalizer instance.
#         localization_method (str): The localization method.
#         obj_conformalizer (ObjectnessConformalizer): The objectness conformalizer instance.
#         objectness_method (str): The objectness method.
#         cls_conformalizer (ClassificationConformalizer): The classification conformalizer instance.
#         classification_method (str): The classification method.
#         multiple_testing_correction (str): The multiple testing correction method.
#         confidence_threshold (float): The confidence threshold.

#     Raises:
#         ValueError: If the multiple_testing_correction is not one of the accepted methods.

#     """

#     def __init__(
#         self,
#         localization_method: Union[LocalizationRiskConformalizer, str, None] = None,
#         objectness_method: Union[ObjectnessConformalizer, str, None] = None,
#         classification_method: Union[ClassificationConformalizer, str, None] = None,
#         multiple_testing_correction: str = "bonferroni",
#         confidence_threshold: float = None,
#         **kwargs,
#     ):
#         if isinstance(localization_method, str):
#             self.localization_method = localization_method
#             self.loc_conformalizer = LocalizationRiskConformalizer(
#                 loss=localization_method
#             )
#         elif isinstance(localization_method, LocalizationRiskConformalizer):
#             self.loc_conformalizer = localization_method
#             self.localization_method = localization_method.method
#         else:
#             self.loc_conformalizer = None
#             self.localization_method = None

#         if isinstance(objectness_method, str):
#             self.objectness_method = objectness_method
#             self.obj_conformalizer = ObjectnessConformalizer(method=objectness_method)
#         elif isinstance(objectness_method, ObjectnessConformalizer):
#             self.obj_conformalizer = objectness_method
#             self.objectness_method = objectness_method.method
#         else:
#             self.obj_conformalizer = None
#             self.objectness_method = None

#         if isinstance(classification_method, str):
#             self.classification_method = classification_method
#             self.cls_conformalizer = ClassificationConformalizer(
#                 method=classification_method
#             )
#         elif isinstance(classification_method, ClassificationConformalizer):
#             self.cls_conformalizer = classification_method
#             self.classification_method = classification_method.method
#         else:
#             self.cls_conformalizer = None
#             self.classification_method = None

#         if multiple_testing_correction not in self.MULTIPLE_TESTING_CORRECTIONS:
#             raise ValueError(
#                 f"multiple_testing_correction {multiple_testing_correction} not accepted, should be one of {self.MULTIPLE_TESTING_CORRECTIONS}"
#             )
#         self.multiple_testing_correction = multiple_testing_correction
#         self.confidence_threshold = confidence_threshold
#         if self.confidence_threshold is not None and self.obj_conformalizer is not None:
#             # TODO: replace by warnings
#             print(
#                 "Warning: confidence_threshold is ignored if objectness_method is not None"
#             )

#     def evaluate(
#         self,
#         preds: ODPredictions,
#         conf_boxes: list,
#         conf_cls: list,
#         verbose: bool = True,
#     ):
#         """
#         Evaluate the conformalizers.

#         Parameters:
#         - preds: The ODPredictions object containing the predictions.
#         - conf_boxes: The conformalized bounding boxes.
#         - conf_cls: The conformalized classification scores.
#         - verbose: Whether to print the evaluation results.
#         """
#         if self.loc_conformalizer is not None:
#             coverage_loc, set_size_loc = self.loc_conformalizer.evaluate(
#                 preds, conf_boxes, verbose=False
#             )
#         else:
#             coverage_loc, set_size_loc = None, None
#         if self.obj_conformalizer is not None:
#             coverage_obj, set_size_obj = self.obj_conformalizer.evaluate(
#                 preds, conf_boxes, verbose=False
#             )
#         else:
#             coverage_obj, set_size_obj = None, None
#         if self.cls_conformalizer is not None:
#             coverage_cls, set_size_cls = evaluate_cls_conformalizer(
#                 preds, conf_cls, self.cls_conformalizer, verbose=False
#             )
#         else:
#             coverage_cls, set_size_cls = None, None
#         global_coverage = compute_global_coverage(
#             preds=preds,
#             conf_boxes=conf_boxes,
#             conf_cls=conf_cls,
#             confidence=self.obj_conformalizer is not None,
#             cls=self.cls_conformalizer is not None,
#             localization=self.loc_conformalizer is not None,
#             loss=(
#                 self.loc_conformalizer.loss
#                 if self.loc_conformalizer is not None
#                 else None
#             ),
#         )
#         if verbose:
#             if self.obj_conformalizer is not None:
#                 print("Confidence:")
#                 print(f"\t Coverage: {torch.mean(coverage_obj):.2f}")
#                 print(f"\t Mean Set Size: {torch.mean(set_size_obj):.2f}")
#             if self.loc_conformalizer is not None:
#                 print("Localization:")
#                 print(f"\t Coverage: {torch.mean(coverage_loc):.2f}")
#                 print(f"\t Mean Set Size: {torch.mean(set_size_loc):.2f}")
#             if self.cls_conformalizer is not None:
#                 print("Classification:")
#                 print(f"\t Coverage: {torch.mean(coverage_cls):.2f}")
#                 print(f"\t Mean Set Size: {torch.mean(set_size_cls):.2f}")
#             if (
#                 sum(x is not None for x in [coverage_obj, coverage_loc, coverage_cls])
#                 > 1
#             ):
#                 print("Global:")
#                 print(f"\t Coverage: {torch.mean(global_coverage):.2f}")

#         return (
#             coverage_obj,
#             coverage_loc,
#             coverage_cls,
#             set_size_obj,
#             set_size_loc,
#             set_size_cls,
#             global_coverage,
#         )


###################################################################
##################### SEQ CRC #####################################
###################################################################


class SeqLocalizationRiskConformalizer(RiskConformalizer):
    """
    A class that performs risk conformalization for localization tasks.
    """

    ACCEPTED_LOSSES = {
        "pixelwise": PixelWiseRecallLoss,
        "boxwise": BoxWiseRecallLoss,
        "classboxwise": ClassBoxWiseRecallLoss,
    }

    def __init__(
        self,
        guarantee_level: str = "image",  # options include "image" and "box"
        prediction_set: str = "additive",
        loss: Optional[str] = None,
        optimizer: str = "binary_search",
    ):
        """
        Initialize the LocalizationRiskConformalizer.

        Parameters:
        - prediction_set (str): The type of prediction set to use. Must be one of ["additive", "multiplicative", "adaptative"].
        - loss (str): The type of loss to use. Must be one of ["pixelwise", "boxwise"].
        - optimizer (str): The type of optimizer to use. Must be one of ["binary_search", "gaussianprocess", "gpr", "kriging"].
        """
        super().__init__()

        self.guarantee_level = guarantee_level

        if loss not in self.ACCEPTED_LOSSES:
            raise ValueError(
                f"loss {loss} not accepted, must be one of {self.ACCEPTED_LOSSES.keys()}"
            )
        self.loss_name = loss
        self.loss = self.ACCEPTED_LOSSES[loss]()

        if prediction_set not in ["additive", "multiplicative", "adaptative"]:
            raise ValueError(f"prediction_set {prediction_set} not accepted")
        self.prediction_set = prediction_set
        self.lbd = None
        if optimizer == "binary_search":
            self.optimizer = BinarySearchOptimizer()
        elif optimizer in ["gaussianprocess", "gpr", "kriging"]:
            self.optimizer = GaussianProcessOptimizer()
        else:
            raise ValueError(f"optimizer {optimizer} not accepted")

    def _get_risk_function(
        self,
        preds: ODPredictions,
        alpha: float,
        objectness_threshold: float,
        override_B=None,
        conf_cls=None,
        **kwargs,
    ) -> Callable[[float], torch.Tensor]:
        """
        Get the risk function for risk conformalization.

        Parameters:
        - preds (ODPredictions): The object detection predictions.
        - alpha (float): The significance level.
        - objectness_threshold (float): The threshold for objectness confidence.

        Returns:
        - risk_function (Callable[[float], float]): The risk function.
        """
        pred_boxes_filtered = list(
            [
                (
                    x[y >= objectness_threshold]
                    if len(x[y >= objectness_threshold]) > 0
                    else x[None, y.argmax()]
                )
                for x, y in zip(preds.pred_boxes, preds.confidence)
            ]
        )

        def risk_function(lbd: float) -> torch.Tensor:
            """
            Compute the risk given a lambda value.

            Parameters:
            - lbd (float): The lambda value.

            Returns:
            - corrected_risk (float): The corrected risk.
            """
            conf_boxes = apply_margins(
                pred_boxes_filtered, [lbd, lbd, lbd, lbd], mode=self.prediction_set
            )
            if self.guarantee_level == "box":
                if self.loss_name == "classboxwise":
                    risk = compute_risk_cls_box_level(
                        conf_boxes,
                        conf_cls,
                        preds.true_boxes,
                        preds.true_cls,
                        loss=self.loss,
                    )
                else:
                    risk = compute_risk_box_level(
                        conf_boxes,
                        preds.true_boxes,
                        loss=self.loss,
                    )
            elif self.guarantee_level == "image":
                risk = compute_risk_image_level(
                    conf_boxes,
                    preds.true_boxes,
                    loss=self.loss,
                )

            n = len(preds)
            corrected_risk = self._correct_risk(
                risk=risk,
                n=n,
                B=override_B if override_B is not None else self.loss.upper_bound,
            )
            return corrected_risk

        return risk_function

    def _correct_risk(self, risk: torch.Tensor, n: int, B: float) -> torch.Tensor:
        """
        Correct the risk using the number of predictions and the upper bound.

        Parameters:
        - risk (torch.Tensor): The risk tensor.
        - n (int): The number of predictions.
        - B (float): The upper bound.

        Returns:
        - corrected_risk (torch.Tensor): The corrected risk tensor.
        """
        return (n / (n + 1)) * torch.mean(risk) + B / (n + 1)

    def calibrate(
        self,
        preds: ODPredictions,
        alpha: float = 0.1,
        steps: int = 13,
        bounds: List[float] = [0, 1000],
        verbose: bool = True,
        confidence_threshold: Union[float, None] = None,
        override_B=None,
        conf_cls=None,
    ) -> float:
        """
        Calibrate the conformalizer.

        Parameters:
        - preds (ODPredictions): The object detection predictions.
        - alpha (float): The significance level.
        - steps (int): The number of steps for optimization.
        - bounds (List[float]): The bounds for optimization.
        - verbose (bool): Whether to print the optimization progress.
        - confidence_threshold (float): The threshold for objectness confidence.

        Returns:
        - lbd (float): The calibrated lambda value.
        """
        if self.lbd is not None:
            print("Replacing previously computed lambda")
        if preds.confidence_threshold is None:
            if confidence_threshold is None:
                raise ValueError(
                    "confidence_threshold must be set in the predictions or in the conformalizer"
                )
            else:
                preds.confidence_threshold = confidence_threshold

        risk_function = self._get_risk_function(
            preds=preds,
            alpha=alpha,
            objectness_threshold=preds.confidence_threshold,
            override_B=override_B,
            conf_cls=conf_cls,
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

    def conformalize(self, preds: ODPredictions) -> List[List[float]]:
        """
        Conformalize the object detection predictions.

        Parameters:
        - preds (ODPredictions): The object detection predictions.

        Returns:
        - conf_boxes (List[List[float]]): The conformalized bounding boxes.
        """
        if self.lbd is None:
            raise ValueError("Conformalizer must be calibrated before conformalizing.")
        conf_boxes = apply_margins(
            preds.pred_boxes, [self.lbd] * 4, mode=self.prediction_set
        )
        preds.confidence_threshold = preds.confidence_threshold
        return conf_boxes

    def evaluate(
        self,
        preds: ODPredictions,
        conf_boxes: List[List[float]],
        conf_cls=None,
        verbose: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluate the conformalized predictions.

        Parameters:
        - preds (ODPredictions): The object detection predictions.
        - conf_boxes (List[List[float]]): The conformalized bounding boxes.
        - verbose (bool): Whether to print the evaluation results.

        Returns:
        - safety (torch.Tensor): The safety scores.
        - set_sizes (torch.Tensor): The set sizes.
        """
        if self.lbd is None:
            raise ValueError("Conformalizer must be calibrated before evaluating.")
        if conf_boxes is None:
            raise ValueError("Predictions must be conformalized before evaluating.")
        true_boxes = preds.true_boxes
        conf_boxes = list(
            [
                (
                    x[y >= preds.confidence_threshold]
                    if len(x[y >= preds.confidence_threshold]) > 0
                    else x[None, y.argmax()]
                )
                for x, y in zip(conf_boxes, preds.confidence)
            ]
        )

        if self.loss_name == "classboxwise":
            risk = compute_risk_cls_box_level(
                conf_boxes,
                conf_cls,
                preds.true_boxes,
                preds.true_cls,
                loss=self.loss,
            )
        else:
            risk = compute_risk_box_level(
                conf_boxes,
                preds.true_boxes,
                loss=self.loss,
            )
        safety = 1 - risk

        def compute_set_size(boxes: List[List[float]]) -> torch.Tensor:
            set_sizes = []
            for image_boxes in boxes:
                for box in image_boxes:
                    set_size = (box[2] - box[0]) * (box[3] - box[1])
                    set_size = set_size**0.5
                    set_sizes.append(set_size)
            set_sizes = torch.stack(set_sizes).ravel()
            return set_sizes

        set_sizes = compute_set_size(conf_boxes)
        if verbose:
            print(f"Safety = {safety}")
            print(f"Average set size = {torch.mean(set_sizes)}")
        return safety, set_sizes


class SeqODRiskConformalizer(ODConformalizer, RiskConformalizer):

    def __init__(
        self,
        guarantee_level: str = "image",  # options include "image" or "box"
        localization_method: Union[LocalizationRiskConformalizer, str, None] = None,
        objectness_method: Union[ObjectnessConformalizer, str, None] = None,
        classification_method: Union[ClassificationConformalizer, str, None] = None,
        confidence_threshold: Optional[float] = None,
        multiple_testing_correction: str = "bonferroni",
        **kwargs,
    ):
        self.guarantee_level = guarantee_level

        if isinstance(localization_method, str):
            self.localization_method = localization_method
            self.loc_conformalizer = SeqLocalizationRiskConformalizer(
                loss=localization_method
            )
        elif isinstance(localization_method, SeqLocalizationRiskConformalizer):
            self.loc_conformalizer = localization_method
            self.localization_method = localization_method.method
        else:
            self.loc_conformalizer = None
            self.localization_method = None

        if isinstance(objectness_method, str):
            self.objectness_method = objectness_method
            self.obj_conformalizer = ObjectnessConformalizer(method=objectness_method)
        elif isinstance(objectness_method, ObjectnessConformalizer):
            self.obj_conformalizer = objectness_method
            self.objectness_method = objectness_method.method
        else:
            self.obj_conformalizer = None
            self.objectness_method = None

        if isinstance(classification_method, str):
            self.classification_method = classification_method
            self.cls_conformalizer = ClassificationConformalizer(
                method=classification_method
            )
        elif isinstance(classification_method, ClassificationConformalizer):
            self.cls_conformalizer = classification_method
            self.classification_method = classification_method.method
        else:
            self.cls_conformalizer = None
            self.classification_method = None

        self.multiple_testing_correction = multiple_testing_correction
        self.confidence_threshold = confidence_threshold
        if self.confidence_threshold is not None and self.obj_conformalizer is not None:
            # TODO: replace by warnings
            print(
                "Warning: confidence_threshold is ignored if objectness_method is not None"
            )

    def calibrate(
        self, preds: ODPredictions, alpha: float = 0.1, verbose: bool = True
    ) -> Tuple[
        Optional[Sequence[float]], Optional[torch.Tensor], Optional[torch.Tensor]
    ]:

        if self.multiple_testing_correction == "bonferroni":
            if (
                self.loc_conformalizer is not None
                and self.cls_conformalizer is not None
            ):
                real_alpha_cls = alpha * 0.7
                real_alpha_loc = alpha * 0.3
                real_alpha = alpha * 0.3
            else:
                real_alpha = 0.5 * alpha
                real_alpha_cls = alpha
                real_alpha_loc = alpha
        elif self.multiple_testing_correction == "none":
            real_alpha = alpha
        else:
            raise ValueError(
                f"multiple_testing_correction {self.multiple_testing_correction} not accepted, should be one of {self.MULTIPLE_TESTING_CORRECTIONS}"
            )

        if self.obj_conformalizer is not None:
            quantile_obj_confidence_minus = self.obj_conformalizer.calibrate(
                preds,
                alpha=real_alpha,  # real_alpha,
                verbose=verbose,
                override_B=True,
            )
            minus_conf_threshold = 1 - quantile_obj_confidence_minus
            quantile_obj_confidence = self.obj_conformalizer.calibrate(
                preds,
                alpha=real_alpha,  # real_alpha,
                verbose=verbose,
            )
            confidence_threshold = 1 - quantile_obj_confidence
            preds.confidence_threshold = minus_conf_threshold
        else:
            quantile_obj_confidence = 1 - self.confidence_threshold
            confidence_threshold = self.confidence_threshold
            preds.confidence_threshold = confidence_threshold

        if self.loc_conformalizer is not None:
            quantile_localization = self.loc_conformalizer.calibrate(
                preds, alpha=real_alpha_loc, verbose=verbose  # real_alpha,
            )
        else:
            quantile_localization = None
        if self.cls_conformalizer is not None:
            cls_preds = get_classif_preds_from_od_preds(preds)
            quantile_classif, score_cls = self.cls_conformalizer.calibrate(
                cls_preds, alpha=real_alpha_cls, verbose=verbose  # real_alpha,
            )
        else:
            quantile_classif, score_cls = None, None

        preds.confidence_threshold = confidence_threshold

        if verbose:
            print(f"Quantiles")
            if self.obj_conformalizer is not None:
                print(f"Confidence: {quantile_obj_confidence}")
            if self.loc_conformalizer is not None:
                print(f"Localization: {quantile_localization}")
            if self.cls_conformalizer is not None:
                print(f"Classification: {quantile_classif}")

        # TODO: do this better, here we want to allow exclusion of objectness for metrics

        # TODO: future move to dictionary for better handling
        return quantile_localization, quantile_obj_confidence, quantile_classif

    def evaluate(
        self,
        preds: ODPredictions,
        conf_boxes: list,
        conf_cls: list,
        verbose: bool = True,
    ):
        """
        Evaluate the conformalizers.

        Parameters:
        - preds: The ODPredictions object containing the predictions.
        - conf_boxes: The conformalized bounding boxes.
        - conf_cls: The conformalized classification scores.
        - verbose: Whether to print the evaluation results.
        """
        if self.loc_conformalizer is not None:
            coverage_loc, set_size_loc = self.loc_conformalizer.evaluate(
                preds, conf_boxes, verbose=False
            )
        else:
            coverage_loc, set_size_loc = None, None
        if self.obj_conformalizer is not None:
            coverage_obj, set_size_obj = self.obj_conformalizer.evaluate(
                preds, conf_boxes, verbose=False
            )
        else:
            coverage_obj, set_size_obj = None, None
        if self.cls_conformalizer is not None:
            coverage_cls, set_size_cls = evaluate_cls_conformalizer(
                preds, conf_cls, self.cls_conformalizer, verbose=False
            )
        else:
            coverage_cls, set_size_cls = None, None
        global_coverage = compute_global_coverage(
            preds=preds,
            conf_boxes=conf_boxes,
            conf_cls=conf_cls,
            confidence=False,
            cls=self.cls_conformalizer is not None,
            localization=self.loc_conformalizer is not None,
            loss=(
                self.loc_conformalizer.loss
                if self.loc_conformalizer is not None
                else None
            ),
        )
        if verbose:
            if (
                self.obj_conformalizer is not None
                and coverage_obj is not None
                and set_size_obj is not None
            ):
                print("Confidence:")
                print(f"\t Coverage: {torch.mean(coverage_obj):.2f}")
                print(f"\t Mean Set Size: {torch.mean(set_size_obj):.2f}")
            if (
                self.loc_conformalizer is not None
                and coverage_loc is not None
                and set_size_loc is not None
            ):
                print("Localization:")
                print(f"\t Coverage: {torch.mean(coverage_loc):.2f}")
                print(f"\t Mean Set Size: {torch.mean(set_size_loc):.2f}")
            if (
                self.cls_conformalizer is not None
                and coverage_cls is not None
                and set_size_cls is not None
            ):
                print("Classification:")
                print(f"\t Coverage: {torch.mean(coverage_cls):.2f}")
                print(f"\t Mean Set Size: {torch.mean(set_size_cls):.2f}")
            if (
                sum(x is not None for x in [coverage_obj, coverage_loc, coverage_cls])
                > 1
            ):
                print("Global:")
                print(f"\t Coverage: {torch.mean(global_coverage):.2f}")

        return (
            coverage_obj,
            coverage_loc,
            coverage_cls,
            set_size_obj,
            set_size_loc,
            set_size_cls,
            global_coverage,
        )


class SeqGlobalODRiskConformalizer(ODRiskConformalizer):

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
        self.obj_conformalizer = ObjectnessConformalizer(method=objectness_method)

        self.classification_method = classification_method

        self.cls_conformalizer = ClassificationConformalizer(
            method=self.classification_method
        )

        self.loc_conformalizer = SeqLocalizationRiskConformalizer(
            prediction_set="additive", loss="classboxwise"
        )
        self.fix_cls = fix_cls
        self.multiple_testing_correction = "none"
        self.confidence_threshold = confidence_threshold
        if self.confidence_threshold is not None and self.obj_conformalizer is not None:
            # TODO: replace by warnings
            print(
                "Warning: confidence_threshold is ignored if objectness_method is not None"
            )

    def calibrate(
        self, preds: ODPredictions, alpha: float = 0.1, verbose: bool = True
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
                f"multiple_testing_correction {self.multiple_testing_correction} not accepted, should be one of {self.MULTIPLE_TESTING_CORRECTIONS}"
            )

        # Confidence

        quantile_obj_confidence_minus = self.obj_conformalizer.calibrate(
            preds,
            alpha=real_alpha * 0.5,
            verbose=verbose,
            override_B=True,
        )
        minus_conf_threshold = 1 - quantile_obj_confidence_minus
        print(
            "Minus confidence threshold",
            minus_conf_threshold,
            quantile_obj_confidence_minus,
        )
        quantile_obj_confidence = self.obj_conformalizer.calibrate(
            preds,
            alpha=real_alpha * 0.5,
            verbose=verbose,
        )
        confidence_threshold = 1 - quantile_obj_confidence
        preds.confidence_threshold = minus_conf_threshold
        print(
            "Plus confidence threshold", confidence_threshold, quantile_obj_confidence
        )

        # Classif
        if self.fix_cls:
            # lbd minus
            cls_preds = get_classif_preds_from_od_preds(preds)
            quantile_classif, score_cls_min = self.cls_conformalizer.calibrate(
                cls_preds, alpha=real_alpha * 0.5, verbose=verbose, lbd_minus=True
            )
            conf_cls = get_conf_cls_for_od(preds, self.cls_conformalizer)
            # for the real lbd+
            quantile_classif, score_cls_plus = self.cls_conformalizer.calibrate(
                cls_preds, alpha=real_alpha * 0.5, verbose=verbose
            )
        else:
            cls_preds = get_classif_preds_from_od_preds(preds)
            quantile_classif, score_cls = self.cls_conformalizer.calibrate(
                cls_preds, alpha=real_alpha * 0.5, verbose=verbose
            )
            conf_cls = get_conf_cls_for_od(preds, self.cls_conformalizer)
            score_cls_min, score_cls_plus = score_cls, score_cls

        # Localization

        quantile_localization = self.loc_conformalizer.calibrate(
            preds, conf_cls=conf_cls, alpha=real_alpha, verbose=verbose
        )

        preds.confidence_threshold = confidence_threshold

        if verbose:
            print(f"Quantiles")
            if self.obj_conformalizer is not None:
                print(f"Confidence: {quantile_obj_confidence}")
            if self.loc_conformalizer is not None:
                print(f"Localization: {quantile_localization}")
            if self.cls_conformalizer is not None:
                print(f"Classification: {quantile_classif}")

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
        preds: ODPredictions,
        conf_boxes: list,
        conf_cls: list,
        verbose: bool = True,
    ):
        """
        Evaluate the conformalizers.

        Parameters:
        - preds: The ODPredictions object containing the predictions.
        - conf_boxes: The conformalized bounding boxes.
        - conf_cls: The conformalized classification scores.
        - verbose: Whether to print the evaluation results.
        """
        if self.loc_conformalizer is not None:
            coverage_loc, set_size_loc = self.loc_conformalizer.evaluate(
                preds, conf_boxes, conf_cls=conf_cls, verbose=False
            )
        else:
            coverage_loc, set_size_loc = None, None
        if self.obj_conformalizer is not None:
            coverage_obj, set_size_obj = self.obj_conformalizer.evaluate(
                preds, conf_boxes, verbose=False
            )
        else:
            coverage_obj, set_size_obj = None, None
        if self.cls_conformalizer is not None:
            coverage_cls, set_size_cls = evaluate_cls_conformalizer(
                preds, conf_cls, self.cls_conformalizer, verbose=False
            )
        else:
            coverage_cls, set_size_cls = None, None

        if self.loc_conformalizer.loss_name == "classboxwise":
            loss = BoxWiseRecallLoss()
        global_coverage = compute_global_coverage(
            preds=preds,
            conf_boxes=conf_boxes,
            conf_cls=conf_cls,
            confidence=self.obj_conformalizer is not None,
            cls=self.cls_conformalizer is not None,
            localization=self.loc_conformalizer is not None,
            loss=loss,
        )
        if verbose:
            print("Confidence:")
            print(f"\t Coverage: {torch.mean(coverage_obj):.2f}")
            print(f"\t Mean Set Size: {torch.mean(set_size_obj):.2f}")
            print("Classification:")
            print(f"\t Coverage: {torch.mean(coverage_cls):.2f}")
            print(f"\t Mean Set Size: {torch.mean(set_size_cls):.2f}")
            print("Localization:")
            print(f"\t Coverage: {torch.mean(coverage_loc):.2f}")
            print(f"\t Mean Set Size: {torch.mean(set_size_loc):.2f}")
            print("Global:")
            print(f"\t Coverage: {torch.mean(global_coverage):.2f}")

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


class AsymptoticLocalizationObjectnessRiskConformalizer(RiskConformalizer):
    """
    A class that performs risk conformalization for object detection predictions with asymptotic localization and objectness.

    Args:
        prediction_set (str): The type of prediction set to use. Must be one of "additive", "multiplicative", or "adaptative".
        localization_loss (str): The type of localization loss to use. Must be one of "pixelwise" or "boxwise".
        optimizer (str): The type of optimizer to use. Must be one of "gaussianprocess", "gpr", "kriging", "mc", or "montecarlo".

    Attributes:
        ACCEPTED_LOSSES (dict): A dictionary mapping accepted localization losses to their corresponding classes.
        loss_name (str): The name of the localization loss.
        loss (Loss): An instance of the localization loss class.
        prediction_set (str): The type of prediction set.
        lbd (tuple): The calibrated lambda values.

    Methods:
        _get_risk_function: Returns the risk function for optimization.
        _correct_risk: Corrects the risk using the number of predictions and the upper bound of the loss.
        calibrate: Calibrates the conformalizer using the given predictions.
        conformalize: Conformalizes the predictions using the calibrated lambda values.
        evaluate: Evaluates the conformalized predictions.

    """

    ACCEPTED_LOSSES = {"pixelwise": PixelWiseRecallLoss, "boxwise": BoxWiseRecallLoss}

    def __init__(
        self,
        prediction_set: str = "additive",
        localization_loss: str = "boxwise",
        optimizer: str = "gpr",
    ):
        super().__init__()
        if localization_loss not in self.ACCEPTED_LOSSES:
            raise ValueError(
                f"loss {localization_loss} not accepted, must be one of {self.ACCEPTED_LOSSES.keys()}"
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
                f"optimizer {optimizer} not accepted in multidim, currently only gpr and mc"
            )

    def _get_risk_function(self, preds, alpha, **kwargs):
        """
        Returns the risk function for optimization.

        Args:
            preds (ODPredictions): The object detection predictions.
            alpha (float): The significance level.

        Returns:
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
                    for x, y in zip(preds.pred_boxes, preds.confidence)
                ]
            )
            conf_boxes = apply_margins(
                pred_boxes_filtered,
                [lbd_loc, lbd_loc, lbd_loc, lbd_loc],
                mode=self.prediction_set,
            )
            risk = compute_risk_box_level(
                conf_boxes,
                preds.true_boxes,
                loss=self.loss,
            )
            n = len(preds)
            corrected_risk = self._correct_risk(
                risk=risk,
                n=n,
                B=self.loss.upper_bound,
            )
            return corrected_risk

        return risk_function

    def _correct_risk(self, risk, n, B):
        """
        Corrects the risk using the number of predictions and the upper bound of the loss.

        Args:
            risk (torch.Tensor): The risk values.
            n (int): The number of predictions.
            B (float): The upper bound of the loss.

        Returns:
            torch.Tensor: The corrected risk values.

        """
        return (n / (n + 1)) * torch.mean(risk) + B / (n + 1)

    def calibrate(
        self,
        preds: ODPredictions,
        alpha: float = 0.1,
        steps: int = 13,
        bounds: list = [(0, 500), (0.0, 1.0)],
        verbose: bool = True,
    ):
        """
        Calibrates the conformalizer using the given predictions.

        Args:
            preds (ODPredictions): The object detection predictions.
            alpha (float): The significance level.
            steps (int): The number of optimization steps.
            bounds (list): The bounds for the optimization variables.
            verbose (bool): Whether to print verbose output.

        Returns:
            tuple: The calibrated lambda values.

        Raises:
            ValueError: If the conformalizer has already been calibrated.

        """
        if self.lbd is not None:
            print("Replacing previously computed lambda")
        risk_function = self._get_risk_function(
            preds=preds,
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

    def conformalize(self, preds: ODPredictions):
        """
        Conformalizes the predictions using the calibrated lambda values.

        Args:
            preds (ODPredictions): The object detection predictions.

        Returns:
            list: The conformalized bounding boxes.

        Raises:
            ValueError: If the conformalizer has not been calibrated.

        """
        if self.lbd is None:
            raise ValueError("Conformalizer must be calibrated before conformalizing.")
        conf_boxes = apply_margins(
            preds.pred_boxes, [self.lbd[0]] * 4, mode=self.prediction_set
        )
        preds.confidence_threshold = 1 - self.lbd[1]
        preds.conf_boxes = conf_boxes
        return conf_boxes

    def evaluate(self, preds: ODPredictions, conf_boxes: list, verbose: bool = True):
        """
        Evaluates the conformalized predictions.

        Args:
            preds (ODPredictions): The object detection predictions.
            conf_boxes (list): The conformalized bounding boxes.
            verbose (bool): Whether to print verbose output.

        Returns:
            tuple: The evaluation results.

        Raises:
            ValueError: If the conformalizer has not been calibrated or the predictions have not been conformalized.

        """
        if self.lbd is None:
            raise ValueError("Conformalizer must be calibrated before evaluating.")
        if preds.conf_boxes is None:
            raise ValueError("Predictions must be conformalized before evaluating.")
        coverage_obj = []
        set_size_obj = []
        for true_boxes, confidence in zip(preds.true_boxes, preds.confidence):
            cov = (
                1
                if len(true_boxes) <= (confidence >= preds.confidence_threshold).sum()
                else 0
            )
            set_size = (confidence >= preds.confidence_threshold).sum()
            set_size_obj.append(set_size)
            coverage_obj.append(cov)
        if verbose:
            print(
                f"Confidence Treshold {preds.confidence_threshold}, Coverage = {torch.mean(coverage_obj)}, Median set size = {torch.mean(set_size_obj)}"
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
        true_boxes = preds.true_boxes
        conf_boxes = list(
            [
                (
                    x[y >= preds.confidence_threshold]
                    if len(x[y >= preds.confidence_threshold]) > 0
                    else x[None, y.argmax()]
                )
                for x, y in zip(conf_boxes, preds.confidence)
            ]
        )
        set_size_loc = compute_set_size(conf_boxes)
        risk = compute_risk_box_level(
            conf_boxes,
            true_boxes,
            loss=self.loss,
        )
        safety = 1 - risk
        if verbose:
            print(f"Safety = {safety}")
            print(f"Average set size = {torch.mean(set_size_loc)}")
        global_coverage = compute_global_coverage(
            preds=preds, also_conf=True, also_cls=False, loss=self.loss
        )
        return coverage_obj, safety, set_size_obj, set_size_loc, global_coverage
