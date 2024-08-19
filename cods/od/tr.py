from typing import Callable, Optional, Tuple, Union

import torch

from cods.base.optim import BinarySearchOptimizer, GaussianProcessOptimizer
from cods.base.tr import ToleranceRegion
from cods.classif.tr import ClassificationToleranceRegion
from cods.od.data import ODPredictions
from cods.od.loss import BoxWiseRecallLoss, ObjectnessLoss, PixelWiseRecallLoss
from cods.od.metrics import compute_global_coverage
from cods.od.utils import (
    apply_margins,
    compute_risk_box_level,
    compute_risk_image_level,
    evaluate_cls_conformalizer,
    get_classif_preds_from_od_preds,
    get_conf_cls_for_od,
)


class LocalizationToleranceRegion(ToleranceRegion):
    """
    Tolerance region for object localization tasks.
    """

    ACCEPTED_LOSSES = {
        "pixelwise": PixelWiseRecallLoss,
        "boxwise": BoxWiseRecallLoss,
    }

    def __init__(
        self,
        prediction_set: str = "additive",
        loss: Union[str, None] = None,
        optimizer: str = "binary_search",
        inequality: Union[str, Callable] = "binomial_inverse_cdf",
    ):
        """
        Initialize the LocalizationToleranceRegion.

        Args:
            prediction_set (str): The type of prediction set to use. Must be one of "additive", "multiplicative", or "adaptative".
            loss (str, None): The type of loss to use. Must be one of "pixelwise" or "boxwise".
            optimizer (str): The type of optimizer to use. Must be one of "binary_search", "gaussianprocess", "gpr", or "kriging".
            inequality (str, Callable): The type of inequality function to use. Must be one of "binomial_inverse_cdf" or a custom callable.

        Raises:
            ValueError: If the loss or optimizer is not accepted.
        """
        super().__init__(inequality=inequality)
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
        delta: float,
        **kwargs,
    ):
        """
        Get the risk function for calibration.

        Args:
            preds (ODPredictions): The object detection predictions.
            alpha (float): The significance level.
            objectness_threshold (float): The objectness threshold.
            delta (float): The confidence level.

        Returns:
            Callable: The risk function.
        """
        pred_boxes_filtered = list(
            [
                x[y >= objectness_threshold]
                for x, y in zip(preds.pred_boxes, preds.confidence)
            ]
        )
        n = sum([len(x) for x in preds.true_boxes])

        def risk_function(lbd):
            conf_boxes = apply_margins(
                pred_boxes_filtered,
                [lbd, lbd, lbd, lbd],
                mode=self.prediction_set,
            )
            risk = compute_risk_box_level(
                conf_boxes,
                preds.true_boxes,
                loss=self.loss,
            )
            corrected_risk = self._correct_risk(
                risk=risk,
                n=n,
                delta=delta,
            )
            return corrected_risk

        return risk_function

    def _correct_risk(
        self,
        risk: float,
        n: int,
        delta: float,
    ):
        """
        Correct the risk using the inequality function.

        Args:
            risk (float): The risk value.
            n (int): The number of samples.
            delta (float): The confidence level.

        Returns:
            float: The corrected risk value.
        """
        return self.f_inequality(
            Rhat=risk,
            n=torch.tensor(n, dtype=torch.float).cuda(),
            delta=torch.tensor(delta, dtype=torch.float).cuda(),
        )

    def calibrate(
        self,
        preds: ODPredictions,
        alpha: float = 0.1,
        delta: float = 0.1,
        steps: int = 13,
        bounds: Tuple[float, float] = (0, 1),
        verbose: bool = True,
        confidence_threshold: Union[float, None] = None,
    ):
        """
        Calibrate the tolerance region.

        Args:
            preds (ODPredictions): The object detection predictions.
            alpha (float): The significance level.
            delta (float): The confidence level.
            steps (int): The number of steps for optimization.
            bounds (Tuple[float, float]): The bounds for optimization.
            verbose (bool): Whether to print verbose output.
            confidence_threshold (float, None): The confidence threshold.

        Returns:
            float: The calibrated lambda value.
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
            delta=delta,
        )

        lbd = self.optimizer.optimize(
            objective_function=risk_function,
            alpha=alpha,
            bounds=bounds,
            steps=steps,
            verbose=verbose,
        )
        self.lbd = lbd
        self.confidence_threshold = preds.confidence_threshold
        return lbd

    def conformalize(self, preds: ODPredictions):
        """
        Conformalize the object detection predictions.

        Args:
            preds (ODPredictions): The object detection predictions.

        Returns:
            list: The conformalized bounding boxes.
        """
        if self.lbd is None:
            raise ValueError(
                "Conformalizer must be calibrated before conformalizing."
            )
        conf_boxes = apply_margins(
            preds.pred_boxes, [self.lbd] * 4, mode=self.prediction_set
        )
        preds.confidence_threshold = self.confidence_threshold
        preds.conf_boxes = conf_boxes
        return conf_boxes

    def evaluate(self, preds: ODPredictions, conf_boxes: list, verbose=True):
        """
        Evaluate the conformalized object detection predictions.

        Args:
            preds (ODPredictions): The object detection predictions.
            conf_boxes (list): The conformalized bounding boxes.
            verbose (bool): Whether to print verbose output.

        Returns:
            Tuple[float, torch.Tensor]: The safety and set sizes.
        """
        if self.lbd is None:
            raise ValueError(
                "Conformalizer must be calibrated before evaluating."
            )
        if preds.conf_boxes is None:
            raise ValueError(
                "Predictions must be conformalized before evaluating."
            )

        conf_boxes = list(
            [
                x[y >= preds.confidence_threshold]
                for x, y in zip(conf_boxes, preds.confidence)
            ]
        )

        risk = compute_risk_box_level(
            conf_boxes,
            preds.true_boxes,
            loss=self.loss,
        )
        safety = 1 - risk

        def compute_set_size(boxes):
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


class ConfidenceToleranceRegion(ToleranceRegion):
    """
    Tolerance region for object confidence tasks.
    """

    ACCEPTED_LOSSES = {"box_number": ObjectnessLoss}

    def __init__(
        self,
        loss: str = "box_number",
        inequality: str = "binomial_inverse_cdf",
        optimizer: str = "binary_search",
    ) -> None:
        """
        Initialize the ConfidenceToleranceRegion.

        Args:
            loss (str): The type of loss to use. Must be one of "box_number".
            inequality (str): The type of inequality function to use. Must be one of "binomial_inverse_cdf" or a custom callable.
            optimizer (str): The type of optimizer to use. Must be one of "binary_search", "gaussianprocess", "gpr", or "kriging".

        Raises:
            ValueError: If the loss or optimizer is not accepted.
        """
        super().__init__(inequality=inequality)
        self.lbd = None
        if loss not in self.ACCEPTED_LOSSES:
            raise ValueError(
                f"Loss {loss} not supported. Choose from {self.ACCEPTED_LOSSES}."
            )
        self.loss_name = loss
        self.loss = self.ACCEPTED_LOSSES[loss]()
        if optimizer == "binary_search":
            self.optimizer = BinarySearchOptimizer()
        elif optimizer in ["gaussianprocess", "gpr", "kriging"]:
            self.optimizer = GaussianProcessOptimizer()
        else:
            raise ValueError(f"optimizer {optimizer} not accepted")

    def calibrate(
        self,
        preds: ODPredictions,
        alpha: float = 0.1,
        delta: float = 0.1,
        steps: int = 13,
        bounds: Tuple[float, float] = (0, 1),
        verbose: bool = True,
        confidence_threshold: Optional[float] = None,
    ) -> float:
        """
        Calibrate the tolerance region.

        Args:
            preds (ODPredictions): The object detection predictions.
            alpha (float): The significance level.
            delta (float): The confidence level.
            steps (int): The number of steps for optimization.
            bounds (Tuple[float, float]): The bounds for optimization.
            verbose (bool): Whether to print verbose output.
            confidence_threshold (float, None): The confidence threshold.

        Returns:
            float: The calibrated lambda value.
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
        self._n_classes = preds.n_classes
        if self.loss is None:
            self.loss = self.accepted_methods[self.loss_name]()
        risk_function = self._get_risk_function(
            preds=preds,
            alpha=alpha,
            delta=delta,
        )

        lbd = self.optimizer.optimize(
            objective_function=risk_function,
            alpha=alpha,
            bounds=bounds,
            steps=steps,
            verbose=verbose,
        )
        self.lbd = lbd
        self.confidence_threshold = confidence_threshold
        return lbd

    def _get_risk_function(
        self, preds: ODPredictions, alpha: float, delta: float, **kwargs
    ) -> Callable:
        """
        Get the risk function for calibration.

        Args:
            preds (ODPredictions): The object detection predictions.
            alpha (float): The significance level.
            delta (float): The confidence level.

        Returns:
            Callable: The risk function.
        """
        n = len(preds)

        def risk_function(lbd: float) -> float:
            risk = []
            for true_boxes, confidence in zip(
                preds.true_boxes, preds.confidence
            ):
                score = self.loss(len(true_boxes), confidence, lbd)
                risk.append(score)
            risk = torch.stack(risk).ravel()
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
        risk: float,
        n: int,
        delta: float,
    ) -> float:
        """
        Correct the risk using the inequality function.

        Args:
            risk (float): The risk value.
            n (int): The number of samples.
            delta (float): The confidence level.

        Returns:
            float: The corrected risk value.
        """
        return self.f_inequality(
            Rhat=risk,
            n=torch.tensor(n, dtype=torch.float).cuda(),
            delta=torch.tensor(delta, dtype=torch.float).cuda(),
        )

    def conformalize(
        self, preds: ODPredictions, verbose: bool = True, **kwargs
    ) -> float:
        """
        Conformalize the predictions.

        Args:
            preds (ODPredictions): The object detection predictions.
            verbose (bool): Whether to print verbose output.

        Returns:
            float: The confidence threshold.
        """
        if self.lbd is None:
            raise ValueError(
                "Conformalizer must be calibrated before conformalizing."
            )
        preds.confidence_threshold = 1 - self.lbd
        return preds.confidence_threshold

    def evaluate(
        self,
        preds: ODPredictions,
        conf_boxes: list,
        verbose: bool = True,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluate the tolerance region.

        Args:
            preds (ODPredictions): The object detection predictions.
            conf_boxes (list): The confidence boxes.
            verbose (bool): Whether to print verbose output.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The coverage and set sizes.
        """
        if self.lbd is None:
            raise ValueError(
                "Tolerance Region must be calibrated before evaluating."
            )
        if preds.confidence_threshold is None:
            raise ValueError(
                "Predictions must be conformalized before evaluating."
            )
        coverage = []
        set_sizes = []
        for true_boxes, confidence in zip(preds.true_boxes, preds.confidence):
            cov = (
                torch.ones(1).float().cuda()
                if len(true_boxes)
                <= (confidence >= preds.confidence_threshold).sum()
                else torch.zeros(1).float().cuda()
            )
            set_size = (confidence >= preds.confidence_threshold).float().sum()
            set_sizes.append(set_size)
            coverage.append(cov)
        coverage = torch.stack(coverage).ravel()
        set_sizes = torch.stack(set_sizes).ravel()
        if verbose:
            print(
                f"Confidence Treshold {preds.confidence_threshold}, Coverage = {torch.mean(coverage)}, Median set size = {torch.median(set_sizes)}"
            )
        return coverage, set_sizes


class ODToleranceRegion(ToleranceRegion):
    MULTIPLE_TESTING_CORRECTIONS = ["bonferroni"]

    def __init__(
        self,
        localization_loss: Union[
            None, str, LocalizationToleranceRegion
        ] = "boxwise",
        confidence_loss: Union[
            None, str, ConfidenceToleranceRegion
        ] = "box_number",
        classification_loss: Union[
            None, str, ClassificationToleranceRegion
        ] = "lac",
        inequality="binomial_inverse_cdf",
        margins=1,  # where to compute 1, 2 or 4 margins with bonferroni corrections
        prediction_set="additive",
        multiple_testing_correction: str = "bonferroni",
        confidence_threshold: Optional[float] = None,
        **kwargs,
    ):
        # TODO: add option of putting a loss object directly
        if isinstance(localization_loss, str):
            self.loc_conformalizer = LocalizationToleranceRegion(
                loss=localization_loss,
                inequality=inequality,
                prediction_set=prediction_set,
                **kwargs,
            )
        elif isinstance(localization_loss, LocalizationToleranceRegion):
            self.loc_conformalizer = localization_loss
        elif localization_loss is None:
            self.loc_conformalizer = None
        else:
            raise ValueError(
                "localization_loss must be a string or a LocalizationToleranceRegion"
            )
        if isinstance(confidence_loss, str):
            self.obj_conformalizer = ConfidenceToleranceRegion(
                loss=confidence_loss, inequality="binomial_inverse_cdf"
            )
        elif isinstance(confidence_loss, ConfidenceToleranceRegion):
            self.obj_conformalizer = confidence_loss
        elif confidence_loss is None:
            self.obj_conformalizer = None
        else:
            raise ValueError(
                "confidence_loss must be a string or a ConfidenceToleranceRegion"
            )
        if isinstance(classification_loss, str):
            self.cls_conformalizer = ClassificationToleranceRegion(
                loss=classification_loss, inequality="binomial_inverse_cdf"
            )
        elif isinstance(classification_loss, ClassificationToleranceRegion):
            self.cls_conformalizer = classification_loss
        elif classification_loss is None:
            self.cls_conformalizer = None
        else:
            raise ValueError(
                "classification_loss must be a string or a ClassificationToleranceRegion"
            )
        self.localization_loss = localization_loss
        self.margins = margins
        self.kwargs = kwargs
        self.inequality_name = inequality
        self.prediction_set = prediction_set

        self.multiple_testing_correction = multiple_testing_correction
        self.confidence_threshold = confidence_threshold
        if (
            self.confidence_threshold is not None
            and self.obj_conformalizer is not None
        ):
            # TODO: replace by warnings
            print(
                "Warning: confidence_threshold is ignored if objectness_method is not None"
            )

    def calibrate(
        self,
        preds,
        alpha=0.1,
        delta=0.1,
        steps=13,
        bounds=[0, 1],
        verbose=True,
    ):
        # set real_alpha
        if self.multiple_testing_correction == "bonferroni":
            # divide alpha by number of conformalizers that aren't None
            real_alpha = alpha / sum(
                x is not None
                for x in [
                    self.loc_conformalizer,
                    self.obj_conformalizer,
                    self.cls_conformalizer,
                ]
            )
            real_delta = delta / sum(
                x is not None
                for x in [
                    self.loc_conformalizer,
                    self.obj_conformalizer,
                    self.cls_conformalizer,
                ]
            )
        else:
            raise ValueError(
                f"multiple_testing_correction {self.multiple_testing_correction} not accepted, should be one of {self.MULTIPLE_TESTING_CORRECTIONS}"
            )

        # calibrate each conformalizer
        if self.obj_conformalizer is not None:
            q_obj = self.obj_conformalizer.calibrate(
                preds,
                alpha=real_alpha,
                delta=real_delta,
                steps=steps,
                bounds=(0.0, 1.0),
                verbose=verbose,
            )
            confidence_threshold = 1 - q_obj
        else:
            confidence_threshold = self.confidence_threshold
        preds.confidence_threshold = confidence_threshold

        if self.loc_conformalizer is not None:
            q_loc = self.loc_conformalizer.calibrate(
                preds,
                alpha=real_alpha,
                delta=real_delta,
                steps=steps,
                bounds=bounds,
                confidence_threshold=preds.confidence_threshold,
                verbose=verbose,
            )
        if self.cls_conformalizer is not None:
            # TODO: stocker cls_preds in OD Preds (rather put this check and call of function in the preds class)
            cls_preds = get_classif_preds_from_od_preds(preds)
            q_cls = self.cls_conformalizer.calibrate(
                cls_preds,
                alpha=real_alpha,
                delta=real_delta,
                steps=steps * 2,
                bounds=[0, 1],  # high precision required here
                objectness_threshold=preds.confidence_threshold,
                verbose=verbose,
            )

        if verbose:
            print(f"Lambdas")
            if self.obj_conformalizer is not None:
                print(f"Confidence: {q_obj}")
            if self.loc_conformalizer is not None:
                print(f"Localization: {q_loc}")
            if self.cls_conformalizer is not None:
                print(f"Classification: {q_cls}")

        return q_loc, q_obj, q_cls

    def conformalize(self, preds: ODPredictions):
        if self.obj_conformalizer is not None:
            self.obj_conformalizer.conformalize(preds)
        else:
            preds.confidence_threshold = self.confidence_threshold
        if self.loc_conformalizer is not None:
            conf_boxes = self.loc_conformalizer.conformalize(preds)
        else:
            # Or should we return something else ?
            conf_boxes = None
        if self.cls_conformalizer is not None:
            conf_cls = get_conf_cls_for_od(preds, self.cls_conformalizer)
        else:
            # Or should we return something else ?
            conf_cls = None
        return conf_boxes, conf_cls

    def evaluate(
        self, preds: ODPredictions, conf_boxes, conf_cls, verbose=True
    ):
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
            # loss=(
            #    self.loc_conformalizer.loss
            #    if self.loc_conformalizer is not None
            #    else None
            # ),
            cls=self.cls_conformalizer is not None,
            localization=self.loc_conformalizer is not None,
        )
        if verbose:
            print("Confidence:")
            print(
                f"\t Coverage: {torch.mean(coverage_obj):.2f}"
            )  # Format coverage_obj with 2 decimal places
            print(
                f"\t Mean Set Size: {torch.mean(set_size_obj):.2f}"
            )  # Format set_size_obj with 2 decimal places
            print("Localization:")
            print(
                f"\t Coverage: {torch.mean(coverage_loc):.2f}"
            )  # Format coverage_obj with 2 decimal places
            print(
                f"\t Mean Set Size: {torch.mean(set_size_loc):.2f}"
            )  # Format set_size_obj with 2 decimal places
            print("Classification:")
            print(
                f"\t Coverage: {torch.mean(coverage_cls):.2f}"
            )  # Format coverage_obj with 2 decimal places
            print(
                f"\t Mean Set Size: {torch.mean(set_size_cls):.2f}"
            )  # Format set_size_obj with 2 decimal places
            print("Global:")
            print(
                f"\t Coverage: {torch.mean(global_coverage):.2f}"
            )  # Format coverage_obj with 2 decimal places

        return (
            coverage_obj,
            coverage_loc,
            coverage_cls,
            set_size_obj,
            set_size_loc,
            set_size_cls,
            global_coverage,
        )
