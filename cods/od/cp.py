from __future__ import annotations

import logging
from types import MappingProxyType

import torch

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
    BoxCountRecallConfidenceLoss,
    BoxCountThresholdConfidenceLoss,
    BoxCountTwosidedConfidenceLoss,
    BoxWiseIoULoss,
    BoxWisePrecisionLoss,
    BoxWiseRecallLoss,
    ClassificationLossWrapper,
    ODBinaryClassificationLoss,
    ODLoss,
    PixelWiseRecallLoss,
    ThresholdedBoxDistanceConfidenceLoss,
    ThresholdedRecallLoss,
)
from cods.od.metrics import ODEvaluator, compute_global_coverage
from cods.od.optim import (
    FirstStepMonotonizingOptimizer,
    SecondStepMonotonizingOptimizer,
)
from cods.od.score import (
    MinAdditiveSignedAssymetricHausdorffNCScore,
    MinMultiplicativeSignedAssymetricHausdorffNCScore,
    UnionAdditiveSignedAssymetricHausdorffNCScore,
    UnionMultiplicativeSignedAssymetricHausdorffNCScore,
)
from cods.od.utils import (
    apply_margins,
    compute_risk_object_level,
    match_predictions_to_true_boxes,
)

logger = logging.getLogger("cods")
# FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
FORMAT = (
    "[%(asctime)s:%(levelname)s:%(filename)s:%(module)s:%(lineno)s - %(funcName)s ] %(message)s"
)
logging.basicConfig(format=FORMAT, datefmt="%Y-%m-%d %H:%M:%S")

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

    BACKENDS = ("auto", "cp", "crc")

    accepted_methods = MappingProxyType(
        {
            "min-hausdorff-additive": MinAdditiveSignedAssymetricHausdorffNCScore,
            "min-hausdorff-multiplicative": MinMultiplicativeSignedAssymetricHausdorffNCScore,
            "union-hausdorff-additive": UnionAdditiveSignedAssymetricHausdorffNCScore,
            "union-hausdorff-multiplicative": UnionMultiplicativeSignedAssymetricHausdorffNCScore,
        }
    )

    PREDICTION_SETS = ("additive", "multiplicative", "adaptive")
    LOSSES = MappingProxyType(
        {
            "pixelwise": PixelWiseRecallLoss,
            "boxwise": BoxWiseRecallLoss,
            "thresholded": ThresholdedRecallLoss,
            "boxwise-precision": BoxWisePrecisionLoss,
            "boxwise-iou": BoxWiseIoULoss,
        }
    )
    OPTIMIZERS = MappingProxyType(
        {
            "binary_search": BinarySearchOptimizer,
            "gaussian_process": GaussianProcessOptimizer,
        }
    )
    GUARANTEE_LEVELS = ("image", "object")

    def __init__(
        self,
        loss: str | ODLoss,
        prediction_set: str,
        guarantee_level: str,
        matching_function: str,
        number_of_margins: int = 1,  # where to compute 1, 2 or 4 margins with bonferroni corrections
        optimizer: str | Optimizer | None = None,
        backend: str = "auto",
        device: str = "cpu",
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
        self.device = device
        self.matching_function = matching_function
        if isinstance(loss, str) and loss in self.LOSSES:
            self.loss = self.LOSSES[loss](device=self.device)
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
            raise NotImplementedError(
                "Not implemented currently for localization",
            )

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

        self.lambda_localization = None

    def calibrate(
        self,
        predictions: ODPredictions,
        alpha: float,
        steps: int = 13,
        overload_confidence_threshold: float | None = None,
        verbose: bool = True,
    ) -> float:
        """Calibrate the conformalizer.

        Parameters
        ----------
        - predictions (ODPredictions): The object detection predictions.
        - alpha (float): The significance level.
        - steps (int): The number of steps for optimization.
        - verbose (bool): Whether to print the optimization progress.
        - confidence_threshold (float): The threshold for objectness confidence.

        Returns
        -------
        - lbd (float): The calibrated lambda value.

        """
        if self.lambda_localization is not None:
            logger.info("Replacing previously computed λ")

        def build_predictions(
            matched_pred_boxes_i,
            matched_pred_cls_i,
            lbd,
        ):
            conf_boxes = apply_margins(
                [matched_pred_boxes_i],
                [lbd, lbd, lbd, lbd],
                mode=self.prediction_set,
            )[0]
            # TODO: rethink this, why do we regenerate it everytime, make it default somehwo
            n_classes = len(predictions.pred_cls[0][0].squeeze())
            conf_cls = [
                torch.arange(n_classes)[None, ...].to(self.device)
                for pred_cls_i_j in matched_pred_cls_i
            ]
            return conf_boxes, conf_cls

        self.optimizer2 = SecondStepMonotonizingOptimizer()
        lambda_localization = self.optimizer2.optimize(
            predictions,
            build_predictions,
            overload_confidence_threshold=overload_confidence_threshold,
            loss=self.loss,
            matching_function=self.matching_function,
            alpha=alpha,
            device=self.device,
            B=1,
            lower_bound=0,
            upper_bound=1000 if self.prediction_set == "additive" else 100,
            steps=steps,
            epsilon=1e-9,
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
        parameters: ODParameters | None = None,
        verbose: bool = True,
    ) -> list[torch.Tensor]:
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
        if parameters is not None and parameters.lambda_localization is not None:
            if verbose:
                logger.info("Using λ for localization from parameters")
            lambda_localization = parameters.lambda_localization
        else:
            if verbose:
                logger.info("Using previous λ for localization")
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
        return apply_margins(
            predictions.pred_boxes,
            lambdas,
            mode=self.prediction_set,
        )


class ConfidenceConformalizer(Conformalizer):
    """ """

    ACCEPTED_LOSSES = MappingProxyType(
        {
            "box_count_threshold": BoxCountThresholdConfidenceLoss,
            "box_count_recall": BoxCountRecallConfidenceLoss,
            "box_thresholded_distance": ThresholdedBoxDistanceConfidenceLoss,
            "box_count_twosided_recall": BoxCountTwosidedConfidenceLoss,
        }
    )

    def __init__(
        self,
        guarantee_level: str,
        matching_function: str,
        loss: str = "box_count_threshold",
        other_losses: list | None = None,
        optimizer: str = "binary_search",
        device="cpu",
    ):
        """ """
        super().__init__()
        self.device = device
        if loss not in self.ACCEPTED_LOSSES:
            raise ValueError(
                f"loss {loss} not accepted, must be one of {self.ACCEPTED_LOSSES.keys()}",
            )
        self.loss_name = loss
        self.matching_function = matching_function
        self.other_losses = other_losses
        self.loss = self.ACCEPTED_LOSSES[loss](
            # other_losses=other_losses,
            device=self.device,
        )
        self.guarantee_level = guarantee_level

        if guarantee_level == "object":
            raise ValueError("Not implemented currently for Confidence")
            # self.risk_function = compute_risk_object_level
        if guarantee_level == "image":
            # self.risk_function = compute_risk_image_level_confidence
            pass

        if optimizer == "binary_search":
            self.optimizer = BinarySearchOptimizer()
        elif optimizer in ["gaussianprocess", "gpr", "kriging"]:
            self.optimizer = GaussianProcessOptimizer()
        else:
            raise ValueError(f"optimizer {optimizer} not accepted")

        self.lambda_minus = None
        self.lambda_plus = None

    def calibrate(
        self,
        predictions: ODPredictions,
        alpha: float = 0.1,
        steps: int = 13,
        bounds: list[float] | None = None,
        verbose: bool = True,
    ) -> tuple[float, float]:
        """ """
        if bounds is None:
            bounds = [0, 1]
        if self.lambda_plus is not None:
            logger.info("Replacing previously computed λ")

        logger.debug("Optimizing for lambda_plus")
        self.optimizer2_plus = FirstStepMonotonizingOptimizer()
        lambda_plus = self.optimizer2_plus.optimize(
            predictions,
            self.loss,
            self.other_losses[0],
            self.other_losses[1],
            self.matching_function,
            alpha,
            self.device,
            B=1,
            verbose=False,
        )
        logger.debug("Optimizing for lambda_minus")
        self.optimizer2_minus = FirstStepMonotonizingOptimizer()
        lambda_minus = self.optimizer2_minus.optimize(
            predictions,
            self.loss,
            self.other_losses[0],
            self.other_losses[1],
            self.matching_function,
            alpha,
            self.device,
            B=0,
            verbose=False,
        )
        self.lambda_plus = lambda_plus
        self.lambda_minus = lambda_minus
        return lambda_minus, lambda_plus

    def conformalize(
        self,
        predictions: ODPredictions,
        verbose: bool = True,
    ) -> float:
        """Conformalize the object detection predictions.

        Parameters
        ----------
        - predictions (ODPredictions): The object detection predictions.

        Returns
        -------
        - conf_boxes (List[List[float]]): The conformalized bounding boxes.

        """
        if self.lambda_minus is None or self.lambda_plus is None:
            raise ValueError(
                "Conformalizer must be calibrated before conformalizing.",
            )
        predictions.confidence_threshold = 1 - self.lambda_plus
        predictions.matching = None
        return 1 - self.lambda_plus


class ODClassificationConformalizer(ClassificationConformalizer):
    """ """

    BACKENDS = ("auto", "cp", "crc")
    GUARANTEE_LEVELS = ("image", "object")
    OPTIMIZERS = MappingProxyType(
        {
            "binary_search": BinarySearchOptimizer,
            "gaussian_process": GaussianProcessOptimizer,
        }
    )
    LOSSES = MappingProxyType(
        {
            "binary": ODBinaryClassificationLoss,
        }
    )

    def __init__(
        self,
        matching_function: str,
        loss="binary",
        prediction_set="lac",
        backend="auto",
        guarantee_level="image",
        optimizer="binary_search",
        device="cpu",
        # TODO(leo) remove if nonessential: **kwargs,
    ):
        """ """
        self.matching_function = matching_function
        # TODO(leo): tmp
        preprocess = "softmax"
        super().__init__(
            method=prediction_set,
            preprocess=preprocess,
            device=device,
        )
        if loss not in self.LOSSES:
            raise ValueError(
                f"loss {loss} not accepted, must be one of {self.LOSSES.keys()}",
            )
        if guarantee_level not in self.GUARANTEE_LEVELS:
            raise ValueError(
                f"guarantee_level {guarantee_level} not accepted, must be one of {self.GUARANTEE_LEVELS}",
            )
        self.guarantee_level = guarantee_level
        if self.guarantee_level == "object":
            raise NotImplementedError(
                "Not implemented currently for classification",
            )
        # if self.guarantee_level == "object":
        #     self.risk_function = compute_risk_object_level
        # elif self.guarantee_level == "image":
        #     self.risk_function = compute_risk_image_level

        if backend not in self.BACKENDS:
            raise ValueError(
                f"backend {backend} not accepted, must be one of {self.BACKENDS}",
            )
        if backend == "auto":
            self.backend = "crc"
            logger.info("Defaulting to CRC backend")
        if backend == "cp":
            raise NotImplementedError("CRC backend is not supported yet")

        self.backend = backend
        self._backend_loss = ODBinaryClassificationLoss()
        self.loss = ClassificationLossWrapper(
            self._backend_loss,
            device=self.device,
        )
        self.lambda_classification = None

        if optimizer not in self.OPTIMIZERS:
            raise ValueError(
                f"optimizer {optimizer} not accepted, must be one of {self.OPTIMIZERS}",
            )
        self.optimizer = self.OPTIMIZERS[optimizer]()

    def calibrate(
        self,
        predictions: ODPredictions,
        alpha: float,
        bounds: list[float] | None = None,
        steps: int = 40,
        verbose: bool = True,
        overload_confidence_threshold: float | None = None,
    ) -> torch.Tensor:
        if bounds is None:
            bounds = [0, 1]
        if self.lambda_classification is not None:
            logger.info("Replacing previously computed λ")

        logger.warning(
            "Currently considering that there is only one matching prediction to each true box for classification pruposes. To add later how to aggregate if multiple preidctions matched.",
        )

        def build_predictions(
            matched_pred_boxes_i,
            matched_pred_cls_i,
            lbd,
        ):
            conf_boxes = matched_pred_boxes_i
            n_classes = len(predictions.pred_cls[0][0].squeeze())
            if self._score_function is None:
                self._score_function = self.ACCEPTED_METHODS[self.method](
                    n_classes,
                )

            # TODO(leo): filter for confidence here!
            def get_conf_cls():
                conf_cls_i = []
                for _j, pred_cls_i_j in enumerate(matched_pred_cls_i):
                    conf_cls_i_j = self._score_function.get_set(
                        pred_cls=pred_cls_i_j,
                        quantile=lbd,
                    )
                    # conf_cls_i_j = torch.where(
                    #     pred_cls_i_j >= 1 - lbd
                    # )[
                    #     0
                    # ]
                    conf_cls_i.append(conf_cls_i_j)
                return conf_cls_i

            conf_cls = get_conf_cls()
            return conf_boxes, conf_cls

        self.optimizer2 = SecondStepMonotonizingOptimizer()
        lambda_classification = self.optimizer2.optimize(
            predictions,
            build_predictions,
            overload_confidence_threshold=overload_confidence_threshold,
            loss=self.loss,
            matching_function=self.matching_function,
            alpha=alpha,
            device=self.device,
            B=1,
            lower_bound=0,
            upper_bound=1,
            steps=25,
            epsilon=1e-10,
            verbose=verbose,
        )
        if verbose:
            logger.info(
                f"Calibrated λ for classification: {lambda_classification}",
            )

        self.lambda_classification = lambda_classification
        return lambda_classification

    def conformalize(
        self,
        predictions: ODPredictions,
        verbose: bool = True,
    ) -> list:
        # TODO: add od parameters to function signature
        # NO MATCHING HERE
        def get_conf_cls():
            conf_cls = []
            for _i, pred_cls_i in enumerate(predictions.pred_cls):
                conf_cls_i = []
                for _j, pred_cls_i_j in enumerate(pred_cls_i):
                    conf_cls_i_j = self._score_function.get_set(
                        pred_cls=pred_cls_i_j,
                        quantile=self.lambda_classification,
                    )
                    conf_cls_i.append(conf_cls_i_j)
                # not all same size : conf_cls_i = torch.stack(conf_cls_i)
                conf_cls.append(conf_cls_i)
            return conf_cls

        return get_conf_cls()


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

    MULTIPLE_TESTING_CORRECTIONS = ("bonferroni",)
    BACKENDS = ("auto",)
    GUARANTEE_LEVELS = ("image", "object")
    MATCHINGS = ("hausdorff", "iou", "giou", "lac", "mix")

    def __init__(
        self,
        backend: str = "auto",
        guarantee_level: str = "image",
        matching_function: str = "hausdorff",
        confidence_threshold: float | None = None,
        multiple_testing_correction: str | None = None,
        confidence_method: ConfidenceConformalizer | str | None = None,
        localization_method: LocalizationConformalizer | str | None = None,
        localization_prediction_set: str = "additive",  # Fix where we type check
        classification_method: ClassificationConformalizer | str | None = None,
        classification_prediction_set: str = "lac",  # Fix where we type check
        optimizer="binary_search",
        device="cpu",
        # TODO(leo) remove if nonessential: **kwargs,
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
        self.device = device

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

        if matching_function not in self.MATCHINGS:
            raise ValueError(
                f"matching function {matching_function} not accepted, must be one of {self.MATCHINGS}",
            )
        self.matching_function = matching_function

        if multiple_testing_correction is None:
            logger.warning(
                "No multiple_testing_correction provided, assuming no correction is needed. The explicit list of alphas is expected for calibration.",
            )
            self.multiple_testing_correction = multiple_testing_correction
        elif multiple_testing_correction not in self.MULTIPLE_TESTING_CORRECTIONS:
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
                matching_function=matching_function,
                loss=localization_method,
                guarantee_level=guarantee_level,
                prediction_set=localization_prediction_set,
                device=device,
                optimizer=optimizer,
                # TODO(leo) remove if nonessential: **kwargs,
            )
        elif isinstance(localization_method, LocalizationConformalizer):
            self.localization_conformalizer = localization_method
            self.localization_method = localization_method.loss_name
            self.localization_prediction_set = localization_method.prediction_set
        else:
            self.localization_conformalizer = None
            self.localization_method = None

        # Classification

        if isinstance(classification_method, str):
            self.classification_method = classification_method
            self.classification_prediction_set = classification_prediction_set
            self.classification_conformalizer = ODClassificationConformalizer(
                matching_function=matching_function,
                guarantee_level=guarantee_level,
                loss=classification_method,
                prediction_set=classification_prediction_set,
                device=device,
                optimizer=optimizer,
            )
        elif isinstance(classification_method, ODClassificationConformalizer):
            self.classification_conformalizer = classification_method
            self.classification_method = classification_method.method
            self.classification_prediction_set = classification_method.prediction_set
        else:
            self.classification_conformalizer = None
            self.classification_method = None
            self.classification_prediction_set = None

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
                matching_function=matching_function,
                device=device,
                other_losses=[
                    conf.loss
                    for conf in [
                        self.localization_conformalizer,
                        self.classification_conformalizer,
                    ]
                    if conf is not None
                ],
                optimizer=optimizer,
                # TODO(leo) remove if nonessential: **kwargs,
            )
            self.confidence_method = confidence_method
        elif isinstance(confidence_method, ConfidenceConformalizer):
            self.confidence_threshold = None
            self.confidence_conformalizer = confidence_method
            self.confidence_method = confidence_method.loss_name
            self.confidence_method.matching_function = matching_function

        self.evaluator = ODEvaluator(
            confidence_loss=self.confidence_conformalizer.loss
            if self.confidence_conformalizer
            else None,
            localization_loss=self.localization_conformalizer.loss
            if self.localization_conformalizer
            else None,
            classification_loss=self.classification_conformalizer.loss
            if self.classification_conformalizer
            else None,
        )

    def calibrate(
        self,
        predictions: ODPredictions,
        global_alpha: float | None = None,
        alpha_confidence: float | None = None,
        alpha_localization: float | None = None,
        alpha_classification: float | None = None,
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
                (alpha_confidence is None != self.confidence_conformalizer is None)
                or (alpha_localization is None != self.localization_conformalizer is None)
                or (alpha_classification is None != self.classification_conformalizer is None)
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
            self.confidence_conformalizer.conformalize(
                predictions,
                verbose=verbose,
            )
            self.confidence_threshold = (
                1 - lambda_confidence_plus
            )  # predictions.confidence_threshold

            optimistic_confidence_threshold = 1 - lambda_confidence_minus

            if verbose:
                logger.info(
                    f"Calibrated Confidence λ : {lambda_confidence_plus:.4f}\n\t and associated Confidence Threshold : {predictions.confidence_threshold}",
                )
        else:
            predictions.confidence_threshold = self.confidence_threshold
            predictions.matching = None
            optimistic_confidence_threshold = self.confidence_threshold
            lambda_confidence_minus = None
            lambda_confidence_plus = None

        # Now that we fixed the confidence threshold, we need to do the matching before moving on to the next steps

        if predictions.matching is not None:
            logger.warning("Overwriting previous matching")
        if verbose:
            logger.info("Matching Predictions to True Boxes")

        match_predictions_to_true_boxes(
            predictions,  # ref to predictions object, modified in place within func call
            distance_function=self.matching_function,
            verbose=verbose,
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
                    f"Calibrated Localization λ : {lambda_localization}",
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
                    f"Calibrated Classification λ : {lambda_classification}",
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
        parameters: ODParameters | None = None,
        verbose: bool = True,
    ) -> ODConformalizedPredictions:
        """Conformalize the given predictions.

        Args:
        ----
            predictions (ODPredictions): The predictions to be conformalized.
            parameters (Optional[ODParameters]): The parameters to be used for conformalization. If None, the last parameters will be used.results
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
                # That's normal, we conformalize on another dataset
                # raise ValueError(
                #     "The parameters have been computed on another set of predictions.",
                # )
                logger.info(
                    "The parameters have been computed on another set of predictions.",
                )
        else:
            if verbose:
                logger.info("Using last parameters for conformalization")

        # TODO: later check this is correct behavior
        if self.confidence_conformalizer is not None:
            if verbose:
                logger.info("Conformalizing Confidence")
            self.confidence_conformalizer.conformalize(
                predictions,
                verbose=verbose,
                # TODO: parameters=parameters,
            )
        elif parameters is not None:
            if verbose:
                logger.info("Using provided confidence threshold")
            predictions.confidence_threshold = parameters.confidence_threshold
            predictions.matching = None
        else:
            if verbose:
                logger.info("Using last confidence threshold")
            predictions.confidence_threshold = self.confidence_threshold
            predictions.matching = None

        if self.localization_conformalizer is not None:
            if verbose:
                logger.info("Conformalizing Localization")
            conf_boxes = self.localization_conformalizer.conformalize(
                predictions,
                verbose=verbose,
                # TODO:parameters=parameters,
            )
        else:
            conf_boxes = None

        if self.classification_conformalizer is not None:
            if verbose:
                logger.info("Conformalizing Classification")
            conf_cls = self.classification_conformalizer.conformalize(
                predictions,
                verbose=verbose,
                # parameters=parameters,
            )
        else:
            conf_cls = None

        if parameters is None:
            # TODO: rethink this
            raise ValueError(
                "Parameters must be provided for conformalization",
            )

        results = ODConformalizedPredictions(
            predictions=predictions,
            parameters=parameters,
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
        print(f"Confidence threshold is {predictions.confidence_threshold}")
        print(f"Matching is : {predictions.matching is None}")
        if predictions.matching is None:
            match_predictions_to_true_boxes(
                predictions,
                distance_function=self.matching_function,
                verbose=False,
                # TODO: overload_confidence_threshold=parameters.confidence_threshold,
            )
            print("Matching complete")

        odresults = self.evaluator.evaluate(
            predictions,
            parameters,
            conformalized_predictions,
        )

        coverage_obj = odresults.confidence_coverages
        set_size_obj = odresults.confidence_set_sizes
        coverage_loc = odresults.localization_coverages
        set_size_loc = odresults.localization_set_sizes
        coverage_cls = odresults.classification_coverages
        set_size_cls = odresults.classification_set_sizes
        global_coverage = odresults.global_coverage

        # TODO: Use parameters to compare distance to ideal coverage and other things

        if verbose:
            # log results
            logger.info("Evaluation Results:")
            if self.confidence_conformalizer is not None:
                logger.info("\t Confidence:")
                logger.info(f"\t\t Risk: {torch.mean(coverage_obj):.2f}")
                logger.info(
                    f"\t\t Mean Set Size: {torch.mean(set_size_obj):.2f}",
                )
            if self.localization_conformalizer is not None:
                logger.info("\t Localization:")
                logger.info(f"\t\t Risk: {torch.mean(coverage_loc):.2f}")
                logger.info(
                    f"\t\t Mean Set Size: {torch.mean(set_size_loc):.2f}",
                )
            if self.classification_conformalizer is not None:
                logger.info("\t Classification:")
                logger.info(f"\t\t Risk: {torch.mean(coverage_cls):.2f}")
                logger.info(
                    f"\t\t Mean Set Size: {torch.mean(set_size_cls):.2f}",
                )
            if global_coverage is not None:
                logger.info("\t Global:")
                # logger.info(
                #    f"\t\t Risk: {torch.mean(global_coverage):.2f}",
                # )
                logger.info(f"\t\t Risk: {torch.mean(global_coverage)}")

        results = odresults
        # to rename coverage to risks
        return results


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

    ACCEPTED_LOSSES = MappingProxyType(
        {
            "pixelwise": PixelWiseRecallLoss,
            "boxwise": BoxWiseRecallLoss,
        }
    )

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
        self.loss = self.ACCEPTED_LOSSES[localization_loss](device=self.device)

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
            pred_boxes_filtered = [
                (x[y >= 1 - lbd_obj] if len(x[y >= 1 - lbd_obj]) > 0 else x[None, y.argmax()])
                for x, y in zip(
                    predictions.pred_boxes,
                    predictions.confidence,
                )
            ]
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
        bounds: list | None = None,
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
        if bounds is None:
            bounds = [(0, 500), (0.0, 1.0)]
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

    def conformalize(self, predictions: ODPredictions, verbose: bool = True):
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
        predictions.matching = None
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
            predictions.true_boxes,
            predictions.confidence,
        ):
            cov = (
                1
                if len(true_boxes) <= (confidence >= predictions.confidence_threshold).sum()
                else 0
            )
            set_size = (confidence >= predictions.confidence_threshold).sum()
            set_size_obj.append(set_size)
            coverage_obj.append(cov)
        if verbose:
            logger.info(
                f"Confidence Treshold {predictions.confidence_threshold}, Coverage = {torch.mean(coverage_obj)}, Median set size = {torch.mean(set_size_obj)}",
            )

        def compute_set_size(boxes):
            set_sizes = []
            for image_boxes in boxes:
                for box in image_boxes:
                    set_size = (box[2] - box[0]) * (box[3] - box[1])
                    set_size = set_size.item()
                    set_size = torch.sqrt(set_size)
                    set_sizes.append(set_size)
            set_sizes = torch.stack(set_sizes)
            return set_sizes

        conf_boxes = conf_boxes
        true_boxes = predictions.true_boxes
        conf_boxes = [
            (
                x[y >= predictions.confidence_threshold]
                if len(x[y >= predictions.confidence_threshold]) > 0
                else x[None, y.argmax()]
            )
            for x, y in zip(conf_boxes, predictions.confidences)
        ]
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
