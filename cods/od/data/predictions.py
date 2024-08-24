from typing import Any, List, Optional, Union

import torch

from cods.base.data import (
    ConformalizedPredictions,
    Parameters,
    Predictions,
    Results,
)


class ODPredictions(Predictions):
    """Class representing predictions for object detection tasks.

    Args:
    ----
        id (int): Unique ID of the predictions.
        dataset_name (str): Name of the dataset.
        split_name (str): Name of the data split.
        image_paths: List of image paths.
        true_boxes: List of true bounding boxes.
        pred_boxes: List of predicted bounding boxes.
        confidences: List of confidence scores for predicted boxes.
        true_cls: List of true class labels.
        pred_cls: List of predicted class labels.

    Attributes:
    ----------
        image_paths: List of image paths.
        true_boxes: List of true bounding boxes.
        pred_boxes: List of predicted bounding boxes.
        confidence: List of confidence scores for predicted boxes.
        true_cls: List of true class labels.
        pred_cls: List of predicted class labels.
        preds_cls: ClassificationPredictions instance.
        n_classes: Number of classes.
        matching: Matching information.
        confidence_threshold: Confidence threshold.

    Methods:
    -------
        __len__: Returns the number of image paths.
        __str__: Returns a string representation of the ODPredictions object.

    """

    def __init__(
        self,
        dataset_name: str,
        split_name: str,
        image_paths,
        true_boxes,
        pred_boxes,
        confidences,
        true_cls,
        pred_cls,
        unique_id: Optional[int] = None,
    ):
        super().__init__(
            unique_id=unique_id,
            dataset_name=dataset_name,
            split_name=split_name,
            task_name="object_detection",
        )
        self.image_paths = image_paths
        self.true_boxes = true_boxes
        self.pred_boxes = pred_boxes
        self.confidence = confidences
        self.true_cls = true_cls
        self.pred_cls = pred_cls

        # ClassificationPredictions instance
        self.preds_cls: Optional[Any] = None

        self.n_classes = len(self.pred_cls[0][0])
        self.matching: Optional[Any] = None
        self.confidence_threshold: Optional[Union[float, torch.Tensor]] = None
        # TODO: if change matching, then must reset the mathcing

    def __len__(self):
        return len(self.image_paths)

    def __str__(self):
        return f"ODPredictions_len={len(self)}"


class ODParameters(Parameters):
    """Class representing parameters for object detection tasks."""

    def __init__(
        self,
        global_alpha: float,
        confidence_threshold: float,
        predictions_id: int,
        alpha_confidence: Optional[float] = None,
        alpha_localization: Optional[float] = None,
        alpha_classification: Optional[float] = None,
        lambda_confidence_plus: Optional[float] = None,
        lambda_confidence_minus: Optional[float] = None,
        lambda_localization: Optional[float] = None,
        lambda_classification: Optional[float] = None,
        unique_id: Optional[int] = None,
    ):
        """Initializes a new instance of the ODParameters class.

        Parameters
        ----------
            global_alpha (float): The global alpha (the sum of the non-None alphas).
            alpha_confidence (float): The alpha for confidence.
            alpha_localization (float): The alpha for localization.
            alpha_classification (float): The alpha for classification
            lambda_confidence_plus (float): The lambda for confidence (conservative).
            lambda_confidence_minus (float): The lambda for confidence (optimistic).
            lambda_localization (float): The lambda for localization.
            lambda_classification (float): The lambda for classification.
            confidence_threshold (float): The confidence threshold.
            predictions_id (int): The unique ID of the predictions.
            unique_id (int): The unique ID of the parameters.

        """
        super().__init__(predictions_id, unique_id)
        self.global_alpha = global_alpha
        self.alpha_confidence = alpha_confidence
        self.alpha_localization = alpha_localization
        self.alpha_classification = alpha_classification
        self.lambda_confidence_plus = lambda_confidence_plus
        self.lambda_confidence_minus = lambda_confidence_minus
        self.lambda_localization = lambda_localization
        self.lambda_classification = lambda_classification
        self.confidence_threshold = confidence_threshold


class ODConformalizedPredictions(ConformalizedPredictions):
    """Class representing conformalized predictions for object detection tasks."""

    def __init__(
        self,
        predictions: ODPredictions,
        parameters: ODParameters,
        conf_boxes: Optional[torch.Tensor] = None,
        conf_cls: Optional[torch.Tensor] = None,
    ):
        """Initializes a new instance of the ODResults class.

        Parameters
        ----------
            predictions (ODPredictions): The object detection predictions.
            parameters (ODParameters): The conformalizers parameters.
            conf_boxes (torch.Tensor): The conformal boxes, after filtering.
            conf_cls (torch.Tensor): The conformal prediction sets for class labels of each box.

        """
        super().__init__(
            predictions_id=predictions.unique_id,
            parameters_id=parameters.unique_id,
        )
        self.preds = predictions
        self.conf_boxes = conf_boxes
        self.conf_cls = conf_cls


class ODResults(Results):
    """Class representing results for object detection tasks."""

    def __init__(
        self,
        predictions: ODPredictions,
        parameters: Parameters,
        conformalized_predictions: ODConformalizedPredictions,
        confidence_set_sizes: Optional[torch.Tensor | List[float]] = None,
        confidence_coverages: Optional[torch.Tensor | List[float]] = None,
        localization_set_sizes: Optional[torch.Tensor | List[float]] = None,
        localization_coverages: Optional[torch.Tensor | List[float]] = None,
        classification_set_sizes: Optional[torch.Tensor | List[float]] = None,
        classification_coverages: Optional[torch.Tensor | List[float]] = None,
        global_coverage: Optional[torch.Tensor | float] = None,
    ):
        """Initializes a new instance of the ODResults class.

        Parameters
        ----------
            predictions (ODPredictions): The object detection predictions.
            parameters (ODParameters): The conformalizers parameters.
            conformalized_predictions (ODConformalizedPredictions): The conformalized predictions.
            confidence_set_sizes (torch.Tensor): The confidence set sizes.
            confidence_coverages (torch.Tensor): The confidence coverages.
            localization_set_sizes (torch.Tensor): The localization set sizes.
            localization_coverages (torch.Tensor): The localization coverages.
            classification_set_sizes (torch.Tensor): The classification set sizes.
            classification_coverages (torch.Tensor): The classification coverages.
            global_coverage (torch.Tensor | float): The global coverage.

        """
        super().__init__(
            predictions_id=predictions.unique_id,
            parameters_id=parameters.unique_id,
            conformalized_id=conformalized_predictions.unique_id,
        )
        self.preds = predictions
        self.confidence_set_sizes = confidence_set_sizes
        self.confidence_coverages = confidence_coverages
        self.localization_set_sizes = localization_set_sizes
        self.localization_coverages = localization_coverages
        self.classification_set_sizes = classification_set_sizes
        self.classification_coverages = classification_coverages
        self.global_coverage = global_coverage
