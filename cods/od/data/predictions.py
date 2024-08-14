from typing import Any, Optional, Union

import torch

from cods.base.data import ConformalizedPredictions, Parameters, Predictions, Results


class ODPredictions(Predictions):
    """
    Class representing predictions for object detection tasks.

    Args:
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
    """
    Class representing parameters for object detection tasks.
    """

    def __init__(
        self,
        global_alpha: float,
        confidence_alpha: Optional[float],
        localization_alpha: Optional[float],
        classification_alpha: Optional[float],
        confidence_lambda: Optional[float],
        localization_lambda: Optional[float],
        classification_lambda: Optional[float],
        predictions_id: int,
        unique_id: Optional[int] = None,
    ):
        """
        Initializes a new instance of the ODParameters class.

        Parameters:
            global_alpha (float): The global alpha (the sum of the non-None alphas).
            confidence_alpha (float): The confidence alpha.
            localization_alpha (float): The localization alpha.
            classification_alpha (float): The classification alpha.
            confidence_lambda (float): The confidence lambda.
            localization_lambda (float): The localization lambda.
            classification_lambda (float): The classification lambda.
            predictions_id (int): The unique ID of the predictions.
            unique_id (int): The unique ID of the parameters.
        """
        super().__init__(predictions_id, unique_id)
        self.global_alpha = global_alpha
        self.confidence_alpha = confidence_alpha
        self.localization_alpha = localization_alpha
        self.classification_alpha = classification_alpha
        self.confidence_lambda = confidence_lambda
        self.localization_lambda = localization_lambda
        self.classification_lambda = classification_lambda


class ODConformalizedPredictions(ConformalizedPredictions):
    """
    Class representing conformalized predictions for object detection tasks.
    """

    def __init__(
        self,
        predictions: ODPredictions,
        parameters: ODParameters,
        conf_boxes: Optional[torch.Tensor],
        conf_cls: Optional[torch.Tensor],
    ):
        """
        Initializes a new instance of the ODResults class.

        Parameters:
            predictions (ODPredictions): The object detection predictions.
            parameters (ODParameters): The conformalizers parameters.
            conf_boxes (torch.Tensor): The conformal boxes.
            conf_cls (torch.Tensor): The conformal prediction sets for class labels of each box.
        """
        super().__init__(
            predictions_id=predictions.unique_id, parameters_id=parameters.unique_id
        )
        self.preds = predictions
        self.conf_boxes = conf_boxes
        self.conf_cls = conf_cls


class ODResults(Results):
    """
    Class representing results for object detection tasks.
    """

    def __init__(
        self,
        predictions: ODPredictions,
        parameters: Parameters,
        conformalized_predictions: ODConformalizedPredictions,
        confidence_set_sizes: torch.Tensor,
        confidence_coverages: torch.Tensor,
        localization_set_sizes: torch.Tensor,
        localization_coverages: torch.Tensor,
        classification_set_sizes: torch.Tensor,
        classification_coverages: torch.Tensor,
    ):
        """
        Initializes a new instance of the ODResults class.

        Parameters:
            predictions (ODPredictions): The object detection predictions.
            parameters (ODParameters): The conformalizers parameters.
            conformalized_predictions (ODConformalizedPredictions): The conformalized predictions.
            confidence_set_sizes (torch.Tensor): The confidence set sizes.
            confidence_coverages (torch.Tensor): The confidence coverages.
            localization_set_sizes (torch.Tensor): The localization set sizes.
            localization_coverages (torch.Tensor): The localization coverages.
            classification_set_sizes (torch.Tensor): The classification set sizes.
            classification_coverages (torch.Tensor): The classification coverages.
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
