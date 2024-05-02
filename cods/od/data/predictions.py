from typing import Any, Optional, Union

import torch

from cods.base.data import Predictions


class ODPredictions(Predictions):
    """
    Class representing predictions for object detection tasks.

    Args:
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
    ):
        super().__init__(dataset_name, split_name, task_name="object_detection")
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

    def __len__(self):
        return len(self.image_paths)

    def __str__(self):
        return f"ODPredictions_len={len(self)}"
