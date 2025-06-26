"""DETR (DEtection TRansformer) model implementation for object detection.

This module provides the DETR model wrapper for object detection with conformal
prediction support, including model loading, prediction generation, and
post-processing utilities for bounding box transformations.
"""

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T

from cods.od.models.model import ODModel
from cods.od.models.utils import ResizeChannels

# TODO all models have a clear set of hyperparameters (dictionary) that can be controlled :
# e.g for YOLO: 4 margins, objectness, translation, scale, etc...


# Utilitary functions
# for output bounding box post-processing
def box_cxcywh_to_xyxy(x):
    """Convert bounding boxes from (cx, cy, w, h) to (x1, y1, x2, y2) format.

    Args:
        x (torch.Tensor): Bounding boxes in (cx, cy, w, h) format.

    Returns:
        torch.Tensor: Bounding boxes in (x1, y1, x2, y2) format.

    """
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


def box_xyxy_to_cxcywh(x):
    """Convert bounding boxes from (x1, y1, x2, y2) to (cx, cy, w, h) format.

    Args:
        x (torch.Tensor): Bounding boxes in (x1, y1, x2, y2) format.

    Returns:
        torch.Tensor: Bounding boxes in (cx, cy, w, h) format.

    """
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)


def rescale_bboxes(out_bbox, size):
    """Rescale bounding boxes to image size.

    Args:
        out_bbox (torch.Tensor): Normalized bounding boxes.
        size (tuple): Image size (width, height).

    Returns:
        torch.Tensor: Rescaled bounding boxes in absolute coordinates.

    """
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b


class DETRModel(ODModel):
    """DETR (DEtection TRansformer) model wrapper for object detection.

    Provides a wrapper around the DETR model with preprocessing, postprocessing,
    and prediction generation capabilities for conformal prediction workflows.

    Attributes:
        MODEL_NAMES (list): List of supported DETR model variants.

    """

    MODEL_NAMES = [
        "detr_resnet50",
        "detr_resnet101",
    ]

    def __init__(
        self,
        model_name="detr_resnet50",
        pretrained=True,
        weights=None,
        device="cpu",
        save=True,
        save_dir_path=None,
    ):
        """Initialize the DETR model.

        Args:
            model_name (str, optional): Name of the DETR model variant. Defaults to 'detr_resnet50'.
            pretrained (bool, optional): Whether to use pretrained weights. Defaults to True.
            weights (str, optional): Path to custom weights. Defaults to None.
            device (str, optional): Device to run the model on. Defaults to 'cpu'.
            save (bool, optional): Whether to save predictions. Defaults to True.
            save_dir_path (str, optional): Directory to save predictions. Defaults to None.

        Raises:
            ValueError: If model_name is not in MODEL_NAMES.
            NotImplementedError: If pretrained is False (only pretrained models supported).

        """
        super().__init__(
            model_name=model_name,
            save_dir_path=save_dir_path,
            pretrained=pretrained,
            weights=weights,
            device=device,
        )
        if model_name not in self.MODEL_NAMES:
            raise ValueError(
                f"Model name {model_name} not available. Available models are {self.MODEL_NAMES}",
            )
        self._model_name = model_name
        if pretrained is True:
            self.model = torch.hub.load(
                "facebookresearch/detr",
                model_name,
                pretrained=pretrained,
            )
        else:
            raise NotImplementedError(
                "Only pretrained models are available for now",
            )
        self.device = device
        self.model.eval()
        self.model.to(device)
        self.transform = T.Compose(
            [
                T.Resize(800),
                T.ToTensor(),
                ResizeChannels(3),
                T.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ],
        )

    # Unsure if this is the right way to do it, there is different ways to define the softmax
    def postprocess(self, outputs, image_sizes):
        """Post-process model outputs to extract predictions.

        Args:
            outputs (dict): Raw model outputs containing 'pred_logits' and 'pred_boxes'.
            image_sizes (torch.Tensor): Image sizes for rescaling bounding boxes.

        Returns:
            tuple: (scaled_pred_boxes, confidences, pred_cls) where:
                - scaled_pred_boxes: Rescaled bounding boxes
                - confidences: Confidence scores
                - pred_cls: Class probabilities

        """
        out_logits, out_bboxes = outputs["pred_logits"], outputs["pred_boxes"]

        # convert to [x0, y0, x1, y1] format
        # print(out_bboxes.shape)
        boxes = box_cxcywh_to_xyxy(out_bboxes)
        # print(boxes.shape)
        # and from relative [0, 1] to absolute [0, height] coordinates
        # had to reverse w and h to apparently match our format
        img_w, img_h = image_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        scaled_pred_boxes = boxes * scale_fct[:, None, :]

        confidence_probas = outputs["pred_logits"].softmax(-1)[..., :-1]
        confidences = confidence_probas.max(-1).values

        prob = F.softmax(out_logits, -1)
        pred_cls = prob[..., :-1]
        # pred_cls, cls_label = prob[..., :-1].max(-1)

        return scaled_pred_boxes, confidences, pred_cls

    def predict_batch(self, batch: list, **kwargs) -> dict:
        """Predicts the output given a batch of input tensors.

        Args:
        ----
            batch (list): The input batch
            **kwargs: Additional keyword arguments passed to the prediction method

        Returns:
        -------
            dict: The predicted output as a dictionary with the following keys:
                - "image_paths" (list): The paths of the input images
                - "true_boxes" (list): The true bounding boxes of the objects in the images
                - "pred_boxes" (list): The predicted bounding boxes of the objects in the images
                - "confidences" (list): The confidence scores of the predicted bounding boxes
                - "true_cls" (list): The true class labels of the objects in the images
                - "pred_cls" (list): The predicted class labels of the objects in the images

        """
        image_paths, image_sizes, images, ground_truth = batch
        img_shapes = torch.FloatTensor(
            np.stack([image.size for image in images]),
        ).to(self.device)
        images = [self.transform(image) for image in images]
        images = [image.to(self.device) for image in images]
        outputs = self.model(images)
        pred_boxes, confidences, pred_cls = self.postprocess(
            outputs,
            img_shapes,
        )
        true_boxes = [
            torch.LongTensor(
                [
                    [
                        box["bbox"][0],
                        box["bbox"][1],
                        box["bbox"][0] + box["bbox"][2],
                        box["bbox"][1] + box["bbox"][3],
                    ]
                    for box in true_box
                ],
            )
            for true_box in ground_truth
        ]
        true_cls = [
            torch.LongTensor([box["category_id"] for box in true_box])
            for true_box in ground_truth
        ]
        true_boxes = true_boxes

        return {
            "image_paths": image_paths,
            "image_shapes": image_sizes,
            "true_boxes": true_boxes,
            "pred_boxes": pred_boxes,
            "confidences": confidences,
            "true_cls": true_cls,
            "pred_cls": pred_cls,
        }
