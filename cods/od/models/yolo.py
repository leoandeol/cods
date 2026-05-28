from abc import ABC, abstractmethod
from typing import ClassVar

import numpy as np
import torch
from ultralytics import YOLO

from cods.od.models.mixins import BDD100KModelMixin, COCOLikeTargetMixin, VOCModelMixin
from cods.od.models.model import ODModel


def xywh2xyxy_scaled(x, width_scale, height_scale):
    y = x.clone()
    y[:, 0] = (x[:, 0] - x[:, 2] / 2) * width_scale  # top left x
    y[:, 1] = (x[:, 1] - x[:, 3] / 2) * height_scale  # top left y
    y[:, 2] = (x[:, 0] + x[:, 2] / 2) * width_scale  # bottom right x
    y[:, 3] = (x[:, 1] + x[:, 3] / 2) * height_scale  # bottom right y
    return y


class AlteredYOLO(YOLO):
    def __init__(self, model_path):
        super().__init__(model_path, verbose=False)
        self.raw_output = None
        self.input_shape = None

    def predict(self, source=None, stream=False, **kwargs):
        def output_hook(module, input, output):
            self.raw_output = output[0].detach().clone()

        def image_hook(module, input, output):
            # print(input[0].shape[2:][::-1])
            self.input_shape = input[0].shape[2:][::-1]

        # Register the forward hook
        start_hook = self.model.model[0].register_forward_hook(image_hook)
        end_hook = self.model.model[-1].register_forward_hook(output_hook)

        # Run prediction
        try:
            results = super().predict(source, stream, verbose=False, **kwargs)
        finally:
            # Remove the hook
            start_hook.remove()
            end_hook.remove()

        return results



class YOLOModel(ABC, ODModel):
    SOURCE_CLASSES :ClassVar[list[str]]= [
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
        "truck", "boat", "traffic light", "fire hydrant", "stop sign",
        "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
        "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
        "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
        "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
        "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
        "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
        "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
        "couch", "potted plant", "bed", "dining table", "toilet", "tv",
        "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
        "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
        "scissors", "teddy bear", "hair drier", "toothbrush",
    ]

    def __init__(
        self,
        model_name="yolov8x.pt",
        device="cpu",
        save_dir_path=None,
    ):
        self.model = AlteredYOLO(model_name)
        self.model.to(device)
        super().__init__(
            model_name=model_name,
            save_dir_path=save_dir_path,
            pretrained=True,
            weights=None,
            device=device,
        )


    @abstractmethod
    def _target_to_boxes_and_labels(self, target):
        ...

    @property
    def device(self):
        return self.model.device

    @device.setter
    def device(self, value):
        self.model.to(device=value)

    def to(self, device):
        self.model.to(device)
        return self

    def postprocess_one(self, raw_output, img_shape, model_width:int, model_height:int):
        original_width, original_height = img_shape

        box_output = raw_output.t()

        # Calculate scaling factors
        width_scale = original_width / model_width
        height_scale = original_height / model_height

        # convert to [x0, y0, x1, y1] format
        out_boxes = box_output[:, :4]
        boxes = xywh2xyxy_scaled(out_boxes, width_scale, height_scale)

        yolo_probs = torch.softmax(box_output[:, 4:], dim=-1)
        pred_cls = self.map_source_probs(yolo_probs)

        confidences = pred_cls.max(dim=-1).values

        return boxes, confidences, pred_cls

    def map_source_probs(self, yolo_probs: torch.Tensor) -> torch.Tensor:
        return yolo_probs

    # Unsure if this is the right way to do it, there is different ways to define the softmax
    def postprocess(
        self,
        raw_output,
        img_shapes,
        model_input_size,
    ):
        model_width, model_height = model_input_size

        outputs = [
            self.postprocess_one(elem, img_shapes[i], model_width, model_height)
            for i, elem in enumerate(raw_output)
        ]

        boxes, confs, probs = zip(*outputs)
        return list(boxes), list(confs), list(probs)

    def predict_batch(self, batch: list, **kwargs) -> dict:
        """Predicts the output given a batch of input tensors.

        Args:
        ----
            batch (list): The input batch

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

        with torch.no_grad():
            self.model(images)

        pred_boxes, confidences, pred_cls = self.postprocess(
            self.model.raw_output,
            img_shapes,
            self.model.input_shape,
        )
        true_boxes, true_cls = zip(
            *[self._target_to_boxes_and_labels(target) for target in ground_truth]
        )
        return {
            "image_paths": image_paths,
            "image_shapes": image_sizes,
            "true_boxes": list(true_boxes),
            "pred_boxes": pred_boxes,
            "confidences": confidences,
            "true_cls": list(true_cls),
            "pred_cls": pred_cls,
        }

class COCOYOLOModel(COCOLikeTargetMixin, YOLOModel):
    unused_coco_91:ClassVar[list[int]] = {0, 12, 26, 29, 30, 45, 66, 68, 69, 71, 83, 91}
    convert_to_91:ClassVar[torch.Tensor] = torch.tensor(list(set(range(91)) - set(unused_coco_91)), dtype=torch.long)
    def map_source_probs(self, yolo_probs: torch.Tensor) -> torch.Tensor:
        cls_probs_new = torch.zeros(
            yolo_probs.shape[0],
            91,
            device=yolo_probs.device,
            dtype=yolo_probs.dtype,
        )
        cls_probs_new[:, self.convert_to_91.to(yolo_probs.device)] = yolo_probs
        return cls_probs_new

class BDD100KYOLOModel(BDD100KModelMixin, YOLOModel):
    ...

class VOCYOLOModel(VOCModelMixin, YOLOModel):
    ...
