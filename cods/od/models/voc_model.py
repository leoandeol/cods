from typing import ClassVar

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T

from cods.od.models.model import ODModel
from cods.od.models.utils import ResizeChannels


def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    return torch.stack(
        [x_c - 0.5 * w, y_c - 0.5 * h, x_c + 0.5 * w, y_c + 0.5 * h],
        dim=-1,
    )

class DETRVOCModel(ODModel):
    MODEL_NAMES = ("detr_resnet50", "detr_resnet101")

    VOC_CLASSES:ClassVar = [
        "aeroplane",
        "bicycle",
        "bird",
        "boat",
        "bottle",
        "bus",
        "car",
        "cat",
        "chair",
        "cow",
        "diningtable",
        "dog",
        "horse",
        "motorbike",
        "person",
        "pottedplant",
        "sheep",
        "sofa",
        "train",
        "tvmonitor",
    ]

    VOC_TO_COCO_NAME:ClassVar = {
        "aeroplane": "airplane",
        "bicycle": "bicycle",
        "bird": "bird",
        "boat": "boat",
        "bottle": "bottle",
        "bus": "bus",
        "car": "car",
        "cat": "cat",
        "chair": "chair",
        "cow": "cow",
        "diningtable": "dining table",
        "dog": "dog",
        "horse": "horse",
        "motorbike": "motorcycle",
        "person": "person",
        "pottedplant": "potted plant",
        "sheep": "sheep",
        "sofa": "couch",
        "train": "train",
        "tvmonitor": "tv",
    }

    COCO_CLASSES:ClassVar = [
        "N/A", "person", "bicycle", "car", "motorcycle", "airplane", "bus",
        "train", "truck", "boat", "traffic light", "fire hydrant", "N/A",
        "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse",
        "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "N/A",
        "backpack", "umbrella", "N/A", "N/A", "handbag", "tie", "suitcase",
        "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
        "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
        "N/A", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana",
        "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza",
        "donut", "cake", "chair", "couch", "potted plant", "bed", "N/A",
        "dining table", "N/A", "N/A", "toilet", "N/A", "tv", "laptop", "mouse",
        "remote", "keyboard", "cell phone", "microwave", "oven", "toaster",
        "sink", "refrigerator", "N/A", "book", "clock", "vase", "scissors",
        "teddy bear", "hair drier", "toothbrush",
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
        super().__init__(
            model_name=model_name,
            save_dir_path=save_dir_path,
            pretrained=pretrained,
            weights=weights,
            device=device,
        )

        if model_name not in self.MODEL_NAMES:
            raise ValueError(f"{model_name} not in {self.MODEL_NAMES}")

        self.device = device
        self.model = torch.hub.load(
            "facebookresearch/detr",
            model_name,
            pretrained=pretrained,
        )
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

        self.voc_name_to_idx = {
            name: idx for idx, name in enumerate(self.VOC_CLASSES)
        }
        self.coco_name_to_idx = {
            name: idx for idx, name in enumerate(self.COCO_CLASSES) if name != "N/A"
        }
        self.voc_to_coco_idx = torch.tensor(
            [
                self.coco_name_to_idx[self.VOC_TO_COCO_NAME[name]]
                for name in self.VOC_CLASSES
            ],
            dtype=torch.long,
        )

    def _extract_voc_objects(self, target):
        objects = target["annotation"].get("object", [])
        if isinstance(objects, dict):
            objects = [objects]
        return objects

    def _voc_target_to_boxes_and_labels(self, target):
        objects = self._extract_voc_objects(target)
        boxes, labels = [], []

        for obj in objects:
            name = obj["name"]
            if name not in self.voc_name_to_idx:
                continue

            bb = obj["bndbox"]
            xmin = int(float(bb["xmin"]))
            ymin = int(float(bb["ymin"]))
            xmax = int(float(bb["xmax"]))
            ymax = int(float(bb["ymax"]))

            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(self.voc_name_to_idx[name])

        if len(boxes) == 0:
            return torch.zeros((0, 4), dtype=torch.long), torch.zeros((0,), dtype=torch.long)

        return torch.tensor(boxes, dtype=torch.long), torch.tensor(labels, dtype=torch.long)

    def postprocess(self, outputs, image_sizes):
        out_logits = outputs["pred_logits"]
        out_bboxes = outputs["pred_boxes"]

        boxes = box_cxcywh_to_xyxy(out_bboxes)

        img_w, img_h = image_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        pred_boxes = boxes * scale_fct[:, None, :]

        prob = F.softmax(out_logits, dim=-1)
        coco_prob = prob[..., :-1]

        voc_idx = self.voc_to_coco_idx.to(coco_prob.device)
        pred_cls = coco_prob[..., voc_idx]
        confidences = pred_cls.max(dim=-1).values

        return pred_boxes, confidences, pred_cls

    def predict_batch(self, batch: list, **kwargs) -> dict:
        image_paths, image_sizes, images, ground_truth = batch

        img_shapes = torch.FloatTensor(
            np.stack([image.size for image in images]),
        ).to(self.device)

        images = [self.transform(image).to(self.device) for image in images]

        outputs = self.model(images)
        pred_boxes, confidences, pred_cls = self.postprocess(outputs, img_shapes)

        true_boxes = []
        true_cls = []

        for target in ground_truth:
            boxes, labels = self._voc_target_to_boxes_and_labels(target)
            true_boxes.append(boxes)
            true_cls.append(labels)

        return {
            "image_paths": image_paths,
            "image_shapes": image_sizes,
            "true_boxes": true_boxes,
            "pred_boxes": pred_boxes,
            "confidences": confidences,
            "true_cls": true_cls,
            "pred_cls": pred_cls,
        }
