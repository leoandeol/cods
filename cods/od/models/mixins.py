from __future__ import annotations

from typing import ClassVar

import torch


class COCOLikeTargetMixin:
    def _target_to_boxes_and_labels(self, target):
        if len(target) == 0:
            return (
                torch.zeros((0, 4), dtype=torch.float32),
                torch.zeros((0,), dtype=torch.long),
            )

        boxes = [
            [
                box["bbox"][0],
                box["bbox"][1],
                box["bbox"][0] + box["bbox"][2],
                box["bbox"][1] + box["bbox"][3],
            ]
            for box in target
        ]

        labels = [box["category_id"] for box in target]

        return (
            torch.tensor(boxes, dtype=torch.float32),
            torch.tensor(labels, dtype=torch.long),
        )

class TargetProjectionMixin:
    TARGET_TO_COCO_NAME: ClassVar[dict[str, str]]
    SOURCE_CLASSES: ClassVar[list[str]]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        source_to_idx = {name: idx for idx, name in enumerate(self.SOURCE_CLASSES) if name is not None and name != "N/A"}

        self.target_to_source_idx = torch.tensor(
            [
                source_to_idx[source_name]
                for source_name in self.TARGET_TO_COCO_NAME.values()
            ],
            dtype=torch.long,
        )

    def map_source_probs(self, yolo_probs: torch.Tensor) -> torch.Tensor:
        idx = self.target_to_source_idx.to(yolo_probs.device)
        return yolo_probs[..., idx]

class BDD100KModelMixin(TargetProjectionMixin, COCOLikeTargetMixin):
    TARGET_TO_COCO_NAME: ClassVar[dict[str, str]] = {
        "bike": "bicycle",
        "bus": "bus",
        "car": "car",
        "motor": "motorcycle",
        "person": "person",
        "rider": "person",
        "traffic light": "traffic light",
        "traffic sign": "stop sign",
        "train": "train",
        "truck": "truck",
    }


class VOCModelMixin(TargetProjectionMixin):
    TARGET_TO_COCO_NAME: ClassVar[dict[str, str]] = {
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

    def _target_to_boxes_and_labels(self, target):
        objects = target["annotation"].get("object", [])
        if isinstance(objects, dict):
            objects = [objects]

        name_to_idx = {name: idx for idx, name in enumerate(self.TARGET_TO_COCO_NAME.keys())}

        boxes = []
        labels = []

        for obj in objects:
            name = obj["name"]
            if name not in name_to_idx:
                continue

            bb = obj["bndbox"]
            boxes.append([
                int(float(bb["xmin"])),
                int(float(bb["ymin"])),
                int(float(bb["xmax"])),
                int(float(bb["ymax"])),
            ])
            labels.append(name_to_idx[name])

        if len(boxes) == 0:
            return (
                torch.zeros((0, 4), dtype=torch.float32),
                torch.zeros((0,), dtype=torch.long),
            )

        return (
            torch.tensor(boxes, dtype=torch.float32),
            torch.tensor(labels, dtype=torch.long),
        )
