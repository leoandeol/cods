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


class DETRBDD100KModel(ODModel):
    MODEL_NAMES = ("detr_resnet50", "detr_resnet101")

    BDD_CLASSES:ClassVar = [
        "bike",
        "bus",
        "car",
        "motor",
        "person",
        "rider",
        "traffic light",
        "traffic sign",
        "train",
        "truck",
    ]

    BDD_TO_COCO_NAME:ClassVar = {
        "bike": "bicycle",
        "bus": "bus",
        "car": "car",
        "motor": "motorcycle",
        "person": "person",
        "rider": "person", # A revoir
        "traffic light": "traffic light",
        "traffic sign": "stop sign", # A revoir
        "train": "train",
        "truck": "truck",
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
        save_dir_path=None
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

        if not pretrained:
            raise NotImplementedError("Only pretrained models are available for now")

        self.model = torch.hub.load(
            "facebookresearch/detr",
            model_name,
            pretrained=True,
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

        self.bdd_name_to_idx = {
            name: idx for idx, name in enumerate(self.BDD_CLASSES)
        }

        self.coco_name_to_idx = {
            name: idx for idx, name in enumerate(self.COCO_CLASSES)
            if name != "N/A"
        }

        self.bdd_to_coco_idx = torch.tensor(
            [
                self.coco_name_to_idx[self.BDD_TO_COCO_NAME[name]]
                for name in self.BDD_CLASSES
            ],
            dtype=torch.long,
        )

    def postprocess(self, outputs, image_sizes):
        out_logits = outputs["pred_logits"]
        out_bboxes = outputs["pred_boxes"]

        boxes = box_cxcywh_to_xyxy(out_bboxes)

        img_w, img_h = image_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        pred_boxes = boxes * scale_fct[:, None, :]

        prob = F.softmax(out_logits, dim=-1)
        coco_prob = prob[..., :-1]

        bdd_idx = self.bdd_to_coco_idx.to(coco_prob.device)
        pred_cls = coco_prob[..., bdd_idx]

        confidences = pred_cls.max(dim=-1).values
        return pred_boxes, confidences, pred_cls

    def predict_batch(self, batch: list, **kwargs) -> dict:
        image_paths, image_sizes, images, ground_truth = batch

        img_shapes = torch.FloatTensor(
            np.stack([image.size for image in images]),
        ).to(self.device)

        images = [self.transform(image).to(self.device) for image in images]
        with torch.no_grad():
            outputs = self.model(images)
        pred_boxes, confidences, pred_cls = self.postprocess(outputs, img_shapes)

        true_boxes = [
            torch.FloatTensor(
                [
                    [
                        box["bbox"][0],
                        box["bbox"][1],
                        box["bbox"][0] + box["bbox"][2],
                        box["bbox"][1] + box["bbox"][3],
                    ]
                    for box in true_box
                ]
            )
            if len(true_box) > 0
            else torch.zeros((0, 4), dtype=torch.float32)
            for true_box in ground_truth
        ]

        true_cls = [
            torch.LongTensor([box["category_id"] for box in true_box])
            if len(true_box) > 0
            else torch.zeros((0,), dtype=torch.long)
            for true_box in ground_truth
        ]

        return {
            "image_paths": image_paths,
            "image_shapes": image_sizes,
            "true_boxes": true_boxes,
            "pred_boxes": pred_boxes,
            "confidences": confidences,
            "true_cls": true_cls,
            "pred_cls": pred_cls,
        }