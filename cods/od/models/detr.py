import tqdm
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torchvision.transforms as T
from cods.od.models import ODModel
from cods.od.data import ODPredictions


# TODO all models have a clear set of hyperparameters (dictionary) that can be controlled :
# e.g for YOLO: 4 margins, objectness, translation, scale, etc...


# Utilitary functions
# for output bounding box post-processing
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)


def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b


class ResizeChannels(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        self.num_channels = num_channels

    # for if 1 channel, repeat 3 times, if 3 channels, don't change the image
    def forward(self, image):
        if image.shape[0] == 1:
            return image.repeat(3, 1, 1)
        else:
            return image


# Class definition


class DETRModel(ODModel):
    COCO_CLASSES = [
        "N/A",
        "person",
        "bicycle",
        "car",
        "motorcycle",
        "airplane",
        "bus",
        "train",
        "truck",
        "boat",
        "traffic light",
        "fire hydrant",
        "N/A",
        "stop sign",
        "parking meter",
        "bench",
        "bird",
        "cat",
        "dog",
        "horse",
        "sheep",
        "cow",
        "elephant",
        "bear",
        "zebra",
        "giraffe",
        "N/A",
        "backpack",
        "umbrella",
        "N/A",
        "N/A",
        "handbag",
        "tie",
        "suitcase",
        "frisbee",
        "skis",
        "snowboard",
        "sports ball",
        "kite",
        "baseball bat",
        "baseball glove",
        "skateboard",
        "surfboard",
        "tennis racket",
        "bottle",
        "N/A",
        "wine glass",
        "cup",
        "fork",
        "knife",
        "spoon",
        "bowl",
        "banana",
        "apple",
        "sandwich",
        "orange",
        "broccoli",
        "carrot",
        "hot dog",
        "pizza",
        "donut",
        "cake",
        "chair",
        "couch",
        "potted plant",
        "bed",
        "N/A",
        "dining table",
        "N/A",
        "N/A",
        "toilet",
        "N/A",
        "tv",
        "laptop",
        "mouse",
        "remote",
        "keyboard",
        "cell phone",
        "microwave",
        "oven",
        "toaster",
        "sink",
        "refrigerator",
        "N/A",
        "book",
        "clock",
        "vase",
        "scissors",
        "teddy bear",
        "hair drier",
        "toothbrush",
    ]

    MODEL_NAMES = [
        "detr_resnet50",
        "detr_resnet101",
    ]

    def __init__(
        self,
        model_name,
        pretrained=True,
        weights=None,
        device="cuda",
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
            raise ValueError(
                f"Model name {model_name} not available. Available models are {self.MODEL_NAMES}"
            )
        self._model_name = model_name
        if pretrained is True:
            self.model = torch.hub.load(
                "facebookresearch/detr", model_name, pretrained=pretrained
            )
        else:
            raise NotImplementedError("Only pretrained models are available for now")
        self.device = device
        self.model.eval()
        self.model.to(device)
        self.transform = T.Compose(
            [
                T.Resize(800),
                T.ToTensor(),
                ResizeChannels(3),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    # Unsure if this is the right way to do it, there is different ways to define the softmax
    def postprocess(self, outputs, image_sizes):
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

    # def build_predictions(
    #     self,
    #     dataset,
    #     dataset_name: str,
    #     split_name: str,
    #     batch_size: int,
    #     shuffle: bool = False,
    #     verbose: bool = True,
    #     **kwargs,
    # ) -> ODPredictions:
    #     preds = self._load_preds_if_exists(
    #         dataset_name=dataset_name,
    #         split_name=split_name,
    #         task_name="object_detection",
    #     )
    #     if preds is not None:
    #         if verbose:
    #             print("Predictions already exist, loading them...")
    #         return preds
    #     elif verbose:
    #         print("Predictions do not exist, building them...")

    #     dataloader = torch.utils.data.DataLoader(
    #         dataset, batch_size=batch_size, shuffle=shuffle, **kwargs
    #     )

    #     pbar = enumerate(tqdm.tqdm(dataloader, disable=not verbose))

    #     all_image_paths = []
    #     all_true_boxes = []
    #     all_pred_boxes = []
    #     all_confidences = []
    #     all_true_cls = []
    #     all_pred_cls = []

    #     with torch.no_grad():
    #         for i, batch in pbar:

    #             image_paths, true_boxes, pred_boxes, confidences, true_cls, pred_cls = (
    #                 self.predict_batch(batch)
    #             )

    #             all_image_paths.append(image_paths)
    #             all_true_boxes.append(true_boxes)
    #             all_pred_boxes.append(pred_boxes)
    #             all_confidences.append(confidences)
    #             all_true_cls.append(true_cls)
    #             all_pred_cls.append(pred_cls)

    #     all_image_paths = list(
    #         [path for arr_path in all_image_paths for path in arr_path]
    #     )
    #     all_true_boxes = list([box for arr_box in all_true_boxes for box in arr_box])
    #     all_pred_boxes = list([box for arr_box in all_pred_boxes for box in arr_box])
    #     all_confidences = list(
    #         [
    #             confidence
    #             for arr_confidence in all_confidences
    #             for confidence in arr_confidence
    #         ]
    #     )
    #     all_true_cls = list([cls for arr_cls in all_true_cls for cls in arr_cls])
    #     all_pred_cls = list(
    #         [proba for arr_proba in all_pred_cls for proba in arr_proba]
    #     )
    #     # or rather np.concatenate(

    #     preds = ODPredictions(
    #         dataset_name=dataset_name,
    #         split_name=split_name,
    #         image_paths=all_image_paths,
    #         true_boxes=all_true_boxes,
    #         pred_boxes=all_pred_boxes,
    #         confidences=all_confidences,
    #         true_cls=all_true_cls,
    #         pred_cls=all_pred_cls,
    #     )
    #     self._save_preds(preds)
    #     return preds

    def predict_batch(self, batch: list, **kwargs) -> dict:
        """
        Predicts the output given a batch of input tensors.

        Args:
            batch (list): The input batch

        Returns:
            dict: The predicted output as a dictionary with the following keys:
                - "image_paths" (list): The paths of the input images
                - "true_boxes" (list): The true bounding boxes of the objects in the images
                - "pred_boxes" (list): The predicted bounding boxes of the objects in the images
                - "confidences" (list): The confidence scores of the predicted bounding boxes
                - "true_cls" (list): The true class labels of the objects in the images
                - "pred_cls" (list): The predicted class labels of the objects in the images
        """
        image_paths, image_sizes, images, ground_truth = batch

        image_paths, image_sizes, images, ground_truth = batch
        img_shapes = torch.FloatTensor(
            np.stack([image.size for image in images])
        ).cuda()
        images = [self.transform(image) for image in images]
        images = list([image.to(self.device) for image in images])
        outputs = self.model(images)
        pred_boxes, confidences, pred_cls = self.postprocess(outputs, img_shapes)
        true_boxes = list(
            [
                torch.LongTensor(
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
                for true_box in ground_truth
            ]
        )
        true_cls = list(
            [
                torch.LongTensor([box["category_id"] for box in true_box])
                for true_box in ground_truth
            ]
        )
        true_boxes = true_boxes

        return {
            "image_paths": image_paths,
            "true_boxes": true_boxes,
            "pred_boxes": pred_boxes,
            "confidences": confidences,
            "true_cls": true_cls,
            "pred_cls": pred_cls,
        }
