import tqdm
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torchvision.transforms as T
from cods.base.models import Model
from cods.od.data import ODPredictions

from torchvision.transforms.functional import pad
import torchvision
import numbers


def get_padding(image):
    w, h = image.size
    max_wh = np.max([w, h])
    h_padding = (max_wh - w) / 2
    v_padding = (max_wh - h) / 2
    l_pad = h_padding if h_padding % 1 == 0 else h_padding + 0.5
    t_pad = v_padding if v_padding % 1 == 0 else v_padding + 0.5
    r_pad = h_padding if h_padding % 1 == 0 else h_padding - 0.5
    b_pad = v_padding if v_padding % 1 == 0 else v_padding - 0.5
    padding = (int(l_pad), int(t_pad), int(r_pad), int(b_pad))
    return padding


class NewPad(object):
    def __init__(self, fill=0, padding_mode="constant"):
        assert isinstance(fill, (numbers.Number, str, tuple))
        assert padding_mode in ["constant", "edge", "reflect", "symmetric"]

        self.fill = fill
        self.padding_mode = padding_mode

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be padded.

        Returns:
            PIL Image: Padded image.
        """
        return pad(img, get_padding(img), self.fill, self.padding_mode)

    def __repr__(self):
        return f"NewPad(padding={self.padding}, fill={self.fill}, padding_mode={self.padding_mode})"


def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


def nms(boxes, scores, nms_thr, top=200):
    """Single class NMS implemented in Numpy."""
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order_ori = scores.detach().cpu().numpy().argsort()[::-1].copy()[:top]

    x1 = x1[order_ori]
    y1 = y1[order_ori]
    x2 = x2[order_ori]
    y2 = y2[order_ori]

    order = np.arange(len(order_ori))

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = torch.maximum(x1[i], x1[order[1:]])
        yy1 = torch.maximum(y1[i], y1[order[1:]])
        xx2 = torch.minimum(x2[i], x2[order[1:]])
        yy2 = torch.minimum(y2[i], y2[order[1:]])

        w = torch.maximum(torch.zeros(len(xx2)).cuda(), xx2 - xx1 + 1)
        h = torch.maximum(torch.zeros(len(yy2)).cuda(), yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr.detach().cpu().numpy() <= nms_thr)[0]
        order = order[inds + 1]

    return order_ori[keep]


def postprocess(
    prediction, num_classes, conf_thre=0.7, nms_thre=0.45, class_agnostic=False
):
    box_corner = prediction.new(prediction.shape)
    box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
    box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
    box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
    box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
    prediction[:, :, :4] = box_corner[:, :, :4]

    output = [None for _ in range(len(prediction))]
    for i, image_pred in enumerate(prediction):
        # If none are remaining => process next image
        if not image_pred.size(0):
            continue
        # Get score and class with highest confidence
        class_conf, class_pred = torch.max(
            image_pred[:, 5 : 5 + num_classes], 1, keepdim=True
        )

        conf_mask = (image_pred[:, 4] * class_conf.squeeze() >= conf_thre).squeeze()
        # Detections ordered as (x1, y1, x2, y2, obj_conf, class_conf, class_pred)
        detections = torch.cat((image_pred[:, :5], class_conf, class_pred.float()), 1)
        detections = detections[conf_mask]
        if not detections.size(0):
            continue

        if class_agnostic:
            nms_out_index = torchvision.ops.nms(
                detections[:, :4],
                detections[:, 4] * detections[:, 5],
                nms_thre,
            )
        else:
            nms_out_index = torchvision.ops.batched_nms(
                detections[:, :4],
                detections[:, 4] * detections[:, 5],
                detections[:, 6],
                nms_thre,
            )

        detections = detections[nms_out_index]
        if output[i] is None:
            output[i] = detections
        else:
            output[i] = torch.cat((output[i], detections))

    return output


class YOLOXModel(Model):
    COCO_CLASSES = {
        0: "__background__",
        1: "person",
        2: "bicycle",
        3: "car",
        4: "motorcycle",
        5: "airplane",
        6: "bus",
        7: "train",
        8: "truck",
        9: "boat",
        10: "traffic light",
        11: "fire hydrant",
        12: "stop sign",
        13: "parking meter",
        14: "bench",
        15: "bird",
        16: "cat",
        17: "dog",
        18: "horse",
        19: "sheep",
        20: "cow",
        21: "elephant",
        22: "bear",
        23: "zebra",
        24: "giraffe",
        25: "backpack",
        26: "umbrella",
        27: "handbag",
        28: "tie",
        29: "suitcase",
        30: "frisbee",
        31: "skis",
        32: "snowboard",
        33: "sports ball",
        34: "kite",
        35: "baseball bat",
        36: "baseball glove",
        37: "skateboard",
        38: "surfboard",
        39: "tennis racket",
        40: "bottle",
        41: "wine glass",
        42: "cup",
        43: "fork",
        44: "knife",
        45: "spoon",
        46: "bowl",
        47: "banana",
        48: "apple",
        49: "sandwich",
        50: "orange",
        51: "broccoli",
        52: "carrot",
        53: "hot dog",
        54: "pizza",
        55: "donut",
        56: "cake",
        57: "chair",
        58: "couch",
        59: "potted plant",
        60: "bed",
        61: "dining table",
        62: "toilet",
        63: "tv",
        64: "laptop",
        65: "mouse",
        66: "remote",
        67: "keyboard",
        68: "cell phone",
        69: "microwave",
        70: "oven",
        71: "toaster",
        72: "sink",
        73: "refrigerator",
        74: "book",
        75: "clock",
        76: "vase",
        77: "scissors",
        78: "teddy bear",
        79: "hair drier",
        80: "toothbrush",
    }

    AVAILABLE_MODELS = [
        "yolox_tiny",
        "yolox_nano",
        "yolox_s",
        "yolox_m",
        "yolox_l",
        "yolox_x",
        "yolov3",
        "yolox_custom",
    ]

    def __init__(self, model="yolox_m", pretrained=True, weights=None, device="cuda"):
        super().__init__(pretrained, weights, device)
        if model not in self.AVAILABLE_MODELS:
            raise NotImplementedError(
                f"Model {model} not available, choose one of {self.AVAILABLE_MODELS}"
            )
        self._model_name = model
        if pretrained is True:
            self.model = torch.hub.load(
                "Megvii-BaseDetection/YOLOX", model, pretrained=True
            )

        else:
            raise NotImplementedError("Only pretrained models are available for now")
        self.model.eval()
        self.model.to(device)
        self.transform = T.Compose(
            [
                NewPad(fill=114),
                T.Resize(640),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    # Unsure if this is the right way to do it, there is different ways to define the softmax
    def postprocess(self, outputs, image_sizes):
        out_bboxes, conf, classif = (
            outputs[:, :, :4],
            outputs[:, :, 4],
            outputs[:, :, 5:],
        )
        # print(outputs.shape)
        # assert False

        keeps = []
        for i in range(len(out_bboxes)):
            # keep = nms(out_bboxes[i], conf[i], 0.5)
            class_conf, class_idx = torch.max(classif[i], dim=1)
            keep = torchvision.ops.batched_nms(
                out_bboxes[i],
                conf[i] * class_conf,
                class_idx,
                0.5,
            )
            keeps.append(keep)

        out_bboxes = list([out_bbox[keep] for out_bbox, keep in zip(out_bboxes, keeps)])
        conf = list([conf[keep] for conf, keep in zip(conf, keeps)])
        classif = list([classif[keep] for classif, keep in zip(classif, keeps)])

        # convert to [x0, y0, x1, y1] format
        # print(out_bboxes.shape)
        boxes = list([box_cxcywh_to_xyxy(out_bboxs) for out_bboxs in out_bboxes])
        # print(boxes.shape)
        # and from relative [0, 1] to absolute [0, height] coordinates
        # had to reverse w and h to apparently match our format
        # img_w, img_h = image_sizes.unbind(1)
        largest_side, idx_lgst = torch.max(image_sizes, dim=1)
        margins = torch.abs(image_sizes[:, 0] - image_sizes[:, 1]) / 2
        ratio = largest_side / 640
        scale_fct = torch.stack([ratio] * 4, dim=1)
        scaled_pred_boxes = list(
            [box * scale_fct[i, None, :] for i, box in enumerate(boxes)]
        )
        # print(image_sizes[0])
        # print(scaled_pred_boxes[0].shape, scaled_pred_boxes[0], margins[0], ratio[0])
        # assert False
        for i, (idx, margin) in enumerate(zip(idx_lgst, margins)):
            if idx == 0:
                # width is largest so we padded the height
                scaled_pred_boxes[i][1] -= margins[i]
                scaled_pred_boxes[i][3] -= margins[i]
            else:
                scaled_pred_boxes[i][0] -= margins[i]
                scaled_pred_boxes[i][2] -= margins[i]

        # confidence_probas = outputs["pred_logits"].softmax(-1)[..., :-1]
        confidences = conf

        cls_probas = classif
        # cls_probas, cls_label = prob[..., :-1].max(-1)

        return scaled_pred_boxes, confidences, cls_probas

    def build_predictions(
        self, dataloader: torch.utils.data.DataLoader, verbose=True, **kwargs
    ) -> ODPredictions:
        pbar = tqdm.tqdm(enumerate(dataloader), disable=not verbose)

        all_image_paths = []
        all_true_boxes = []
        all_pred_boxes = []
        all_confidences = []
        all_true_cls = []
        all_cls_probas = []

        with torch.no_grad():
            for i, batch in pbar:
                image_paths, image_sizes, images, ground_truth = batch
                # todo not sure true_boxes is in the right format
                # print(images[0].size, image_sizes[0])
                img_shapes = torch.FloatTensor(
                    np.stack([image.size for image in images])
                ).cuda()
                images = [self.transform(image) for image in images]
                images = torch.stack([image.to(self.device) for image in images])
                outputs = self.model(images)
                pred_boxes, confidences, cls_probas = self.postprocess(
                    outputs, img_shapes
                )
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
                # true_boxes = list(
                #     [
                #         box_cxcywh_to_xyxy(
                #             torch.FloatTensor([box["bbox"] for box in true_box])[
                #                 None, ...
                #             ]
                #         )[0]
                #         for true_box in true_boxes
                #     ]
                # )
                # print(true_boxes.shape)
                true_boxes = true_boxes
                all_image_paths.append(image_paths)
                all_true_boxes.append(true_boxes)
                all_pred_boxes.append(pred_boxes)
                all_confidences.append(
                    list(
                        [
                            confidence.detach().cpu().numpy()
                            for confidence in confidences
                        ]
                    )
                )
                all_true_cls.append(true_cls)
                all_cls_probas.append(
                    list([cls_proba.detach().cpu().numpy() for cls_proba in cls_probas])
                )

        all_image_paths = list(
            [path for arr_path in all_image_paths for path in arr_path]
        )
        all_true_boxes = list([box for arr_box in all_true_boxes for box in arr_box])
        all_pred_boxes = list([box for arr_box in all_pred_boxes for box in arr_box])
        all_confidences = list(
            [
                confidence
                for arr_confidence in all_confidences
                for confidence in arr_confidence
            ]
        )
        all_true_cls = list([cls for arr_cls in all_true_cls for cls in arr_cls])
        all_cls_probas = list(
            [proba for arr_proba in all_cls_probas for proba in arr_proba]
        )
        # or rather np.concatenate(

        return ODPredictions(
            image_paths=all_image_paths,
            true_boxes=all_true_boxes,
            pred_boxes=all_pred_boxes,
            confidences=all_confidences,
            true_cls=all_true_cls,
            pred_cls=all_cls_probas,
        )
