import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from cods.od.data import MSCOCODataset
from cods.od.loss import (
    ClassificationLossWrapper,
    ODBinaryClassificationLoss,
    PixelWiseRecallLoss,
)
from cods.od.models import DETRModel, YOLOModel
from cods.od.utils import (
    assymetric_hausdorff_distance_old,
    generalized_iou,
    match_predictions_to_true_boxes,
)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = (
    "1"  # chose the GPU. If only one, then "0"
)

logging.getLogger().setLevel(logging.INFO)

# set [COCO_PATH] to the directory to your local copy of the COCO dataset
COCO_PATH = "/datasets/shared_datasets/coco/"

data = MSCOCODataset(root=COCO_PATH, split="val")

calibration_ratio = (
    0.5  # set 0.5 to use 50% for calibration and 50% for testing
)

use_smaller_subset = True  # TODO: Temp

if use_smaller_subset:
    data_cal, data_val = data.split_dataset(
        calibration_ratio, shuffle=False, n_calib_test=800
    )
else:
    data_cal, data_val = data.split_dataset(calibration_ratio, shuffle=False)

print(f"{len(data) = }")
print(f"{len(data_cal) = }")
print(f"{len(data_val) = }")


def build_preds(model):
    preds_cal = model.build_predictions(
        data_cal,
        dataset_name="mscoco",
        split_name="cal",
        batch_size=12,
        collate_fn=data._collate_fn,  # TODO: make this a default for COCO
        shuffle=False,
        force_recompute=False,  # False,
        deletion_method="nms",
    )
    preds_val = model.build_predictions(
        data_val,
        dataset_name="mscoco",
        split_name="test",
        batch_size=12,
        collate_fn=data._collate_fn,
        shuffle=False,
        force_recompute=False,  # False,
        deletion_method="nms",
    )
    return preds_cal, preds_val
