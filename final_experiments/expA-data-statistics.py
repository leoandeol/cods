import logging
import os
import pickle
import traceback
from itertools import product

import numpy as np

from cods.od.cp import ODConformalizer
from cods.od.data import MSCOCODataset
from cods.od.models import DETRModel

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # chose the GPU. If only one, then "0"

logging.getLogger().setLevel(logging.DEBUG)

# set [COCO_PATH] to the directory to your local copy of the COCO dataset
COCO_PATH = "/datasets/shared_datasets/coco/"

data = MSCOCODataset(root=COCO_PATH, split="val")

calibration_ratio = 0.5  # set 0.5 to use 50% for calibration and 50% for testing

use_smaller_subset = True  # TODO: Temp

if use_smaller_subset:
    data_cal, data_val = data.split_dataset(calibration_ratio, shuffle=False, n_calib_test=800)
else:
    data_cal, data_val = data.split_dataset(calibration_ratio, shuffle=False)

# model and weights are downloaded from https://github.com/facebookresearch/detr
model = DETRModel(model_name="detr_resnet50", pretrained=True, device="cpu")
# model = YOLOModel(model_name="yolov8x.pt", pretrained=True)


print(f"{len(data) = }")
print(f"{len(data_cal) = }")
print(f"{len(data_val) = }")

for split in ["cal", "val"]:
    preds_cal = model.build_predictions(
        data_val if split == "val" else data_cal,
        dataset_name="mscoco",
        split_name="test" if split == "val" else "cal",
        batch_size=12,
        collate_fn=data._collate_fn,
        shuffle=False,
        force_recompute=False,  # False,
        deletion_method="nms",
        filter_preds_by_confidence=1e-3,
    )
    n_objects_pred = [len(pred) for pred in preds_cal.confidences]
    n_objects_true = [len(true) for true in preds_cal.true_boxes]
    print(f"Split: {'Calibration' if split == 'cal' else 'Validation'}")
    print(f"\t Statistics of {split} dataset:")
    print(
        f"\t Min (Predictions vs Ground Truth): {np.min(n_objects_pred)} vs {np.min(n_objects_true)}"
    )
    print(
        f"\t Max (Predictions vs Ground Truth): {np.max(n_objects_pred)} vs {np.max(n_objects_true)}"
    )
    print(
        f"\t Median (Predictions vs Ground Truth): {np.median(n_objects_pred)} vs {np.median(n_objects_true)}"
    )
    print(
        f"\t Mean (Predictions vs Ground Truth): {np.mean(n_objects_pred)} vs {np.mean(n_objects_true)}"
    )
    print(
        f"\t Quantiles [0.90, 0.95, 0.97, 0.99] (Predictions vs Ground Truth): "
        f"{np.quantile(n_objects_pred, [0.90, 0.95, 0.97, 0.99])} vs {np.quantile(n_objects_true, [0.90, 0.95, 0.97, 0.99])}"
    )

    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 5))
    plt.hist(
        n_objects_pred,
        bins=30,
        alpha=0.5,
        label="Predicted Objects",
        color="blue",
        density=True,
    )
    plt.hist(
        n_objects_true,
        bins=30,
        alpha=0.5,
        label="Ground Truth Objects",
        color="orange",
        density=True,
    )
    plt.xlabel("Number of Objects per Image")
    plt.ylabel("Density")
    plt.title(f"Histogram of Object Counts ({'Calibration' if split == 'cal' else 'Validation'})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"histogram_objects_{'calibration' if split == 'cal' else 'validation'}.png")
    plt.close()
    print(f"Saved histogram for {split} split.")

    CONFIDENCE_THRESHOLD = 1 - 0.9259869  # Set your desired threshold

    # Filter predictions by confidence threshold
    n_objects_pred_thresh = [
        np.sum(np.array(conf) >= CONFIDENCE_THRESHOLD) for conf in preds_cal.confidences
    ]

    plt.figure(figsize=(8, 5))
    plt.hist(
        n_objects_pred,
        bins=30,
        alpha=0.5,
        label="Predicted Objects",
        color="blue",
        density=True,
    )
    plt.hist(
        n_objects_pred_thresh,
        bins=30,
        alpha=0.5,
        label=f"Predicted Objects (conf >= {CONFIDENCE_THRESHOLD:.3f})",
        color="green",
        density=True,
    )
    plt.hist(
        n_objects_true,
        bins=30,
        alpha=0.5,
        label="Ground Truth Objects",
        color="orange",
        density=True,
    )
    plt.xlabel("Number of Objects per Image")
    plt.ylabel("Density")
    plt.title(
        f"Histogram of Object Counts ({'Calibration' if split == 'cal' else 'Validation'})\nFiltered by Confidence >= {CONFIDENCE_THRESHOLD:.3f}"
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        f"histogram_objects_{'calibration' if split == 'cal' else 'validation'}_conf_{CONFIDENCE_THRESHOLD:.3f}.png"
    )
    plt.close()
    print(f"Saved filtered histogram for {split} split (conf >= {CONFIDENCE_THRESHOLD:.3f}).")
