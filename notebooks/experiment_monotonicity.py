import argparse
import logging
import os

from cods.od.data import MSCOCODataset
from cods.od.models import DETRModel

parser = argparse.ArgumentParser()
# add argument for matching_function, localization_method, localization_prediction_set, confidence_method, classification_prediction_set
parser.add_argument(
    "--matching_function",
    type=str,
    default="giou",
    help="Matching function to use. Options: giou, hausdorff",
)
parser.add_argument(
    "--localization_method",
    type=str,
    default="boxwise",
    help="Localization method to use. Options: boxwise, imagewise",
)
parser.add_argument(
    "--localization_prediction_set",
    type=str,
    default="additive",
    help="Localization prediction set to use. Options: additive, multiplicative",
)
parser.add_argument(
    "--confidence_method",
    type=str,
    default="box_count_threshold",
    help="Confidence method to use. Options: box_count_threshold, box_count_recall, box_thresholded_distance",
)
parser.add_argument(
    "--classification_prediction_set",
    type=str,
    default="lac",
    help="Classification prediction set to use. Options: lac, aps",
)
args = parser.parse_args()

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

# model and weights are downloaded from https://github.com/facebookresearch/detr
model = DETRModel(model_name="detr_resnet50", pretrained=True, device="cpu")
# model = YOLOModel(model_name="yolov8x.pt", pretrained=True)


print(f"{len(data) = }")
print(f"{len(data_cal) = }")
print(f"{len(data_val) = }")

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

from cods.od.cp import ODConformalizer

matching_function = args.matching_function
localization_method = args.localization_method
localization_prediction_set = args.localization_prediction_set
confidence_method = args.confidence_method
classification_prediction_set = args.classification_prediction_set

from contextlib import redirect_stdout

with open(
    f"monotonicity_{matching_function}_{localization_method}_{localization_prediction_set}_{confidence_method}_{classification_prediction_set}.txt",
    "w",
) as f:
    with redirect_stdout(f):
        print(f"{matching_function = }")
        print(f"{localization_method = }")
        print(f"{localization_prediction_set = }")
        print(f"{confidence_method = }")
        print(f"{classification_prediction_set = }")
        conf = ODConformalizer(
            guarantee_level="image",
            matching_function=matching_function,
            multiple_testing_correction=None,
            confidence_method=confidence_method,
            localization_method=localization_method,
            localization_prediction_set=localization_prediction_set,
            classification_method="binary",
            classification_prediction_set=classification_prediction_set,
            backend="auto",
            optimizer="binary_search",
        )
        parameters = conf.calibrate(
            preds_cal,
            alpha_confidence=0.02,
            alpha_classification=0.05,
            alpha_localization=0.05,
        )
        conf_preds_val = conf.conformalize(preds_val, parameters)
        conf.evaluate(
            preds_val,
            parameters,
            conf_preds_val,
            include_confidence_in_global=False,
        )

import matplotlib.pyplot as plt

n = len(conf.localization_conformalizer.optimizer2.all_risks_raw) / 13
n = int(n)
plt.figure(figsize=(16, 9))
for i in range(13):
    if i == 0:
        plt.plot(
            conf.localization_conformalizer.optimizer2.all_lbds_cnf[
                i * n : (i + 1) * n
            ],
            conf.localization_conformalizer.optimizer2.all_risks_raw[
                i * n : (i + 1) * n
            ],
            c="b",
            label="Raw Loss",
        )
        plt.plot(
            conf.localization_conformalizer.optimizer2.all_lbds_cnf[
                i * n : (i + 1) * n
            ],
            conf.localization_conformalizer.optimizer2.all_risks_mon[
                i * n : (i + 1) * n
            ],
            c="r",
            label="Monotonized Loss",
        )
    else:
        plt.plot(
            conf.localization_conformalizer.optimizer2.all_lbds_cnf[
                i * n : (i + 1) * n
            ],
            conf.localization_conformalizer.optimizer2.all_risks_raw[
                i * n : (i + 1) * n
            ],
            c="b",
        )
    plt.plot(
        conf.localization_conformalizer.optimizer2.all_lbds_cnf[
            i * n : (i + 1) * n
        ],
        conf.localization_conformalizer.optimizer2.all_risks_mon[
            i * n : (i + 1) * n
        ],
        c="r",
    )
    # plt.show()
# Put X and Y legend
plt.xlabel("Lambda Confidence")
plt.ylabel("Risk")
plt.title("Raw and Monotonized Losses for Localization")
# focus on the interval between 0.9 and 1
# plt.xlim(0.9995, 1)
plt.xlim(0.9, 1)
# plt.xscale("log")
plt.legend()
plt.savefig(
    f"monotonicity_zoomed_loc_{matching_function}_{localization_method}_{localization_prediction_set}_{confidence_method}_{classification_prediction_set}.png"
)

n = len(conf.localization_conformalizer.optimizer2.all_risks_raw) / 13
n = int(n)
plt.figure(figsize=(16, 9))
for i in range(13):
    if i == 0:
        plt.plot(
            conf.localization_conformalizer.optimizer2.all_lbds_cnf[
                i * n : (i + 1) * n
            ],
            conf.localization_conformalizer.optimizer2.all_risks_raw[
                i * n : (i + 1) * n
            ],
            c="b",
            label="Raw Loss",
        )
        plt.plot(
            conf.localization_conformalizer.optimizer2.all_lbds_cnf[
                i * n : (i + 1) * n
            ],
            conf.localization_conformalizer.optimizer2.all_risks_mon[
                i * n : (i + 1) * n
            ],
            c="r",
            label="Monotonized Loss",
        )
    else:
        plt.plot(
            conf.localization_conformalizer.optimizer2.all_lbds_cnf[
                i * n : (i + 1) * n
            ],
            conf.localization_conformalizer.optimizer2.all_risks_raw[
                i * n : (i + 1) * n
            ],
            c="b",
        )
    plt.plot(
        conf.localization_conformalizer.optimizer2.all_lbds_cnf[
            i * n : (i + 1) * n
        ],
        conf.localization_conformalizer.optimizer2.all_risks_mon[
            i * n : (i + 1) * n
        ],
        c="r",
    )
    # plt.show()
# Put X and Y legend
plt.xlabel("Lambda Confidence")
plt.ylabel("Risk")
plt.title("Raw and Monotonized Losses for Localization")
# focus on the interval between 0.9 and 1
# plt.xlim(0.9995, 1)
# plt.xscale("log")
plt.legend()
plt.savefig(
    f"monotonicity_all_loc_{matching_function}_{localization_method}_{localization_prediction_set}_{confidence_method}_{classification_prediction_set}.png"
)

n = len(conf.classification_conformalizer.optimizer2.all_risks_raw) / 25
n = int(n)
plt.figure(figsize=(16, 9))
for i in range(25):
    if i == 0:
        plt.plot(
            conf.classification_conformalizer.optimizer2.all_lbds_cnf[
                i * n : (i + 1) * n
            ],
            conf.classification_conformalizer.optimizer2.all_risks_raw[
                i * n : (i + 1) * n
            ],
            c="b",
            label="Raw Loss",
        )
        plt.plot(
            conf.classification_conformalizer.optimizer2.all_lbds_cnf[
                i * n : (i + 1) * n
            ],
            conf.classification_conformalizer.optimizer2.all_risks_mon[
                i * n : (i + 1) * n
            ],
            c="r",
            label="Monotonized Loss",
        )
    else:
        plt.plot(
            conf.classification_conformalizer.optimizer2.all_lbds_cnf[
                i * n : (i + 1) * n
            ],
            conf.classification_conformalizer.optimizer2.all_risks_raw[
                i * n : (i + 1) * n
            ],
            c="b",
        )
        plt.plot(
            conf.classification_conformalizer.optimizer2.all_lbds_cnf[
                i * n : (i + 1) * n
            ],
            conf.classification_conformalizer.optimizer2.all_risks_mon[
                i * n : (i + 1) * n
            ],
            c="r",
        )
    # plt.show()
# Put X and Y legend
plt.xlabel("Lambda Confidence")
plt.ylabel("Risk")
plt.title("Raw and Monotonized Losses for Classification")
plt.legend()
plt.savefig(
    f"monotonicity_all_class_{matching_function}_{localization_method}_{localization_prediction_set}_{confidence_method}_{classification_prediction_set}.png"
)
