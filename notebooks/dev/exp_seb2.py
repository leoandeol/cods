import logging
import os

from cods.od.data import MSCOCODataset
from cods.od.models import DETRModel

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # chose the GPU. If only one, then "0"

logging.getLogger().setLevel(logging.DEBUG)


# set [COCO_PATH] to the directory to your local copy of the COCO dataset
COCO_PATH = "/datasets/shared_datasets/coco/"

data = MSCOCODataset(root=COCO_PATH, split="val")

calibration_ratio = 0.5  # set 0.5 to use 50% for calibration and 50% for testing

use_smaller_subset = False  # TODO: Temp

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

# TODO: debug lac
conf = ODConformalizer(
    guarantee_level="image",
    matching_function="mix",  # "giou",
    multiple_testing_correction=None,
    confidence_method="box_count_threshold",  # "nb_boxes",
    # confidence_threshold=0.5,
    localization_method="pixelwise",  # "pixelwise",
    localization_prediction_set="additive",  # "multiplicative",
    classification_method="binary",
    classification_prediction_set="lac",
    backend="auto",
    optimizer="binary_search",
)

# TODO(leo): we can replace this by anything, doesn't even need a guarantee (confidence)
parameters = conf.calibrate(
    preds_cal,
    alpha_confidence=0.05,
    alpha_localization=0.07,
    alpha_classification=0.08,
)


conformal_preds = conf.conformalize(preds_val, parameters=parameters)

# TODO: Rewrite it so we only compute the confidence loss and not the max of three. Main loss of condiecne shoudl be just itself but in calibration use the proxy maximum loss with the others
results_val = conf.evaluate(
    preds_val,
    parameters=parameters,
    conformalized_predictions=conformal_preds,
    include_confidence_in_global=False,
)

from cods.od.visualization import create_pdf_with_plots

create_pdf_with_plots(preds_val, conformal_preds, idx_to_label=MSCOCODataset.NAMES)
