import logging
import os
import pickle
import traceback
from itertools import product

from cods.od.cp import ODConformalizer
from cods.od.data import MSCOCODataset
from cods.od.models import DETRModel

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = (
    "0"  # chose the GPU. If only one, then "0"
)

logging.getLogger().setLevel(logging.DEBUG)

# set [COCO_PATH] to the directory to your local copy of the COCO dataset
COCO_PATH = "/datasets/shared_datasets/coco/"

data = MSCOCODataset(root=COCO_PATH, split="val")

calibration_ratio = (
    0.5  # set 0.5 to use 50% for calibration and 50% for testing
)

use_smaller_subset = False  # TODO: Temp

if use_smaller_subset:
    data_cal, data_val = data.split_dataset(
        calibration_ratio, shuffle=False, n_calib_test=800
    )
else:
    data_cal, data_val = data.split_dataset(calibration_ratio, shuffle=True)

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
    filter_preds_by_confidence=1e-3,
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
    filter_preds_by_confidence=1e-3,
)

results = {}

alphas = [[0.03, 0.1, 0.1]]
matching_functions = ["mix"]  # , "hausdorff", "lac"]#["giou", "hausdorff"]
confidence_methods = [
    "box_count_threshold",
    "box_count_recall",
    "box_thresholded_distance",
]
localization_methods = ["pixelwise", "boxwise"]
classification_prediction_sets = ["lac", "aps"]
localization_prediction_sets = ["additive", "multiplicative"]

configs = []
for alpha in alphas:
    for matching_function in matching_functions:
        for confidence_method in confidence_methods:
            for localization_method in localization_methods:
                for (
                    classification_prediction_set
                ) in classification_prediction_sets:
                    for (
                        localization_prediction_set
                    ) in localization_prediction_sets:
                        configs.append(
                            {
                                "alpha": alpha,
                                "matching_function": matching_function,
                                "confidence_method": confidence_method,
                                "localization_method": localization_method,
                                "classification_prediction_set": classification_prediction_set,
                                "localization_prediction_set": localization_prediction_set,
                            }
                        )
for config in configs:
    try:
        conf = ODConformalizer(
            guarantee_level="image",
            matching_function=config["matching_function"],
            multiple_testing_correction=None,
            confidence_method=config["confidence_method"],
            localization_method=config["localization_method"],
            localization_prediction_set=config["localization_prediction_set"],
            classification_method="binary",
            classification_prediction_set=config[
                "classification_prediction_set"
            ],
            backend="auto",
            optimizer="binary_search",
        )

        parameters = conf.calibrate(
            preds_cal,
            alpha_confidence=config["alpha"][0],
            alpha_localization=config["alpha"][1],
            alpha_classification=config["alpha"][2],
        )

        conformal_preds_cal = conf.conformalize(
            preds_cal, parameters=parameters
        )

        results_cal = conf.evaluate(
            preds_cal,
            parameters=parameters,
            conformalized_predictions=conformal_preds_cal,
            include_confidence_in_global=False,
        )

        conformal_preds = conf.conformalize(preds_val, parameters=parameters)

        results_val = conf.evaluate(
            preds_val,
            parameters=parameters,
            conformalized_predictions=conformal_preds,
            include_confidence_in_global=False,
        )

        config_str = f"alpha-{config['alpha']}-{config['matching_function']}_{config['confidence_method']}_{config['localization_method']}_{config['classification_prediction_set']}_{config['localization_prediction_set']}"

        results[config_str] = results_val

        print(f"Results for config {config_str}:")
        print(f"  {results_val}")
        # Save results to a pickle file

        output_path = "./final_experiments/results-exp1-detr.pkl"
        with open(output_path, "wb") as f:
            pickle.dump(results, f)

        print(f"Results have been pickled to {output_path}")
    except Exception as e:
        print(f"Error with config {config}: {e}")
        print(traceback.format_exc())
        continue
