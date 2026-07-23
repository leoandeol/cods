"""LTT baselines experiment in the exp1-detr.py standard: for each config of the
reference grid, calibrate/conformalize/evaluate both LTT-Bonferroni
(`LearnThenTestConformalizer`) and LTT-Split-Fixed-Sequence
(`SplitFixedSequenceConformalizer`), with incremental pickle checkpointing.

Runtime note: at full scale (~2500 calibration images), Bonferroni takes ~2 min
per config and SFS ~25 min per config (it sweeps the full 3D lambda grid on the
graph split). Trim the config lists below or run a single matching function first
if you need results quickly; the checkpoint file makes reruns resumable.
"""

import logging
import os
import pickle
import traceback

import numpy as np

from cods.od.cp import LearnThenTestConformalizer, SplitFixedSequenceConformalizer
from cods.od.data import MSCOCODataset
from cods.od.models import DETRModel

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # chose the GPU. If only one, then "0"

logging.getLogger().setLevel(logging.WARNING)

# set [COCO_PATH] to the directory to your local copy of the COCO dataset
COCO_PATH = "/datasets/shared_datasets/coco/"

MODEL_NAME = os.environ.get("MODEL_NAME", "detr_resnet50")  # or detr_resnet101
DEVICE = os.environ.get("DEVICE", "cpu")  # e.g. "cuda:0" for GPU inference
BATCH_SIZE = int(os.environ.get("BS", "12"))

data = MSCOCODataset(root=COCO_PATH, split="val")

calibration_ratio = 0.5  # set 0.5 to use 50% for calibration and 50% for testing

data_cal, data_val = data.split_dataset(calibration_ratio, shuffle=False)

# model and weights are downloaded from https://github.com/facebookresearch/detr
model = DETRModel(model_name=MODEL_NAME, pretrained=True, device=DEVICE)

print(f"{len(data) = }")
print(f"{len(data_cal) = }")
print(f"{len(data_val) = }")

preds_cal = model.build_predictions(
    data_cal,
    dataset_name="mscoco",
    split_name="cal",
    batch_size=BATCH_SIZE,
    collate_fn=data._collate_fn,
    shuffle=False,
    force_recompute=False,
    deletion_method="nms",
    filter_preds_by_confidence=3e-3,
)
preds_val = model.build_predictions(
    data_val,
    dataset_name="mscoco",
    split_name="test",
    batch_size=BATCH_SIZE,
    collate_fn=data._collate_fn,
    shuffle=False,
    force_recompute=False,
    deletion_method="nms",
    filter_preds_by_confidence=3e-3,
)
preds_cal.to("cpu")
preds_val.to("cpu")

results = {}

# Same reference grid as exp1-detr.py
alphas = [[0.02, 0.05, 0.05], [0.03, 0.1, 0.1]]
matching_functions = ["mix", "hausdorff", "lac", "giou"]
confidence_methods = [
    "box_count_threshold",
    "box_count_recall",
]
localization_methods = ["thresholded", "pixelwise", "boxwise"]
classification_prediction_sets = ["lac", "aps"]
localization_prediction_sets = ["additive", "multiplicative"]

# LTT-specific settings (shared by both variants so they are directly comparable)
methods = ["ltt_bonferroni", "ltt_split_fixed_sequence"]
delta = 0.1  # high-probability level: P(R <= alpha) >= 1 - delta
n_lambda = 40
n_sequence_steps = 50  # SFS only: D discretization steps of the learned path
split_ratio = 0.5  # SFS only: graph-selection fraction of the calibration set
# lambda_cls grid: uniform coverage plus a log-spaced tail hugging 1 (DETR class
# scores are unnormalized and often tiny, so LAC thresholds 1-lambda must reach
# down to 1e-7 for the label sets to move; harmless for APS).
cls_grid = np.unique(
    np.concatenate(
        [
            np.linspace(0, 1, n_lambda),
            1.0 - np.logspace(np.log10(5e-2), np.log10(1e-7), 25),
        ]
    )
)

force_recompute = False

configs = []
for alpha in alphas:
    for matching_function in matching_functions:
        for confidence_method in confidence_methods:
            for localization_method in localization_methods:
                for classification_prediction_set in classification_prediction_sets:
                    for localization_prediction_set in localization_prediction_sets:
                        configs.append(
                            {
                                "alpha": alpha,
                                "matching_function": matching_function,
                                "confidence_method": confidence_method,
                                "localization_method": localization_method,
                                "classification_prediction_set": classification_prediction_set,
                                "localization_prediction_set": localization_prediction_set,
                            },
                        )

output_path = f"./paper_experiments/results-exp4-ltt-{MODEL_NAME}.pkl"
parameters_path = f"./paper_experiments/parameters-exp4-ltt-{MODEL_NAME}.pkl"
all_parameters = {}

for config in configs:
    for method in methods:
        config_str = (
            f"{method}_delta-{delta}_alpha-{config['alpha']}-"
            f"{config['matching_function']}_{config['confidence_method']}_"
            f"{config['localization_method']}_{config['classification_prediction_set']}_"
            f"{config['localization_prediction_set']}"
        )
        try:
            # Load pickle if exists
            if os.path.exists(output_path):
                with open(output_path, "rb") as f:
                    results = pickle.load(f)
                if config_str in results and not force_recompute:
                    print(f"Already computed {config_str}, skipping...")
                    continue

            lambda_localization_max = (
                1000.0 if config["localization_prediction_set"] == "additive" else 2.0
            )
            common_kwargs = {
                "guarantee_level": "image",
                "matching_function": config["matching_function"],
                "confidence_method": config["confidence_method"],
                "localization_method": config["localization_method"],
                "localization_prediction_set": config["localization_prediction_set"],
                "classification_method": "binary",
                "classification_prediction_set": config["classification_prediction_set"],
                "n_lambda_confidence": n_lambda,
                "n_lambda_localization": n_lambda,
                "lambda_classification_grid": cls_grid,
                "lambda_localization_max": lambda_localization_max,
            }
            if method == "ltt_bonferroni":
                conf = LearnThenTestConformalizer(**common_kwargs)
            else:
                conf = SplitFixedSequenceConformalizer(
                    split_ratio=split_ratio,
                    n_sequence_steps=n_sequence_steps,
                    **common_kwargs,
                )

            preds_cal.matching = None
            preds_cal.confidence_threshold = None
            parameters = conf.calibrate(
                preds_cal,
                alpha_confidence=config["alpha"][0],
                alpha_localization=config["alpha"][1],
                alpha_classification=config["alpha"][2],
                global_delta=delta,
                verbose=False,
            )

            preds_val.matching = None
            preds_val.confidence_threshold = None
            conformal_preds = conf.conformalize(preds_val, parameters=parameters)

            results_val = conf.evaluate(
                preds_val,
                parameters=parameters,
                conformalized_predictions=conformal_preds,
                include_confidence_in_global=False,
                verbose=False,
            )

            results[config_str] = results_val
            all_parameters[config_str] = parameters

            print(f"Results for config {config_str}:")
            print(f"  {results_val}")
            # Save results to a pickle file

            with open(output_path, "wb") as f:
                pickle.dump(results, f)
            with open(parameters_path, "wb") as f:
                pickle.dump(all_parameters, f)

            print(f"Results have been pickled to {output_path}")
        except ValueError as e:
            # LTT-specific: config not certifiable at this (alpha, delta, n)
            print(f"NOT CERTIFIABLE {config_str}: {e}")
            continue
        except Exception as e:
            print(f"Error with config {config}: {e}")
            print(traceback.format_exc())
            continue
