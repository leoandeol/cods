# Purpose:
# Plot monotization losses for confidence, localization and classification losses
# Do it sequentially, all else being equal, for :
# - two models : DETR, YOLO
# - a few losses :
# - three matchings : hausdorff, lac, mix
# - filtering by confidence at 1e-3 and no filtering.
# - two arrangements of alphas: [0.03, 0.05, 0.05], [0.03, 0.1, 0.1]

import logging
import os

import matplotlib.pyplot as plt

from cods.od.cp import ODConformalizer
from cods.od.data import MSCOCODataset
from cods.od.models import DETRModel, YOLOModel

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = (
    "1"  # chose the GPU. If only one, then "0"
)

logger = logging.getLogger("cods")
logger.setLevel(logging.INFO)


COCO_PATH = "/datasets/shared_datasets/coco/"

MODEL_NAMES = ["yolov8x.pt", "detr_resnet50", "detr_resnet101"]


def setup_experiment(
    model_name,
    filter_by_confidence,
    config,
    name_of_experiment,
    override=False,
):
    # Check if file exists:
    if (
        os.path.exists(
            f"./final_experiments/figs/monotization_conf_{name_of_experiment}.png",
        )
        and not override
    ):
        logger.info(
            f"File {name_of_experiment} already exists. Set override=True to overwrite.",
        )
        return

    data = MSCOCODataset(root=COCO_PATH, split="val")
    calibration_ratio = 0.5
    data_cal, data_val = data.split_dataset(calibration_ratio, shuffle=False)

    if model_name not in MODEL_NAMES:
        raise ValueError(f"Model name {model_name} not in {MODEL_NAMES}")

    if model_name == "yolov8x.pt":
        model = YOLOModel(model_name=model_name, pretrained=True, device="cpu")
    else:
        model = DETRModel(model_name=model_name, pretrained=True, device="cpu")

    preds_cal = model.build_predictions(
        data_cal,
        dataset_name="mscoco",
        split_name="cal",
        batch_size=12,
        collate_fn=data._collate_fn,  # TODO: make this a default for COCO
        shuffle=False,
        force_recompute=False,  # False,
        deletion_method="nms",
        filter_preds_by_confidence=filter_by_confidence,
    )

    conf = ODConformalizer(
        guarantee_level="image",
        matching_function=config["matching_function"],
        localization_method=config["localization_method"],
        localization_prediction_set=config["localization_prediction_set"],
        classification_method="binary",
        classification_prediction_set=config["classification_prediction_set"],
        confidence_method=config["confidence_method"],
        multiple_testing_correction=None,
        backend="auto",
        optimizer="binary_search",
    )

    conf.calibrate(
        preds_cal,
        alpha_confidence=config["alpha_confidence"],
        alpha_localization=config["alpha_localization"],
        alpha_classification=config["alpha_classification"],
        verbose=True,
    )

    # First step: confidence, all risks
    plt.figure(figsize=(16, 9))
    plt.plot(
        conf.confidence_conformalizer.optimizer2_minus.all_lbds,
        conf.confidence_conformalizer.optimizer2_minus.all_risks_raw,
        c="b",
        linewidth=3,
    )
    plt.plot(
        conf.confidence_conformalizer.optimizer2_minus.all_lbds,
        conf.confidence_conformalizer.optimizer2_minus.all_risks_mon,
        c="r",
        linewidth=2,
    )

    plt.plot(
        conf.confidence_conformalizer.optimizer2_plus.all_lbds,
        conf.confidence_conformalizer.optimizer2_plus.all_risks_raw,
        c="b",
        linewidth=3,
        label="Maximum of risks, raw",
    )
    plt.plot(
        conf.confidence_conformalizer.optimizer2_plus.all_lbds,
        conf.confidence_conformalizer.optimizer2_plus.all_risks_mon,
        c="r",
        linewidth=2,
        label="Maximum of risks, monotonized",
    )
    ##TODO: temporary test
    plt.xlim(0.99, 1.0)

    plt.yscale("log")
    plt.legend()
    plt.savefig(
        f"./final_experiments/figs/monotization_conf_{name_of_experiment}.png",
    )

    # Second step: localization
    plt.figure(figsize=(16, 9))
    n = len(conf.localization_conformalizer.optimizer2.all_risks_raw) / 13
    n = int(n)
    for i in range(13):
        plt.plot(
            list(
                reversed(
                    conf.localization_conformalizer.optimizer2.all_risks_raw[
                        i * n : (i + 1) * n
                    ],
                ),
            ),
            c="b",
            alpha=1 - 0.5 * i / 13,
            label="Raw Loss" if i == 0 else None,
        )
        plt.plot(
            list(
                reversed(
                    conf.localization_conformalizer.optimizer2.all_risks_mon[
                        i * n : (i + 1) * n
                    ],
                ),
            ),
            c="r",
            alpha=1 - 0.5 * i / 13,
            label="Monotonized Loss" if i == 0 else None,
        )
    plt.legend()
    plt.savefig(
        f"./final_experiments/figs/monotization_loc_{name_of_experiment}.png",
    )
    plt.close()

    # Second step bis: classification
    plt.figure(figsize=(16, 9))
    n = len(conf.classification_conformalizer.optimizer2.all_risks_raw) // 25
    for i in range(25):
        plt.plot(
            list(
                reversed(
                    conf.classification_conformalizer.optimizer2.all_risks_raw[
                        i * n : (i + 1) * n
                    ],
                ),
            ),
            c="b",
            alpha=1 - 0.5 * i / 25,
            label="Raw Loss" if i == 0 else None,
        )
        plt.plot(
            list(
                reversed(
                    conf.classification_conformalizer.optimizer2.all_risks_mon[
                        i * n : (i + 1) * n
                    ],
                ),
            ),
            c="r",
            alpha=1 - 0.5 * i / 25,
            label="Monotonized Loss" if i == 0 else None,
        )
    plt.legend()
    plt.savefig(
        f"./final_experiments/figs/monotization_cls_{name_of_experiment}.png",
    )
    plt.close()


def experiment_models():
    MODELS = ["yolov8x.pt", "detr_resnet50"]
    for model in MODELS:
        config = {
            "matching_function": "mix",
            "localization_method": "pixelwise",
            "localization_prediction_set": "multiplicative",
            "classification_prediction_set": "lac",
            "confidence_method": "box_count_recall",
            "alpha_confidence": 0.06,
            "alpha_localization": 0.1,
            "alpha_classification": 0.1,
        }
        setup_experiment(
            model_name=model,
            filter_by_confidence=None,  # 1e-3,
            config=config,
            name_of_experiment=f"comparing_models_{model}",
        )


def experiment_losses():
    LOSSES = [
        ["box_count_twosided_recall", "pixelwise", "lac"],
        ["box_count_recall", "boxwise", "aps"],
        ["box_count_threshold", "pixelwise", "lac"],
        # ["boxwise-precision", "lac"],
        # ["boxwise-iou", "aps"],
    ]
    for cnf_loss, loc_loss, cls_loss in LOSSES:
        config = {
            "matching_function": "mix",
            "localization_method": loc_loss,
            "localization_prediction_set": "multiplicative",
            "classification_prediction_set": cls_loss,
            "confidence_method": cnf_loss,
            "alpha_confidence": 0.06,
            "alpha_localization": 0.1,
            "alpha_classification": 0.1,
        }
        setup_experiment(
            model_name="detr_resnet50",
            filter_by_confidence=None,  # 1e-3,
            config=config,
            name_of_experiment=f"comparing_losses_{cnf_loss}_{loc_loss}_{cls_loss}",
        )


def experiment_matchings():
    MATCHING = ["hausdorff", "lac", "mix", "giou"]
    for matching in MATCHING:
        config = {
            "matching_function": matching,
            "localization_method": "pixelwise",
            "localization_prediction_set": "multiplicative",
            "classification_prediction_set": "lac",
            "confidence_method": "box_count_recall",
            "alpha_confidence": 0.06,
            "alpha_localization": 0.1,
            "alpha_classification": 0.1,
        }
        setup_experiment(
            model_name="detr_resnet50",
            filter_by_confidence=None,  # 1e-3,
            config=config,
            name_of_experiment=f"comparing_matchings_{matching}",
        )


def experiment_filtering():
    FILTERINGS = [None, 1e-3]
    for filtering in FILTERINGS:
        config = {
            "matching_function": "mix",
            "localization_method": "pixelwise",
            "localization_prediction_set": "multiplicative",
            "classification_prediction_set": "lac",
            "confidence_method": "box_count_recall",
            "alpha_confidence": 0.06,
            "alpha_localization": 0.1,
            "alpha_classification": 0.1,
        }
        setup_experiment(
            model_name="detr_resnet50",
            filter_by_confidence=filtering,
            config=config,
            name_of_experiment=f"comparing_filterings_{filtering}",
        )


def experiment_alphas():
    ALPHAS = [
        [0.03, 0.05, 0.05],
        [0.03, 0.1, 0.1],
    ]
    for alphas in ALPHAS:
        config = {
            "matching_function": "mix",
            "localization_method": "pixelwise",
            "localization_prediction_set": "multiplicative",
            "classification_prediction_set": "lac",
            "confidence_method": "box_count_recall",
            "alpha_confidence": alphas[0],
            "alpha_localization": alphas[1],
            "alpha_classification": alphas[2],
        }
        setup_experiment(
            model_name="detr_resnet50",
            filter_by_confidence=1e-3,
            config=config,
            name_of_experiment=f"comparing_alphas_{alphas[0]}_{alphas[1]}_{alphas[2]}",
        )


if __name__ == "__main__":
    logger.debug("Skipping Model Experiments")
    # logger.info("Starting Model Experiments")
    # experiment_models()
    logger.info("Starting Losses Experiments")
    experiment_losses()
    logger.info("Starting Matchings Experiments")
    experiment_matchings()
    logger.info("Starting Filtering Experiments")
    experiment_filtering()
    logger.info("Starting Alphas Experiments")
    experiment_alphas()
