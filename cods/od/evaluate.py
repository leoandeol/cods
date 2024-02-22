import torch

from cods.od.data import MSCOCODataset
from cods.od.models import DETRModel
from cods.od.cp import ODConformalizer, ODRiskConformalizer
from cods.od.tr import ODToleranceRegion
from cods.od.metrics import unroll_metrics


# MODES = ["classification", "localization", "detection", "combined"]

MODES = ["classification", "localization", "detection"]

DEFAULT_CONFIG = [
    # {
    #     "dataset": "mscoco",
    #     "model": "detr",
    #     "alpha": 0.1,
    #     "mode" "objectness_threshold": 0.8,
    #     "margins": 1,
    #     "method": "min-hausdorff-additive",
    # }
    {
        "mode": "cp",  # or crc or tr
        "dataset": "mscoco",
        "model": "detr",
        "alpha": 0.1,
        "confidence_threshold": 0.8,
        "margins": 1,  # for cp only, otherwise ignored, default = 1 other values 2 or 4
        "method": "min-hausdorff-additive",  # for cp only, otherwise ignored, other values "min-hausdorff-multiplicative", "min-hausdorff-additive", "min-hausdorff-multiplicative"
        # for tr and crc must specify loss
        # for tr must specify inequality
    },
    {
        "mode": "crc",  # or crc or tr
        "dataset": "mscoco",
        "model": "detr",
        "alpha": 0.1,
        "confidence_threshold": 0.8,
        "margins": 1,  # for cp only, otherwise ignored, default = 1 other values 2 or 4
        # "method": "min-hausdorff-additive",  # for cp only, otherwise ignored, other values "min-hausdorff-multiplicative", "min-hausdorff-additive", "min-hausdorff-multiplicative"
        "loss": "boxwise",  # or pixelwise
        # for tr and crc must specify loss
        # for tr must specify inequality
    },
    {
        "mode": "tr",  # or crc or tr
        "dataset": "mscoco",
        "model": "detr",
        "alpha": 0.1,
        "confidence_threshold": 0.8,
        "margins": 1,  # for cp only, otherwise ignored, default = 1 other values 2 or 4
        # "method": "min-hausdorff-additive",  # for cp only, otherwise ignored, other values "min-hausdorff-multiplicative", "min-hausdorff-additive", "min-hausdorff-multiplicative"
        "loss": "boxwise",  # or pixelwise
        "inequality": "bernstein",  # or "hoeffding", "bernstein"
        # for tr must specify inequality
    },
]


class Benchmark:
    DATASETS = {
        "mscoco": MSCOCODataset,
    }

    MODELS = {
        "detr": DETRModel,
    }

    def __init__(self, configs):
        self.configs = configs

    def run(self, threads=1):

        torch.set_grad_enabled(False)
        if threads > 1:
            raise NotImplementedError("Multithreading not implemented yet")

        # ignored for now
        # print("Dataset:", config["dataset"])
        # print("Model:", config["model"])
        print("Currently ignored: defaulting to MSCOCO & DETR")
        # dataset_name = config["dataset"]
        # model_name = config["model"]

        # Load dataset
        dataset = MSCOCODataset(root="/datasets/shared_datasets/coco/", split="val")
        data_cal, data_val = dataset.random_split(0.5, shuffled=False)

        # Load model
        detr = DETRModel(model_name="detr_resnet50", pretrained=True)

        # Build predictions
        preds_cal = detr.build_predictions(
            data_cal,
            dataset_name="mscoco",
            split_name="cal",
            batch_size=12,
            collate_fn=dataset._collate_fn,
            shuffle=False,
        )
        preds_val = detr.build_predictions(
            data_val,
            dataset_name="mscoco",
            split_name="test",
            batch_size=12,
            collate_fn=dataset._collate_fn,
            shuffle=False,
        )

        # Run all experiments (calibration, conformalization, evaluation) for Combined OD Tasks (Classification, Localization, Detection)
        # with all configurations (Conformal Prediction, Conformal Risk Control, Tolerance Regions)

        alpha = 0.1
        delta = 0.1
        verbose = True
        all_metrics = {}
        for config in self.configs:
            alpha = config["alpha"]
            mode = config["mode"]
            objectness_threshold = config.get("objectness_threshold", None)
            # to finish
            margins = config.get("margins", None)
            method = config.get("method", None)
            loss = config.get("loss", None)
            inequality = config.get("inequality", None)

            # Calibration

            if mode == "cp":
                conformalizer = ODConformalizer(
                    localization_method=method,
                    objectness_method="box_number",
                    classification_method="lac",
                    multiple_testing_correction="bonferroni",
                    margins=margins,
                )
            elif mode == "tr":
                conformalizer = ODToleranceRegion(
                    localization_method=loss,
                    objectness_method="box_number",
                    classification_method="lac",
                    multiple_testing_correction="bonferroni",
                    inequality=inequality,
                )
            elif mode == "crc":
                conformalizer = ODRiskConformalizer(
                    localization_method=loss,
                    objectness_method="box_number",
                    classification_method="lac",
                    multiple_testing_correction="bonferroni",
                )
            else:
                raise ValueError(f"Invalid mode: {mode}")

            if mode == "tr":
                conformalizer.calibrate(
                    preds_cal, alpha=alpha, delta=delta, verbose=verbose
                )
            else:
                conformalizer.calibrate(preds_cal, alpha=alpha, verbose=verbose)
            conf_boxes, conf_cls = conformalizer.conformalize(preds_val)
            metrics = conformalizer.evaluate(
                preds_val, conf_boxes, conf_cls, verbose=verbose
            )
            all_metrics[f"{conformalizer.__class__.__name__}-{method}"] = metrics
            unroll_metrics(
                val_preds=preds_val, conf_boxes=conf_boxes, conf_cls=conf_cls
            )


if __name__ == "__main__":
    benchmark = Benchmark(DEFAULT_CONFIG)
    benchmark.run()
