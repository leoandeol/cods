import argparse
import json
import pickle
from itertools import product
from logging import getLogger
from time import time

import numpy as np
import torch
from tqdm import tqdm

import wandb
from cods.od.cp import ODConformalizer
from cods.od.data import MSCOCODataset
from cods.od.metrics import get_recall_precision
from cods.od.models import DETRModel, YOLOModel

logger = getLogger("cods")

MODES = ["classification", "localization", "detection"]


class Benchmark:
    DATASETS = {
        "mscoco": MSCOCODataset,
    }

    MODELS = {"detr": DETRModel, "yolo": YOLOModel}

    def __init__(self, config, device):
        self.config = config
        self.device = device
        logger.info("Loaded config")

        self.run_id = "experiment-" + wandb.util.generate_id()

    def run(self, threads=1):
        torch.set_grad_enabled(False)
        if threads > 1:
            raise NotImplementedError("Multithreading not implemented yet")

        experiments = []
        # TODO(leo): Add model_name to the config
        param_combinations = product(
            self.config["dataset"],
            self.config["model"],
            self.config["alphas"],
            self.config["guarantee_level"],
            self.config["matching_function"],
            self.config["confidence_method"],
            self.config["localization_method"],
            self.config["classification_method"],
            self.config["localization_prediction_set"],
            self.config["classification_prediction_set"],
            self.config["batch_size"],
            self.config["optimizer"],
            self.config["nms_iou_threshold"],
        )

        for combination in param_combinations:
            (
                dataset,
                model,
                alphas,
                guarantee_level,
                matching_function,
                confidence_method,
                localization_method,
                classification_method,
                localization_prediction_set,
                classification_prediction_set,
                batch_size,
                optimizer,
                nms_iou_threshold,
            ) = combination
            if dataset not in self.DATASETS:
                raise ValueError(
                    f"Invalid dataset: {dataset}, must be one of {self.DATASETS.keys()}",
                )
            if model not in self.MODELS:
                raise ValueError(
                    f"Invalid model: {model}, must be one of {self.MODELS.keys()}",
                )
            experiments.append(
                {
                    "dataset": dataset,
                    "model": model,
                    "alphas": alphas,
                    "guarantee_level": guarantee_level,
                    "matching_function": matching_function,
                    "confidence_method": confidence_method,
                    "localization_method": localization_method,
                    "classification_method": classification_method,
                    "localization_prediction_set": localization_prediction_set,
                    "classification_prediction_set": classification_prediction_set,
                    "batch_size": batch_size,
                    "optimizer": optimizer,
                    "nms_iou_threshold": nms_iou_threshold,
                },
            )

        # with concurrent.futures.ProcessPoolExecutor(max_workers=8) as executor:
        #     executor.map(self.run_experiment, experiments)

        # pickle the results with a unique name
        unique_name = str(time())

        results = []
        for experiment in tqdm(experiments):
            # try:
            experiment_result = self.run_experiment(experiment)
            results.append(experiment_result)
            # updating the pickle file with the results
            logger.info("Updating pickle file with results")
            with open(f"results-{unique_name}.pkl", "wb") as f:
                pickle.dump(results, f)
            # except Exception as e:
            # logger.error(f"Experiment failed: {e}")

    def run_experiment(self, experiment, verbose=False):
        if verbose:
            # print config
            logger.info("Running experiment with config:")
            for key, value in experiment.items():
                logger.info(f"{key}: {value}")

        wandb.init(
            project="cods-benchmark",
            reinit=True,
            config=experiment,
            group=self.run_id,
        )
        wandb.log({"run_id": self.run_id})

        # Log detailed information about ongoing experiment (parameters)
        logger.info("")
        # Load dataset
        if experiment["dataset"] not in self.DATASETS:
            raise NotImplementedError(
                f"Dataset {experiment['dataset']} not implemented yet.",
            )
        dataset = self.DATASETS[experiment["dataset"]](
            root="/datasets/shared_datasets/coco/",
            split="val",
        )
        data_cal, data_val = dataset.split_dataset(0.5, shuffle=False)

        # Load model
        if experiment["model"] not in self.MODELS:
            raise NotImplementedError(
                f"Model { experiment['model']} not implemented yet.",
            )
        model = self.MODELS[experiment["model"]](
            pretrained=True,
            device=self.device,
        )

        # Build predictions
        batch_size = experiment["batch_size"]
        preds_cal = model.build_predictions(
            data_cal,
            dataset_name=experiment["dataset"],
            split_name="cal",
            batch_size=batch_size,
            collate_fn=dataset._collate_fn,
            shuffle=False,
            iou_threshold=experiment["nms_iou_threshold"],
            deletion_method="bayesod"
            if experiment["localization_prediction_set"] == "uncertainty"
            else "nms",
        )
        preds_val = model.build_predictions(
            data_val,
            dataset_name=experiment["dataset"],
            split_name="test",
            batch_size=batch_size,
            collate_fn=dataset._collate_fn,
            shuffle=False,
            iou_threshold=experiment["nms_iou_threshold"],
            deletion_method="bayesod"
            if experiment["localization_prediction_set"] == "uncertainty"
            else "nms",
        )

        conf = ODConformalizer(
            guarantee_level=experiment["guarantee_level"],
            matching_function=experiment["matching_function"],
            confidence_method=experiment["confidence_method"],
            localization_method=experiment["localization_method"],
            classification_method=experiment["classification_method"],
            localization_prediction_set=experiment[
                "localization_prediction_set"
            ],
            classification_prediction_set=experiment[
                "classification_prediction_set"
            ],
            optimizer=experiment["optimizer"],
            device=self.device,
        )

        alphas = experiment["alphas"]
        alpha_confidence = alphas[0]
        alpha_localization = alphas[1]
        alpha_classification = alphas[2]
        parameters = conf.calibrate(
            preds_cal,
            alpha_confidence=alpha_confidence,
            alpha_localization=alpha_localization,
            alpha_classification=alpha_classification,
            verbose=verbose,
        )

        conformal_preds = conf.conformalize(
            preds_val,
            parameters=parameters,
            verbose=verbose,
        )

        results_val = conf.evaluate(
            preds_val,
            parameters=parameters,
            conformalized_predictions=conformal_preds,
            include_confidence_in_global=False,
            verbose=verbose,
        )
        results = {
            "confidence_set_sizes": torch.mean(
                results_val.confidence_set_sizes,
            ),
            "confidence_losses": results_val.confidence_coverages,
            "localization_set_sizes": torch.mean(
                results_val.localization_set_sizes,
            ),
            "localization_losses": results_val.localization_coverages,
            "classification_set_sizes": torch.mean(
                results_val.classification_set_sizes,
            ),
            "classification_losses": results_val.classification_coverages,
            "confidence_mean_risk": torch.mean(
                results_val.confidence_coverages,
            ),
            "confidence_std_risk": torch.std(results_val.confidence_coverages),
            "localization_mean_risk": torch.mean(
                results_val.localization_coverages,
            ),
            "localization_std_risk": torch.std(
                results_val.localization_coverages,
            ),
            "classification_mean_risk": torch.mean(
                results_val.classification_coverages,
            ),
            "classification_std_risk": torch.std(
                results_val.classification_coverages,
            ),
            "global_losses": results_val.global_coverage,
            "global_mean_risk": torch.mean(results_val.global_coverage),
            "global_std_risk": torch.std(results_val.global_coverage),
        }

        wandb.log(results)

        # Trop couteux
        # metrics = unroll_metrics(
        #     predictions=preds_val,
        #     conformalized_predictions=conformal_preds,
        #     verbose=verbose,
        # )metrics = unroll_metrics(
        #     predictions=preds_val,
        #     conformalized_predictions=conformal_preds,
        #     verbose=verbose,
        # )
        # Just recall-precision for now
        recalls, precisions, scores = get_recall_precision(
            preds_val,
            SCORE_THRESHOLD=preds_val.confidence_threshold,
            IOU_THRESHOLD=experiment["nms_iou_threshold"],
        )

        metrics = {
            "recall": np.mean(recalls),
            "precision": np.mean(precisions),
        }

        wandb.log(metrics)

        experiment.update(results)
        experiment.update(metrics)

        wandb.finish()

        return experiment


def parse_args():
    parser = argparse.ArgumentParser(description="Run benchmark with config")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the JSON config file",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to use for training and evaluation",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    with open(args.config) as f:
        config = json.load(f)

    benchmark = Benchmark(config, device=args.device)
    benchmark.run()
