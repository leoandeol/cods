import pickle
from logging import getLogger
from time import time

from tqdm import tqdm

import wandb

logger = getLogger("cods")

import torch

from cods.od.cp import ODConformalizer
from cods.od.data import MSCOCODataset
from cods.od.metrics import unroll_metrics
from cods.od.models import DETRModel

# MODES = ["classification", "localization", "detection", "combined"]

MODES = ["classification", "localization", "detection"]

DEFAULT_CONFIG = {
    "dataset": ["mscoco"],
    "model": ["detr"],
    "alphas": [[0.045, 0.05, 0.05]],
    "guarantee_level": ["object", "image"],
    "matching_function": ["hausdorff", "iou"],
    "confidence_method": ["nb_boxes", "better"],
    "localization_method": ["pixelwise", "boxwise", "thresholded"],
    "classification_method": ["binary"],
    "localization_prediction_set": [
        "additive",
        "multiplicative",
        # "uncertainty",
    ],
    "classification_prediction_set": ["lac", "aps"],
    # Below: fixed parameters
    "batch_size": 12,
    "optimizer": "binary_search",
}


class Benchmark:
    DATASETS = {
        "mscoco": MSCOCODataset,
    }

    MODELS = {
        "detr": DETRModel,
    }

    def __init__(self, config=None):
        if config is None:
            self.config = DEFAULT_CONFIG
            logger.info("Using default config")
        else:
            self.config = config
            logger.info("Using provided config")

        self.run_id = "experiment-" + wandb.util.generate_id()

    def run(self, threads=1):
        torch.set_grad_enabled(False)
        if threads > 1:
            raise NotImplementedError("Multithreading not implemented yet")

        experiments = []
        for dataset in self.config["dataset"]:
            if dataset not in self.DATASETS:
                raise ValueError(
                    f"Invalid dataset: {dataset}, must be one of {self.DATASETS.keys()}"
                )
            for model in self.config["model"]:
                if model not in self.MODELS:
                    raise ValueError(
                        f"Invalid model: {model}, must be one of {self.MODELS.keys()}"
                    )
                for alphas in self.config["alphas"]:
                    for guarantee_level in self.config["guarantee_level"]:
                        for matching_function in self.config[
                            "matching_function"
                        ]:
                            for confidence_method in self.config[
                                "confidence_method"
                            ]:
                                for localization_method in self.config[
                                    "localization_method"
                                ]:
                                    for classification_method in self.config[
                                        "classification_method"
                                    ]:
                                        for (
                                            localization_prediction_set
                                        ) in self.config[
                                            "localization_prediction_set"
                                        ]:
                                            for (
                                                classification_prediction_set
                                            ) in self.config[
                                                "classification_prediction_set"
                                            ]:
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
                                                        "batch_size": self.config[
                                                            "batch_size"
                                                        ],
                                                        "optimizer": self.config[
                                                            "optimizer"
                                                        ],
                                                    }
                                                )

        # pickle the results with a unique name
        unique_name = str(time())

        results = []
        for experiment in tqdm(experiments):
            experiment_result = self.run_experiment(experiment)
            results.append(experiment_result)
            # updating the pickle file with the results
            logger.info("Updating pickle file with results")
            with open(f"results-{unique_name}.pkl", "wb") as f:
                pickle.dump(results, f)

    def run_experiment(self, experiment, verbose=False):
        wandb.init(
            project="cods-benchmark",
            reinit=True,
            config=experiment,
            group=self.run_id,
        )
        wandb.log({"run_id": self.run_id})
        
        # Log detailed information about ongoing experiment (parameters)
        logger.info(f"")
        # Load dataset
        if experiment["dataset"] not in self.DATASETS:
            raise NotImplementedError(
                f"Dataset {experiment['dataset']} not implemented yet."
            )
        dataset = MSCOCODataset(
            root="/datasets/shared_datasets/coco/", split="val"
        )
        data_cal, data_val = dataset.random_split(0.5, shuffled=False)

        # Load model
        if experiment["model"] != "detr":
            raise NotImplementedError(
                f"Model { experiment['model']} not implemented yet."
            )
        model = DETRModel(model_name="detr_resnet50", pretrained=True)

        # Build predictions
        batch_size = experiment["batch_size"]
        preds_cal = model.build_predictions(
            data_cal,
            dataset_name="mscoco",
            split_name="cal",
            batch_size=batch_size,
            collate_fn=dataset._collate_fn,
            shuffle=False,
        )
        preds_val = model.build_predictions(
            data_val,
            dataset_name="mscoco",
            split_name="test",
            batch_size=batch_size,
            collate_fn=dataset._collate_fn,
            shuffle=False,
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
            preds_val, parameters=parameters, verbose=verbose
        )

        results_val = conf.evaluate(
            preds_val,
            parameters=parameters,
            conformalized_predictions=conformal_preds,
            include_confidence_in_global=False,
            verbose=verbose,
        )
        results = {
            "confidence_set_sizes": results_val.confidence_set_sizes,
            "confidence_losses": results_val.confidence_coverages,
            "localization_set_sizes": results_val.localization_set_sizes,
            "localization_losses": results_val.localization_coverages,
            "classification_set_sizes": results_val.classification_set_sizes,
            "classification_losses": results_val.classification_coverages,
            "confidence_mean_risk": torch.mean(
                results_val.confidence_coverages
            ),
            "confidence_std_risk": torch.std(results_val.confidence_coverages),
            "localization_mean_risk": torch.mean(
                results_val.localization_coverages
            ),
            "localization_std_risk": torch.std(
                results_val.localization_coverages
            ),
            "classification_mean_risk": torch.mean(
                results_val.classification_coverages
            ),
            "classification_std_risk": torch.std(
                results_val.classification_coverages
            ),
            "global_losses": results_val.global_coverage,
            "global_mean_risk": torch.mean(results_val.global_coverage),
            "global_std_risk": torch.std(results_val.global_coverage),
        }

        wandb.log(results)

        metrics = unroll_metrics(
            predictions=preds_val,
            conformalized_predictions=conformal_preds,
            verbose=verbose,
        )

        wandb.log(metrics)

        experiment.update(results)
        experiment.update(metrics)

        wandb.finish()

        return experiment


if __name__ == "__main__":
    benchmark = Benchmark(DEFAULT_CONFIG)
    benchmark.run()
