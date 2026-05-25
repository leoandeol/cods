import json
import pickle
from pathlib import Path

from cods.od.data.predictions import ODResults
import ray
from ray import tune
import pandas as pd
import numpy as np
import torch

from cods.od.cp import ODConformalizer


N_CPUS = 64
CPUS_PER_TRIAL = 2

OUT_DIR = Path("/home/thomas.mullor/UQ/cods/grid_results/bdd100k_detr_ray_grid")
PRED_CACHE = OUT_DIR / "predictions.pkl"


def compute_metrics(results:ODResults):
    conf_loss = results.confidence_coverages.mean().item()
    loc_loss = results.localization_coverages.mean().item()
    cls_loss = results.classification_coverages.mean().item()

    conf_size = results.confidence_set_sizes.mean().item()
    loc_size = results.localization_set_sizes.mean().item()
    cls_size = results.classification_set_sizes.mean().item()

    global_loss = max(conf_loss, loc_loss, cls_loss)

    # A redefinir correctement
    score = (
        10.0 * conf_loss
        + 10.0 * loc_loss
        + 10.0 * cls_loss
        + 0.05 * conf_size
        + 0.50 * loc_size
        + 0.50 * cls_size
    )

    return {
        "score": score,
        "global_risk": global_loss,
        "global_coverage_proxy": 1.0 - global_loss,
        "confidence_loss": conf_loss,
        "localization_loss": loc_loss,
        "classification_loss": cls_loss,
        "confidence_coverage_proxy": 1.0 - conf_loss,
        "localization_coverage_proxy": 1.0 - loc_loss,
        "classification_coverage_proxy": 1.0 - cls_loss,
        "confidence_size_mean": conf_size,
        "localization_size_mean": loc_size,
        "classification_size_mean": cls_size,
        "raw_global_coverage_field": results.global_coverage.mean().item() ,
    }


def load_predictions():
    with open(PRED_CACHE, "rb") as f:
        payload = pickle.load(f)
    return payload["preds_cal"], payload["preds_val"]


def trainable(config):
    preds_cal, preds_val = load_predictions()

    try:
        conf = ODConformalizer(
            backend="auto",
            optimizer="binary_search",
            guarantee_level="image",
            matching_function=config["matching_function"],
            multiple_testing_correction=None,
            confidence_method=config["confidence_method"],
            localization_method=config["localization_method"],
            localization_prediction_set=config["localization_prediction_set"],
            classification_method="binary",
            classification_prediction_set=config["classification_prediction_set"],
        )

        params = conf.calibrate(
            preds_cal,
            alpha_confidence=config["alpha_confidence"],
            alpha_localization=config["alpha_localization"],
            alpha_classification=config["alpha_classification"],
        )

        conf_val = conf.conformalize(preds_val, parameters=params)

        results_val:ODResults = conf.evaluate(
            preds_val,
            parameters=params,
            conformalized_predictions=conf_val,
            include_confidence_in_global=False,
        )

        metrics = compute_metrics(results_val)

        tune.report({
            "status": "ok",
            **metrics,
        })

    except Exception as e:
        tune.report({
            "status": "failed",
            "score": 1e9,
            "global_risk": 1e9,
            "error_type": type(e).__name__,
            "error_message": str(e),
        })


search_space = {
    "matching_function": tune.grid_search(["mix"]),
    "confidence_method": tune.grid_search([
        "box_count_recall",
        "box_count_threshold",
    ]),
    "localization_method": tune.grid_search([
        "pixelwise",
        "boxwise",
        "thresholded",
    ]),
    "classification_prediction_set": tune.grid_search(["lac"]),
    "localization_prediction_set": tune.grid_search([
        "multiplicative",
        "additive",
    ]),
    "alpha_confidence": tune.grid_search([0.02, 0.03]),
    "alpha_localization": tune.grid_search([0.05, 0.06, 0.08]),
    "alpha_classification": tune.grid_search([0.07, 0.08, 0.09]),
}

def main():
    ray.init(
        num_cpus=N_CPUS,
        include_dashboard=False,
        ignore_reinit_error=True,
    )

    tuner = tune.Tuner(
        tune.with_resources(
            trainable,
            resources={"cpu": CPUS_PER_TRIAL},
        ),
        param_space=search_space,
        tune_config=tune.TuneConfig(
            metric="score",
            mode="min",
            max_concurrent_trials=N_CPUS // CPUS_PER_TRIAL,
        ),
        run_config=ray.air.RunConfig(
            name="bdd100k_detr_ray_grid",
            storage_path=str(OUT_DIR.resolve()),
            verbose=1,
        ),
    )

    result_grid = tuner.fit()

    rows = []
    for result in result_grid:
        row = {
            **result.config,
            **result.metrics,
            "log_dir": result.path,
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "all_results.csv", index=False)

    ok_df = df[df["status"] == "ok"].copy()
    ok_df = ok_df.sort_values("score", ascending=True)

    best = ok_df.iloc[0].to_dict()

    with open(OUT_DIR / "best_result.json", "w") as f:
        json.dump(best, f, indent=2, default=str)

    print("BEST CONFIG :")
    print(json.dumps({k: best[k] for k in search_space.keys()}, indent=2))

    print("BEST METRICS :")
    metric_keys = [
        "score",
        "global_risk",
        "global_coverage_proxy",
        "confidence_loss",
        "localization_loss",
        "classification_loss",
        "confidence_size_mean",
        "localization_size_mean",
        "classification_size_mean",
    ]
    print(json.dumps({k: best[k] for k in metric_keys}, indent=2))

    print(f"\nSaved table: {OUT_DIR / 'all_results.csv'}")

    ray.shutdown()


if __name__ == "__main__":
    main()