import pickle
from pathlib import Path

from cods.od.data.bdd100k_dataset import BDD100KDataset
from cods.od.models.bdd100k_model import DETRBDD100KModel
from cods.od.data.voc_dataset import VOCDataset
from cods.od.models.voc_model import DETRVOCModel

DEVICE = "cpu"

BASE_OUT_DIR = Path("/home/thomas.mullor/UQ/cods/grid_results")

DATASET_NAME = "bdd100k"
# VOC
MODEL_NAME = "detr_resnet50"

N_CALIB_TEST = 400
BATCH_SIZE = 4
FILTER_PREDS_BY_CONFIDENCE = 4e-2

if DATASET_NAME == "bdd100k":
    PRED_CACHE = BASE_OUT_DIR / "bdd100k_detr_ray_grid"  / "predictions.pkl"
    BDD_ROOT = "/datasets/shared_datasets/bdd100k"
    dataset_class = BDD100KDataset
    model_class = DETRBDD100KModel
elif DATASET_NAME == "voc":
    PRED_CACHE = BASE_OUT_DIR / "voc_detr_ray_grid" / "predictions.pkl"
    BDD_ROOT = "/datasets/shared_datasets"
    dataset_class = VOCDataset
    model_class = DETRVOCModel
else:
    raise ValueError(f"Unknown dataset: {DATASET_NAME}")

def main():
    print("BUILDING PREDICTIONS CACHE")

    print(f"Loading {DATASET_NAME} dataset")
    dataset = dataset_class(
        root=BDD_ROOT,
        split="val",
    )

    print("Splitting dataset (cal / val)")
    data_cal, data_val = dataset.split_dataset(
        proportion=0.5,
        shuffle=False,
        n_calib_test=min(N_CALIB_TEST, len(dataset)),
    )

    print(f"CAL size: {len(data_cal)}")
    print(f"VAL size: {len(data_val)}")

    print("Loading DETR")
    model = model_class(
        model_name=MODEL_NAME,
        pretrained=True,
        device=DEVICE,
    )

    print("Building calibration predictions")
    preds_cal = model.build_predictions(
        data_cal,
        dataset_name=DATASET_NAME,
        split_name="bdd_ray_cal",
        batch_size=BATCH_SIZE,
        collate_fn=dataset._collate_fn,
        shuffle=False,
        force_recompute=True,
        deletion_method="nms",
        filter_preds_by_confidence=FILTER_PREDS_BY_CONFIDENCE,
    )

    print("Building validation predictions")
    preds_val = model.build_predictions(
        data_val,
        dataset_name=DATASET_NAME,
        split_name="ray_val",
        batch_size=BATCH_SIZE,
        collate_fn=dataset._collate_fn,
        shuffle=False,
        force_recompute=True,
        deletion_method="nms",
        filter_preds_by_confidence=FILTER_PREDS_BY_CONFIDENCE,
    )

    print("Saving predictions ...")

    with open(PRED_CACHE, "wb") as f:
        pickle.dump(
            {
                "preds_cal": preds_cal,
                "preds_val": preds_val,
            },
            f,
        )

    print(f"Saved at: {PRED_CACHE}")


if __name__ == "__main__":
    main()