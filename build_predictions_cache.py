import pickle
from pathlib import Path

from cods.od.data import MSCOCODataset
from cods.od.data.bdd100k_dataset import BDD100KDataset
from cods.od.data.voc_dataset import VOCDataset
from cods.od.models.detr import BDD100KDETRModel, COCODETRModel, VOCDETRModel
from cods.od.models.yolo import BDD100KYOLOModel, COCOYOLOModel, VOCYOLOModel

DEVICE = "cpu"
BASE_OUT_DIR = Path("/home/thomas.mullor/UQ/cods/grid_results")

N_CALIB_TEST = 400
BATCH_SIZE = 4
FILTER_PREDS_BY_CONFIDENCE = 4e-2


EXPERIMENTS = [
    {
        "dataset": "voc",
        "root": "/datasets/shared_datasets",
        "dataset_class": VOCDataset,
        "dataset_kwargs": {"split": "val"},
        "models": {
            "detr_resnet101": VOCDETRModel,
            "yolov8x": VOCYOLOModel,
        },
    },
    {
        "dataset": "coco",
        "root": "/datasets/shared_datasets/COCO",
        "dataset_class": MSCOCODataset,
        "dataset_kwargs": {"split": "val"},
        "models": {
            "detr_resnet101": COCODETRModel,
            "yolov8x": COCOYOLOModel,
        },
    },
    {
        "dataset": "bdd100k",
        "root": "/datasets/shared_datasets/bdd100k",
        "dataset_class": BDD100KDataset,
        "dataset_kwargs": {"split": "val"},
        "models": {
            "detr_resnet101": BDD100KDETRModel,
            "yolov8x": BDD100KYOLOModel,
        },
    },
]


def model_name_for_constructor(model_key):
    if model_key == "detr_resnet101":
        return "detr_resnet101"
    if model_key == "yolov8x":
        return "yolov8x.pt"
    raise ValueError(model_key)


def build_one_cache(exp, model_key, model_class):
    dataset_name = exp["dataset"]
    root = exp["root"]

    out_dir = BASE_OUT_DIR / f"{dataset_name}_{model_key}"
    out_dir.mkdir(parents=True, exist_ok=True)

    pred_cache = out_dir / "predictions.pkl"

    print("\n" + "=" * 80)
    print(f"BUILD CACHE: dataset={dataset_name} model={model_key}")
    print("=" * 80)

    dataset = exp["dataset_class"](
        root=root,
        **exp["dataset_kwargs"],
    )

    data_cal, data_val = dataset.split_dataset(
        proportion=0.5,
        shuffle=False,
        n_calib_test=min(N_CALIB_TEST, len(dataset)),
    )

    print(f"CAL size: {len(data_cal)}")
    print(f"VAL size: {len(data_val)}")

    model = model_class(
        model_name=model_name_for_constructor(model_key),
        pretrained=True,
        device=DEVICE,
    )

    preds_cal = model.build_predictions(
        data_cal,
        dataset_name=dataset_name,
        split_name=f"{dataset_name}_{model_key}_cal",
        batch_size=BATCH_SIZE,
        collate_fn=dataset._collate_fn,
        shuffle=False,
        force_recompute=True,
        deletion_method="nms",
        filter_preds_by_confidence=FILTER_PREDS_BY_CONFIDENCE,
    )

    preds_val = model.build_predictions(
        data_val,
        dataset_name=dataset_name,
        split_name=f"{dataset_name}_{model_key}_val",
        batch_size=BATCH_SIZE,
        collate_fn=dataset._collate_fn,
        shuffle=False,
        force_recompute=True,
        deletion_method="nms",
        filter_preds_by_confidence=FILTER_PREDS_BY_CONFIDENCE,
    )

    with open(pred_cache, "wb") as f:
        pickle.dump(
            {
                "preds_cal": preds_cal,
                "preds_val": preds_val,
                "dataset": dataset_name,
                "model": model_key,
            },
            f,
        )

    print(f"Saved: {pred_cache}")


def main():
    for exp in EXPERIMENTS:
        for model_key, model_class in exp["models"].items():
            build_one_cache(exp, model_key, model_class)


if __name__ == "__main__":
    main()