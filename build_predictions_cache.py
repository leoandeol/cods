import json
import pickle
import platform
from pathlib import Path

import torch
import torchvision
import ultralytics

from cods.od.data import MSCOCODataset
from cods.od.data.bdd100k_dataset import BDD100KDataset
from cods.od.data.voc_dataset import VOCDataset
from cods.od.models.detr import BDD100KDETRModel, COCODETRModel, VOCDETRModel
from cods.od.models.yolo import BDD100KYOLOModel, COCOYOLOModel, VOCYOLOModel

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available. This script requires a GPU to run.")

DEVICE = "cuda"
BASE_OUT_DIR = Path("/home/thomas.mullor/UQ/cods/results_topk_boxes")

#N_CALIB_TEST = 400

BATCH_SIZE = 16
FILTER_PREDS_BY_CONFIDENCE = 0

MAX_PREDS_PER_IMAGE = {
    "voc": 40,
    "coco": 60,
    "bdd100k": 60,
}

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
        "root": "/datasets/shared_datasets/coco",
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

EXPECTED_N_CLASSES = {
    "voc": 20,
    "bdd100k": 10,
    "coco": 91,
}


def check_predictions(preds, dataset_name, split_name):
    expected_n_classes = EXPECTED_N_CLASSES[dataset_name]

    n_images = len(preds.image_paths)

    assert n_images > 0, f"{split_name}: empty predictions"
    assert len(preds.pred_boxes) == n_images
    assert len(preds.pred_cls) == n_images
    assert len(preds.confidences) == n_images
    assert len(preds.true_boxes) == n_images
    assert len(preds.true_cls) == n_images

    n_empty_pred = 0
    n_empty_gt = 0

    for i in range(n_images):
        pred_boxes = preds.pred_boxes[i]
        pred_cls = preds.pred_cls[i]
        confidences = preds.confidences[i]
        true_boxes = preds.true_boxes[i]
        true_cls = preds.true_cls[i]

        assert pred_boxes.ndim == 2 and pred_boxes.shape[-1] == 4, (
            f"{split_name} image {i}: bad pred_boxes shape {pred_boxes.shape}"
        )
        assert pred_cls.ndim == 2 and pred_cls.shape[-1] == expected_n_classes, (
            f"{split_name} image {i}: bad pred_cls shape {pred_cls.shape}, "
            f"expected last dim {expected_n_classes}"
        )
        assert confidences.ndim == 1, (
            f"{split_name} image {i}: bad confidences shape {confidences.shape}"
        )
        assert pred_boxes.shape[0] == pred_cls.shape[0] == confidences.shape[0], (
            f"{split_name} image {i}: mismatch pred lengths "
            f"boxes={pred_boxes.shape[0]}, cls={pred_cls.shape[0]}, conf={confidences.shape[0]}"
        )

        assert true_boxes.ndim == 2 and true_boxes.shape[-1] == 4, (
            f"{split_name} image {i}: bad true_boxes shape {true_boxes.shape}"
        )
        assert true_cls.ndim == 1, (
            f"{split_name} image {i}: bad true_cls shape {true_cls.shape}"
        )
        assert true_boxes.shape[0] == true_cls.shape[0], (
            f"{split_name} image {i}: mismatch GT lengths"
        )

        if pred_boxes.shape[0] == 0:
            n_empty_pred += 1
        if true_boxes.shape[0] == 0:
            n_empty_gt += 1

        if pred_cls.numel() > 0:
            row_sums = pred_cls.sum(dim=-1)
            assert torch.isfinite(pred_cls).all(), f"{split_name} image {i}: non-finite pred_cls"
            assert torch.isfinite(pred_boxes).all(), f"{split_name} image {i}: non-finite pred_boxes"
            assert torch.isfinite(confidences).all(), f"{split_name} image {i}: non-finite confidences"
            assert (row_sums >= 0).all(), f"{split_name} image {i}: negative class prob sum"

        if true_cls.numel() > 0:
            assert int(true_cls.min()) >= 0, f"{split_name} image {i}: negative true class"
            assert int(true_cls.max()) < expected_n_classes, (
                f"{split_name} image {i}: true class out of range, "
                f"max={int(true_cls.max())}, expected < {expected_n_classes}"
            )

    stats = {
        "split": split_name,
        "n_images": n_images,
        "n_empty_pred": n_empty_pred,
        "n_empty_gt": n_empty_gt,
        "mean_pred_per_image": float(sum(len(x) for x in preds.pred_boxes) / n_images),
        "mean_gt_per_image": float(sum(len(x) for x in preds.true_boxes) / n_images),
        "expected_n_classes": expected_n_classes,
    }

    print(f"\nSANITY CHECK - {split_name}")
    for k, v in stats.items():
        print(f"  {k}: {v}")

    return stats


def build_metadata(dataset_name, model_key, model_class, data_cal, data_val, preds_cal, preds_val):
    return {
        "dataset": dataset_name,
        "model_key": model_key,
        "model_class": model_class.__name__,
        "n_cal": len(data_cal),
        "n_val": len(data_val),
        "cal_image_paths": list(preds_cal.image_paths),
        "val_image_paths": list(preds_val.image_paths),
        "deletion_method": "nms",
        "batch_size": BATCH_SIZE,
        "device": DEVICE,
        "torch_version": torch.__version__,
        "torchvision_version": torchvision.__version__,
        "ultralytics_version": ultralytics.__version__,
        "python_version": platform.python_version(),
        "filtering_strategy": "topk_per_image",
        "filter_preds_by_confidence": FILTER_PREDS_BY_CONFIDENCE,
        "max_preds_per_image": MAX_PREDS_PER_IMAGE[dataset_name],
        "saved_on_cpu": True,
    }

def model_name_for_constructor(model_key):
    if model_key == "detr_resnet101":
        return "detr_resnet101"
    if model_key == "yolov8x":
        return "yolov8x.pt"
    raise ValueError(model_key)

def keep_topk_predictions(preds, k:int):
    new_pred_boxes = []
    new_pred_cls = []
    new_confidences = []

    for boxes, cls, conf in zip(
        preds.pred_boxes,
        preds.pred_cls,
        preds.confidences,
    ):
        if len(conf) <= k:
            new_pred_boxes.append(boxes)
            new_pred_cls.append(cls)
            new_confidences.append(conf)
            continue

        order = torch.argsort(conf, descending=True)[:k]

        new_pred_boxes.append(boxes[order])
        new_pred_cls.append(cls[order])
        new_confidences.append(conf[order])

    preds.pred_boxes = new_pred_boxes
    preds.pred_cls = new_pred_cls
    preds.confidences = new_confidences

    preds.matching = None
    return preds

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
        n_calib_test = None
        #n_calib_test=min(N_CALIB_TEST, len(dataset)),
    )

    print(f"CAL size: {len(data_cal)}")
    print(f"VAL size: {len(data_val)}")

    model = model_class(
        model_name=model_name_for_constructor(model_key),
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
    preds_cal = keep_topk_predictions(preds_cal, k = MAX_PREDS_PER_IMAGE[dataset_name])
    preds_val = keep_topk_predictions(preds_val, k = MAX_PREDS_PER_IMAGE[dataset_name])

    preds_cal = preds_cal.to("cpu")
    preds_val = preds_val.to("cpu")

    cal_stats = check_predictions(
        preds_cal,
        dataset_name=dataset_name,
        split_name="cal",
    )

    val_stats = check_predictions(
        preds_val,
        dataset_name=dataset_name,
        split_name="val",
    )

    metadata = build_metadata(
        dataset_name=dataset_name,
        model_key=model_key,
        model_class=model_class,
        data_cal=data_cal,
        data_val=data_val,
        preds_cal=preds_cal,
        preds_val=preds_val,
    )

    metadata["cal_stats"] = cal_stats
    metadata["val_stats"] = val_stats

    with open(pred_cache, "wb") as f:
        pickle.dump(
            {
                "preds_cal": preds_cal,
                "preds_val": preds_val,
                "metadata": metadata
            },
            f,
        )
    metadata_path = out_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved predictions: {pred_cache}")
    print(f"Saved metadata:    {metadata_path}")


def main():
    for exp in EXPERIMENTS:
        for model_key, model_class in exp["models"].items():
            build_one_cache(exp, model_key, model_class)


if __name__ == "__main__":
    main()