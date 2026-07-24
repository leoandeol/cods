"""Per-task-alpha COCO experiment (different alpha per task), DETR-R101, both SeqCRC and LTT.

Spec:
  dataset=coco  model=detr_resnet101  matching=mix
  confidence=box_count_recall (no fixed threshold)
  localization=pixelwise / multiplicative   classification=binary / lac
  alpha_confidence=0.03  alpha_localization=0.10  alpha_classification=0.10   (delta=0.10 for LTT)
"""

import csv
import json
import logging
import os
import urllib.request

import torch

logging.getLogger().setLevel(logging.WARNING)

ROOT = "data/coco"
ANN = f"{ROOT}/annotations/instances_val2017.json"
IMG_DIR = f"{ROOT}/val2017"
N_TOTAL = int(os.environ.get("N_TOTAL", "600"))
BS = int(os.environ.get("BS", "8"))
MODEL = os.environ.get("MODEL", "detr_resnet101")
DEVICE = os.environ.get("DEVICE", "cpu")  # e.g. "cuda:0" for GPU inference
PREFILT = float(os.environ.get("PREFILT", "1e-3"))
MAX_BOXES = int(os.environ.get("MAX_BOXES", "120"))
A_CNF, A_LOC, A_CLS = 0.03, 0.10, 0.10
DELTA = 0.10


def cap_boxes(preds, k):
    """Keep the top-k highest-confidence predictions per image (in place). A fixed
    per-image, label-free transform applied identically to every split, so calibration
    and test predictions stay exchangeable."""
    for i in range(len(preds)):
        if len(preds.confidences[i]) > k:
            idx = torch.topk(preds.confidences[i], k).indices.sort().values
            preds.pred_boxes[i] = preds.pred_boxes[i][idx]
            preds.confidences[i] = preds.confidences[i][idx]
            preds.pred_cls[i] = preds.pred_cls[i][idx]
    return preds


def ensure_images():
    with open(ANN) as f:
        ann = json.load(f)
    images = ann["images"][:N_TOTAL]
    os.makedirs(IMG_DIR, exist_ok=True)
    for im in images:
        dst = os.path.join(IMG_DIR, im["file_name"])
        if not os.path.exists(dst):
            url = im.get("coco_url") or f"http://images.cocodataset.org/val2017/{im['file_name']}"
            urllib.request.urlretrieve(url, dst)
    return [im["id"] for im in images]


def build():
    from cods.od.data import MSCOCODataset
    from cods.od.models import DETRModel

    image_ids = ensure_images()
    data = MSCOCODataset(root=ROOT, split="val", image_ids=image_ids)
    data_cal, data_val = data.split_dataset(0.5, shuffle=False)
    model = DETRModel(model_name=MODEL, pretrained=True, device=DEVICE)
    kw = {
        "dataset_name": "mscoco",
        "batch_size": BS,
        "collate_fn": data._collate_fn,
        "shuffle": False,
        "force_recompute": False,
        "deletion_method": "nms",
        "filter_preds_by_confidence": PREFILT,
    }
    preds_cal = model.build_predictions(data_cal, split_name="cal", **kw)
    preds_val = model.build_predictions(data_val, split_name="test", **kw)
    return cap_boxes(preds_cal, MAX_BOXES), cap_boxes(preds_val, MAX_BOXES)


def m(x):
    return float(torch.mean(x)) if x is not None else float("nan")


CFG = {
    "guarantee_level": "image",
    "matching_function": "mix",
    "confidence_method": "box_count_recall",
    "localization_method": "pixelwise",
    "localization_prediction_set": "multiplicative",
    "classification_method": "binary",
    "classification_prediction_set": "lac",
}


def run_seqcrc(preds_cal, preds_val):
    from cods.od.cp import ODConformalizer

    c = ODConformalizer(backend="auto", multiple_testing_correction=None, **CFG)
    preds_cal.matching = None
    p = c.calibrate(
        preds_cal,
        alpha_confidence=A_CNF,
        alpha_localization=A_LOC,
        alpha_classification=A_CLS,
        verbose=False,
    )
    preds_val.matching = None
    cpv = c.conformalize(preds_val, parameters=p, verbose=False)
    r = c.evaluate(
        preds_val,
        parameters=p,
        conformalized_predictions=cpv,
        include_confidence_in_global=False,
        verbose=False,
    )
    return p, r, True, (A_CNF, A_LOC, A_CLS)


def run_ltt(preds_cal, preds_val):
    from cods.od.cp import LearnThenTestConformalizer

    c = LearnThenTestConformalizer(
        n_lambda_confidence=40,
        n_lambda_localization=40,
        n_lambda_classification=40,
        lambda_localization_max=2.0,
        **CFG,
    )
    preds_cal.matching = None
    try:
        p = c.calibrate(
            preds_cal,
            alpha_confidence=A_CNF,
            alpha_localization=A_LOC,
            alpha_classification=A_CLS,
            global_delta=DELTA,
            verbose=False,
        )
    except ValueError as e:
        return None, None, False, str(e)
    preds_val.matching = None
    cpv = c.conformalize(preds_val, parameters=p, verbose=False)
    r = c.evaluate(
        preds_val,
        parameters=p,
        conformalized_predictions=cpv,
        include_confidence_in_global=False,
        verbose=False,
    )
    return p, r, True, (A_CNF, A_LOC, A_CLS)


def main():
    preds_cal, preds_val = build()
    print(
        f"n_cal={len(preds_cal)} n_test={len(preds_val)} n_classes={preds_val.n_classes} model={MODEL}"
    )

    seq = run_seqcrc(preds_cal, preds_val)
    print("SeqCRC done", flush=True)
    ltt = run_ltt(preds_cal, preds_val)
    print("LTT done", flush=True)

    # -------- TABLE A: full parameter spec (one row per method) --------
    cols = [
        "dataset",
        "model",
        "method",
        "mode",
        "matching_function",
        "confidence_method",
        "fixed_conf_thr",
        "localization_method",
        "localization_pred_set",
        "classification_method",
        "classification_pred_set",
        "alpha_confidence",
        "alpha_localization",
        "alpha_classification",
    ]
    base = {
        "dataset": "coco",
        "model": MODEL,
        "matching_function": CFG["matching_function"],
        "confidence_method": CFG["confidence_method"],
        "fixed_conf_thr": "-",
        "localization_method": CFG["localization_method"],
        "localization_pred_set": CFG["localization_prediction_set"],
        "classification_method": CFG["classification_method"],
        "classification_pred_set": CFG["classification_prediction_set"],
        "alpha_confidence": A_CNF,
        "alpha_localization": A_LOC,
        "alpha_classification": A_CLS,
    }
    spec = {
        "SeqCRC": {**base, "method": "split_triple_crc", "mode": "crc"},
        "LTT": {**base, "method": "split_triple_ltt", "mode": f"ltt (delta={DELTA})"},
    }
    print("\n########## TABLE A - Experiment parameters ##########\n")
    print(f"| {'field':<24} | {'SeqCRC':<22} | {'LTT':<22} |")
    print(f"|{'-' * 26}|{'-' * 24}|{'-' * 24}|")
    for col in cols:
        print(f"| {col:<24} | {spec['SeqCRC'][col]!s:<22} | {spec['LTT'][col]!s:<22} |")

    # -------- TABLE B: calibrated lambdas + results --------
    print("\n\n########## TABLE B - Calibrated lambdas & test results ##########\n")
    hdr = (
        f"| {'Method':<7} | {'cert?':<5} | {'thr':>5} | {'lam_loc':>7} | {'lam_cls':>7} "
        f"| {'R_cnf':>6} | {'R_loc':>6} | {'R_cls':>6} | {'R_glob':>6} "
        f"| {'|cnf|':>6} | {'|loc|':>6} | {'|cls|':>6} |"
    )
    print(hdr)
    print("|" + "|".join("-" * len(c) for c in hdr.split("|")[1:-1]) + "|")
    rows = []
    for name, (par, res, ok, info) in [("SeqCRC", seq), ("LTT", ltt)]:
        row = {
            **spec[name],
            "n_cal": len(preds_cal),
            "n_test": len(preds_val),
            "n_classes": int(preds_val.n_classes),
            "conf_prefilter": PREFILT,
            "delta": DELTA if name == "LTT" else "",
            "certified": ok,
        }
        if ok:
            print(
                f"| {name:<7} | {'yes':<5} | {par.confidence_threshold:>5.3f} "
                f"| {par.lambda_localization:>7.3f} | {par.lambda_classification:>7.4f} "
                f"| {m(res.confidence_coverages):>6.3f} | {m(res.localization_coverages):>6.3f} "
                f"| {m(res.classification_coverages):>6.3f} | {m(res.global_coverage):>6.3f} "
                f"| {m(res.confidence_set_sizes):>6.2f} | {m(res.localization_set_sizes):>6.3f} "
                f"| {m(res.classification_set_sizes):>6.2f} |"
            )
            row.update(
                confidence_threshold=round(float(par.confidence_threshold), 4),
                lambda_localization=round(float(par.lambda_localization), 4),
                lambda_classification=round(float(par.lambda_classification), 4),
                R_cnf=round(m(res.confidence_coverages), 4),
                R_loc=round(m(res.localization_coverages), 4),
                R_cls=round(m(res.classification_coverages), 4),
                R_glob=round(m(res.global_coverage), 4),
                size_cnf=round(m(res.confidence_set_sizes), 3),
                size_loc=round(m(res.localization_set_sizes), 4),
                size_cls=round(m(res.classification_set_sizes), 3),
                reason="",
            )
        else:
            print(f"| {name:<7} | {'NO':<5} | not certifiable at these (alpha, delta, n) |")
            row["reason"] = info
        rows.append(row)

    fields = [
        *cols,
        "n_cal",
        "n_test",
        "n_classes",
        "conf_prefilter",
        "delta",
        "certified",
        "confidence_threshold",
        "lambda_localization",
        "lambda_classification",
        "R_cnf",
        "R_loc",
        "R_cls",
        "R_glob",
        "size_cnf",
        "size_loc",
        "size_cls",
        "reason",
    ]
    with open("coco_triple_results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print("\n[wrote coco_triple_results.csv]")
    print(
        "thr=confidence_threshold; lam_loc=mult. box margin; lam_cls=LAC quantile. "
        "R_*=empirical test risk\n(targets a_cnf=0.03, a_loc=0.10, a_cls=0.10). "
        "|cnf|=mean #kept boxes; |loc|=mean box-area factor; |cls|=mean label-set size."
    )


if __name__ == "__main__":
    main()
