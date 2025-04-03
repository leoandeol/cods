from cods.od.data import MSCOCODataset
from cods.od.models import YOLOModel, DETRModel
import logging
import os
from cods.od.utils import (
    match_predictions_to_true_boxes,
    generalized_iou,
    assymetric_hausdorff_distance_old,
)
import numpy as np
from cods.od.loss import (
    ODBinaryClassificationLoss,
    ClassificationLossWrapper,
    PixelWiseRecallLoss,
)
from  cods.od.cp import ODConformalizer
import torch
import pickle
import matplotlib.pyplot as plt
import torch.nn.functional as F

def build_preds(model):
    preds_cal = model.build_predictions(
        data_cal,
        dataset_name="mscoco",
        split_name="cal",
        batch_size=12,
        collate_fn=data._collate_fn,  # TODO: make this a default for COCO
        shuffle=False,
        force_recompute=False,  # False,
        deletion_method="nms",
    )
    preds_val = model.build_predictions(
        data_val,
        dataset_name="mscoco",
        split_name="test",
        batch_size=12,
        collate_fn=data._collate_fn,
        shuffle=False,
        force_recompute=False,  # False,
        deletion_method="nms",
    )
    return preds_cal, preds_val


def filter_preds(preds, confidence_threshold=0.001):
    filters = [
        conf > confidence_threshold
        if (conf > confidence_threshold).any()
        else conf.argmin(0)[None]
        for conf in preds.confidences
    ]
    preds.pred_boxes = [pbs[f] for pbs, f in zip(preds.pred_boxes, filters)]
    preds.pred_cls = [pcs[f] for pcs, f in zip(preds.pred_cls, filters)]
    preds.confidences = [
        conf[f] for conf, f in zip(preds.confidences, filters)
    ]
    return preds


def compute_set_sizes(preds_cal, preds_val, conf_thr=0.5, matching_function="lac", calibration=True, temperature=1.0):
    
    if calibration:
        
        old_pred_cls_cal = preds_cal.pred_cls
        old_pred_cls_val = preds_val.pred_cls
        
        preds_cal.pred_cls = list([p.detach().clone() for p in preds_cal.pred_cls])
        preds_val.pred_cls = list([p.detach().clone() for p in preds_val.pred_cls])
        
        preds_cal.pred_cls = list([F.softmax(p/temperature, dim=-1)for p in preds_cal.pred_cls])
        preds_val.pred_cls = list([F.softmax(p/temperature, dim=-1)for p in preds_val.pred_cls])
    

    conf = ODConformalizer(
        multiple_testing_correction=None,
        #confidence_method="box_count_recall",  # "box_thresholded_distance",  # "nb_boxes",
        confidence_threshold=conf_thr,
        localization_method="pixelwise",
        localization_prediction_set="additive",
        classification_method="binary",
        classification_prediction_set="lac",
        backend="auto",
        optimizer="binary_search",
        matching_function=matching_function,
        device="cuda",
    )


    parameters = conf.calibrate(
        preds_cal,
        #alpha_confidence=0.03,
        alpha_localization=0.03,
        alpha_classification=0.07,
        verbose=False, 
    )

    conformal_preds = conf.conformalize(preds_val, parameters=parameters, verbose=False)

    results_val = conf.evaluate(
        preds_val,
        parameters=parameters,
        conformalized_predictions=conformal_preds,
        include_confidence_in_global=False,
        verbose=True
    ) 
    
    if calibration:
        preds_cal.pred_cls = old_pred_cls_cal
        preds_val.pred_cls = old_pred_cls_val

    results = {}

    key_loc = f"{matching_function}-{conf_thr}-{calibration}-{temperature}-loc"
    key_cls = f"{matching_function}-{conf_thr}-{calibration}-{temperature}-cls"
                
    cls_ss = results_val.classification_set_sizes
    loc_ss = results_val.localization_set_sizes
    results[key_cls] = (torch.mean(cls_ss.float()).item())
    results[key_loc] = (torch.mean(loc_ss.float()).item())

    return results

if __name__ == "__main__":
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = (
        "0"  # chose the GPU. If only one, then "0"
    )

    logging.getLogger().setLevel(logging.INFO)
    
    COCO_PATH = "/datasets/shared_datasets/coco/"

    data = MSCOCODataset(root=COCO_PATH, split="val")

    calibration_ratio = (
        0.5  # set 0.5 to use 50% for calibration and 50% for testing
    )

    use_smaller_subset = True  # TODO: Temp

    if use_smaller_subset:
        data_cal, data_val = data.split_dataset(
            calibration_ratio, shuffle=False, n_calib_test=800
        )
    else:
        data_cal, data_val = data.split_dataset(calibration_ratio, shuffle=False)
        
    model_detr50 = DETRModel(model_name="detr_resnet50", pretrained=True, device="cuda")
    #model_detr101 = DETRModel(model_name="detr_resnet101", pretrained=True, device="cuda")
    #model_yolov8x = YOLOModel(model_name="yolov8x.pt", pretrained=True, device="cuda")
    #models = [model_detr50, model_detr101, model_yolov8x]
    
    temps = np.logspace(-1.3, -1, 10)
    res = {}
    model = model_detr50
    preds_cal, preds_val = build_preds(model)
    preds_cal = filter_preds(preds_cal)
    preds_val = filter_preds(preds_val)
    for temp in temps:
        key = f"{model.model_name}-cal-{temp}"
        res[key] = compute_set_sizes(preds_cal, preds_val, conf_thr=0.5, matching_function="mix", calibration=True, temperature=temp)


    with open("results_calibration.pkl", "wb") as f:
        pickle.dump(res, f)