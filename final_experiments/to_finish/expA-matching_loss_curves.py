import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from cods.od.data import MSCOCODataset
from cods.od.loss import (
    ClassificationLossWrapper,
    ODBinaryClassificationLoss,
    PixelWiseRecallLoss,
)
from cods.od.models import DETRModel, YOLOModel
from cods.od.utils import (
    assymetric_hausdorff_distance_old,
    generalized_iou,
    match_predictions_to_true_boxes,
)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = (
    "1"  # chose the GPU. If only one, then "0"
)

logging.getLogger().setLevel(logging.INFO)

# set [COCO_PATH] to the directory to your local copy of the COCO dataset
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

print(f"{len(data) = }")
print(f"{len(data_cal) = }")
print(f"{len(data_val) = }")


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
        filter_preds_by_confidence=1e-3,
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
        filter_preds_by_confidence=1e-3,
    )
    return preds_cal, preds_val


def compute_curves(preds_cal):
    # Measure, for several distance function, the average distance in the match
    loss_cls = ClassificationLossWrapper(ODBinaryClassificationLoss())
    loss_loc = PixelWiseRecallLoss()
    results = {}
    thresholds = np.linspace(0, 0.95, 50)
    FUNCTIONS = {
        "giou": generalized_iou,
        "hausdorff": assymetric_hausdorff_distance_old,
        "lac": lambda x, y: 1,
        "mix": assymetric_hausdorff_distance_old,
    }
    for matching_function in [
        "lac",
        "hausdorff",
        "mix",
    ]:  # ["hausdorff", "giou"]:
        did_first_k = False
        for k_top in [
            1,
            2,
            5,
            10,
            20,
            40,
        ]:  # for class_factor in [0]:#[0, 1, 5, 25]:
            class_factor = k_top
            key = f"{matching_function}-{class_factor}"  # {'class_factor' if class_factor else ''}"
            key_loc = f"{key}-loc"
            key_cls = f"{key}-cls"
            key_cls_rank = f"{key}-rankmean"
            key_cls_rank95 = f"{key}-rank95"
            key_cls_rank97 = f"{key}-rank97"
            results[key] = []
            results[key_loc] = []
            results[key_cls] = []
            if not did_first_k:
                results[key_cls_rank] = []
                results[key_cls_rank95] = []
                results[key_cls_rank97] = []
            curr_func = FUNCTIONS[matching_function]
            for conf_thr in thresholds:
                match_predictions_to_true_boxes(
                    preds_cal,
                    distance_function=matching_function,
                    overload_confidence_threshold=conf_thr,
                    # class_factor=class_factor,
                )
                distances = []
                dist_cls = []
                dist_loc = []
                rank_cls = []
                rank95_cls = []
                rank97_cls = []
                for i in range(len(preds_cal)):
                    true_boxs = preds_cal.true_boxes[i]
                    true_clss = preds_cal.true_cls[i]
                    pred_boxs = preds_cal.pred_boxes[i]
                    pred_clss = preds_cal.pred_cls[i]
                    matching = preds_cal.matching[i]
                    if len(matching) == 0 or len(matching[0]) == 0:
                        continue
                    try:
                        pred_boxs = torch.stack(
                            [pred_boxs[match_idx[0]] for match_idx in matching]
                        )  # if (len(preds_cal.matching[i]) > 0) or (np.array(list([len(x)>0 for x in preds_cal.matching[i]]))).sum() else []
                    except:
                        print(matching)
                        break
                    pred_clss = torch.stack(
                        [pred_clss[match_idx[0]] for match_idx in matching]
                    )
                    conf_boxs = pred_boxs
                    conf_clss = list(
                        [
                            torch.topk(pcls, k_top, dim=0)[1]
                            for pcls in pred_clss
                        ]
                    )
                    if not did_first_k:
                        sorted_indices = torch.argsort(
                            pred_clss, descending=True, dim=1
                        )
                        ranks = (
                            sorted_indices == true_clss.unsqueeze(1)
                        ).nonzero(as_tuple=True)[1]
                        # Correction because rank starts at 1 not 0
                        ranks += 1
                        # print(ranks)
                        # print(ranks.float())
                        # break
                        # rank_cls_val = torch.mean(ranks.float()).item()
                        rank_cls_val = (
                            ranks.float().mean().item()
                        )  # .detach().cpu().numpy() # ADDED MEAN
                        rank_cls.append(rank_cls_val)
                    loss_cls_val = loss_cls(
                        true_boxs, true_clss, conf_boxs, conf_clss
                    )
                    loss_loc_val = loss_loc(
                        true_boxs, true_clss, conf_boxs, conf_clss
                    )
                    dist_cls.append(loss_cls_val)
                    dist_loc.append(loss_loc_val)

                    for j in range(len(true_boxs)):
                        if len(preds_cal.matching[i][j]) == 0:
                            continue
                        match_idx = preds_cal.matching[i][j][0]
                        # try:
                        #     match_idx = preds_cal.matching[i][j][0]
                        # except:
                        #     print(preds_cal.matching[i])
                        dist = curr_func(
                            preds_cal.true_boxes[i][j],
                            preds_cal.pred_boxes[i][match_idx],
                        )

                        # Compute classification and localization loss

                        distances.append(dist)
                results[key].append(np.mean(distances))
                results[key_cls].append(np.mean(dist_cls))
                results[key_loc].append(np.mean(dist_loc))
                if not did_first_k:
                    # rank_cls = np.concatenate(rank_cls)
                    results[key_cls_rank].append(np.mean(rank_cls))
                    results[key_cls_rank95].append(np.percentile(rank_cls, 95))
                    results[key_cls_rank97].append(np.percentile(rank_cls, 97))
            did_first_k = True
    results["thresholds"] = thresholds
    return results


def build_plots(models):
    for model in models:
        preds_cal, preds_val = build_preds(model)
        results = compute_curves(preds_cal)

        # four subplots
        fig, axs = plt.subplots(1, 4, figsize=(24, 6))

        thresholds = results["thresholds"]

        # Ax 1 : Distance
        for k, v in results.items():
            if "giou" in k:
                v = 1 - np.array(v)
            if len(k.split("-")) == 2:
                v = np.array(v)
                v /= v.max()
                axs[0].plot(thresholds, v, label=k)
            axs[0].legend()
            axs[0].set_title("Distance")
            axs[0].set_xlabel("Confidence threshold")
            axs[0].set_xticks(np.arange(0, 1, 0.05))
            axs[0].set_ylabel(
                "Normalized distance" if "giou" in k else "Distance"
            )

        # Ax 2 : Localization loss
        for k, v in results.items():
            if "loc" in k:
                v = np.array(v)
                # v /= v.max()
                axs[1].plot(thresholds, v, label=k)
            axs[1].legend()
            axs[1].set_title("Localization loss")
            axs[1].set_xlabel("Confidence threshold")
            axs[1].set_xticks(np.arange(0, 1, 0.05))
            axs[1].set_ylabel("Loss")

        # Ax 3 : Classification loss
        for k, v in results.items():
            if "cls" in k:
                v = np.array(v)
                # v /= v.max()
                axs[2].plot(thresholds, v, label=k)
            axs[2].legend()
            axs[2].set_title("Classification loss")
            axs[2].set_xlabel("Confidence threshold")
            axs[2].set_xticks(np.arange(0, 1, 0.05))
            # axs[2].set_yticks(np.arange(0,1,0.05))
            axs[2].set_ylabel("Loss")

        # Ax 4 : Classification rank
        for k, v in results.items():
            # print(k, len(v))
            # print(v)
            if "rank" in k:
                v = np.array(v)
                # v /= v.max()
                axs[3].plot(thresholds, v, label=k)
            axs[3].legend()
            axs[3].set_title("Classification rank")
            axs[3].set_xlabel("Confidence threshold")
            axs[3].set_xticks(np.arange(0, 1, 0.05))
            axs[3].set_ylabel("Rank")

        # plt.title(f"Model {type(model)}")
        fig.suptitle(f"Model {model.model_name}")
        plt.savefig(f"figs/plot_curves_{model.model_name}.png")
        plt.show()


if __name__ == "__main__":
    model_detr50 = DETRModel(
        model_name="detr_resnet50", pretrained=True, device="cpu"
    )
    model_detr101 = DETRModel(
        model_name="detr_resnet101", pretrained=True, device="cpu"
    )
    model_yolov8x = YOLOModel(
        model_name="yolov8x.pt", pretrained=True, device="cpu"
    )

    build_plots([model_detr50, model_detr101, model_yolov8x])
