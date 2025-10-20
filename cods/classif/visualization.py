import matplotlib.pyplot as plt
import numpy as np

from cods.classif.data import ClassificationPredictions


def plot_predictions(
    idxs: list,
    preds: ClassificationPredictions,
    conf_cls: list | None = None,
):
    if isinstance(idxs, int):
        idxs = [idxs]
    if len(preds) != len(conf_cls):
        raise ValueError(
            f"len(preds)={len(preds)} and len(conf_cls)={len(conf_cls)} must be equal",
        )

    n = len(idxs)
    _, axs = plt.subplots(int(np.ceil(n / 4)), min(4, n), figsize=(12, 12))
    for i, idx_pred in enumerate(idxs):
        curr_ax = axs[i // 4, i % 4] if n > 4 else (axs[i] if n > 1 else axs)
        image_path = preds.image_paths[idx_pred]
        image = plt.imread(image_path)
        curr_ax.imshow(image)
        true_cls = preds.true_cls[idx_pred].item()
        true_cls_name = preds.idx_to_cls[true_cls]
        pred_cls = preds.pred_cls[idx_pred].argmax().item()
        pred_cls_name = preds.idx_to_cls[pred_cls]
        if conf_cls is None:
            curr_ax.set_title(
                f"True: {true_cls_name},\n Pred: {pred_cls_name}",
            )
        else:
            conf_cls_name = [
                preds.idx_to_cls[x.item()] for x in conf_cls[idx_pred]
            ]
            curr_ax.set_title(
                f"True: {true_cls_name},\n Pred: {pred_cls_name},\n Conf: {conf_cls_name if len(conf_cls_name) == 1 else 'len=' + str(len(conf_cls_name))}",
            )
            curr_ax.axis("off")
    plt.show()
