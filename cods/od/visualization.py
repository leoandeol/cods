import matplotlib.pyplot as plt
from PIL import Image

from cods.od.models.detr import DETRModel


def plot_preds(
    preds,
    idx,
    conf_boxes: list,
    conf_cls: list,
    confidence_threshold=None,
    save_as=None,
):
    """
    Plot the predictions of an object detection model.

    Args:
        preds (object): Object containing the predictions.
        idx (int): Index of the image to plot.
        conf_boxes (list): List of confidence boxes.
        conf_cls (list): List of confidence classes.
        confidence_threshold (float, optional): Confidence threshold for filtering predictions. If not provided, the threshold from `preds` will be used. Defaults to None.
        save_as (str, optional): File path to save the plot. Defaults to None.
    """
    img_path = preds.image_paths[idx]
    pred_boxes = preds.pred_boxes[idx]
    true_boxes = preds.true_boxes[idx]
    conf_boxes = conf_boxes[idx]
    conf_cls = conf_cls[idx]
    true_cls = preds.true_cls[idx]
    conf = preds.confidence[idx]
    cls_probas = preds.pred_cls[idx]

    if confidence_threshold is None and preds.confidence_threshold is not None:
        confidence_threshold = preds.confidence_threshold
        print("Using confidence threshold from preds")
    else:
        raise ValueError("Confidence Threshold should be provided")

    keep = conf > confidence_threshold
    pred_boxes = pred_boxes[keep]
    conf_boxes = conf_boxes[keep]
    conf_cls = conf_cls[keep]
    cls_probas = cls_probas[keep]

    image = Image.open(img_path)
    plt.figure(figsize=(14, 14))
    plt.imshow(image)

    def draw_rect(ax, box, color, proba):
        """
        Draw a rectangle on the plot.

        Args:
            ax (object): Axes object of the plot.
            box (list): List of coordinates [x1, y1, x2, y2] of the rectangle.
            color (str): Color of the rectangle.
            proba (int or numpy.ndarray): Probability or probability distribution of the class.
        """
        x1, y1, x2, y2 = box
        ax.add_patch(
            plt.Rectangle(
                (x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor=color, linewidth=2
            )
        )
        if isinstance(proba, int) or len(proba.shape) == 0:
            text = f"{DETRModel.COCO_CLASSES[proba]}" if proba >= 0 else "conf"
            ax.text(
                x2 - 30, y2, text, fontsize=15, bbox=dict(facecolor=color, alpha=0.5)
            )
        else:
            cl = proba.argmax()
            text = f"{DETRModel.COCO_CLASSES[cl]}: {proba[cl]:0.2f}"

            ax.text(x1, y1, text, fontsize=15, bbox=dict(facecolor=color, alpha=0.5))

    ax = plt.gca()
    for box, cl in zip(true_boxes, true_cls):
        draw_rect(ax, box, "green", cl)

    for box, prob in zip(pred_boxes, cls_probas):
        box = box.detach().cpu().numpy()
        draw_rect(ax, box, "red", prob)

    for box in conf_boxes:
        box = box.detach().cpu().numpy()
        draw_rect(ax, box, "purple", -1)

    plt.axis("off")
    if save_as is not None:
        plt.savefig(save_as, bbox_inches="tight")
    plt.show()
