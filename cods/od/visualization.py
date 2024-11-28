import matplotlib.pyplot as plt
import torch
from PIL import Image

from cods.od.data import ODConformalizedPredictions, ODPredictions


def plot_preds(
    predictions: ODPredictions,
    conformalized_predictions: ODConformalizedPredictions,
    idx,
    confidence_threshold=None,
    idx_to_label: dict = None,
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
    img_path = predictions.image_paths[idx]
    pred_boxes = predictions.pred_boxes[idx]
    true_boxes = predictions.true_boxes[idx]
    true_cls = predictions.true_cls[idx]
    conf = predictions.confidence[idx]
    cls_probas = predictions.pred_cls[idx]

    conf_boxes = conformalized_predictions.conf_boxes[idx]
    conf_cls = conformalized_predictions.conf_cls[idx]

    if (
        confidence_threshold is None
        and predictions.confidence_threshold is not None
    ):
        confidence_threshold = predictions.confidence_threshold
        print("Using confidence threshold from preds")
    else:
        raise ValueError("Confidence Threshold should be provided")

    keep = conf > confidence_threshold
    pred_boxes = pred_boxes[keep]
    conf_boxes = conf_boxes[keep]
    cls_probas = cls_probas[keep]

    image = Image.open(img_path)
    image_width, image_height = image.size
    image.save("./test.png")
    plt.figure(figsize=(14, 14))
    plt.imshow(image)

    def draw_rect(ax, box, color, proba, conf=False):
        """
        Draw a rectangle on the plot.

        Args:
            ax (object): Axes object of the plot.
            box (list): List of coordinates [x1, y1, x2, y2] of the rectangle.
            color (str): Color of the rectangle.
            proba (int or numpy.ndarray): Probability or probability distribution of the class.
        """
        x1, y1, x2, y2 = box
        # correct coordinates to not go outside of bounds
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(image_width, x2)
        y2 = min(image_height, y2)

        ax.add_patch(
            plt.Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                fill=False,
                edgecolor=color,
                linewidth=2,
            )
        )
        if conf:
            if len(proba) <= 3:
                # Print up to the three labels of the prediction sets
                if idx_to_label is not None:
                    text = ", ".join([f"{idx_to_label[cl]}" for cl in proba])
                else:
                    text = ", ".join([f"{cl}" for cl in proba])
                ax.text(
                    x1,
                    y1,
                    text,
                    fontsize=15,
                    bbox=dict(facecolor=color, alpha=0.5),
                )
            else:
                # Print nb of labels
                text = f"{len(proba)} labels"
                ax.text(
                    x1,
                    y1,
                    text,
                    fontsize=15,
                    bbox=dict(facecolor=color, alpha=0.5),
                )
        else:
            if isinstance(proba, int) or len(proba.shape) == 0:
                if isinstance(proba, torch.Tensor):
                    proba = proba.item()
                if idx_to_label is not None:
                    text = f"{idx_to_label[proba]}" if proba >= 0 else "conf"
                else:
                    text = f"{proba}" if proba >= 0 else "conf"
                ax.text(
                    x2 - 30,
                    y2,
                    text,
                    fontsize=15,
                    bbox=dict(facecolor=color, alpha=0.5),
                )
            else:
                cl = proba.argmax().item()
                if idx_to_label is not None:
                    text = f"{idx_to_label[cl]}: {proba[cl]:0.2f}"
                else:
                    text = f"{cl}: {proba[cl]:0.2f}"

                ax.text(
                    x1,
                    y1,
                    text,
                    fontsize=15,
                    bbox=dict(facecolor=color, alpha=0.5),
                )

    ax = plt.gca()
    for box, cl in zip(true_boxes, true_cls):
        draw_rect(ax, box, "green", cl)

    for box, prob in zip(pred_boxes, cls_probas):
        box = box.detach().cpu().numpy()
        draw_rect(ax, box, "red", prob)

    for box, conf_cls_i in zip(conf_boxes, conf_cls):
        box = box.detach().cpu().numpy()
        draw_rect(ax, box, "purple", conf_cls_i, conf=True)

    plt.axis("off")
    if save_as is not None:
        plt.savefig(save_as, bbox_inches="tight")
    plt.show()
