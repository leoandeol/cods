import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image

from cods.od.data import ODConformalizedPredictions, ODPredictions


def _plot_preds_no_overlap(
    idx,
    predictions: "ODPredictions",
    conformalized_predictions: "ODConformalizedPredictions" = None,
    confidence_threshold=None,
    idx_to_label: dict | None = None,
    save_as=None,
):
    """Plot predictions for one image while ensuring text labels (with background boxes)
    do NOT overlap with each other.

    - Greedy collision-avoidance: tries TL/TR/BL/BR/Above placements, then nudges.
    - Uses figure renderer + display coords to compute bbox intersections.

    Args:
        idx: index of the image to plot
        predictions: with image_paths, pred_boxes, true_boxes, true_cls, confidences, pred_cls, matching
        conformalized_predictions: (optional) with conf_boxes, conf_cls
        confidence_threshold: if None, falls back to predictions.confidence_threshold or 0.0
        idx_to_label: optional mapping int -> str
        save_as: optional path to save figure

    """
    is_conformal = conformalized_predictions is not None

    img_path = predictions.image_paths[idx]
    pred_boxes = predictions.pred_boxes[idx]
    true_boxes = predictions.true_boxes[idx]
    true_cls = predictions.true_cls[idx]
    conf = predictions.confidences[idx]
    cls_probas = predictions.pred_cls[idx]

    if is_conformal:
        conf_boxes = conformalized_predictions.conf_boxes[idx]
        conf_cls = conformalized_predictions.conf_cls[idx]

    # Confidence threshold handling (robust & permissive)
    if confidence_threshold is None:
        confidence_threshold = (
            predictions.confidence_threshold
            if getattr(predictions, "confidence_threshold", None) is not None
            else 0.0
        )

    keep = conf > confidence_threshold
    pred_boxes = pred_boxes[keep]
    cls_probas = cls_probas[keep]
    if is_conformal:
        conf_boxes = conf_boxes[keep]
        conf_cls = [c for k, c in zip(keep, conformalized_predictions.conf_cls[idx]) if k]

    # Load image
    image = Image.open(img_path).convert("RGB")
    image_width, image_height = image.size

    # Set up plot
    plt.figure(figsize=(14, 14))
    ax = plt.gca()
    ax.imshow(image)
    ax.set_axis_off()

    # ----- Helpers -----

    def clamp_box_to_image(x1, y1, x2, y2):
        x1 = max(0, min(image_width, x1))
        x2 = max(0, min(image_width, x2))
        y1 = max(0, min(image_height, y1))
        y2 = max(0, min(image_height, y2))
        return x1, y1, x2, y2

    # Keep a list of placed label bboxes in DISPLAY coords: [(x0,y0,x1,y1), ...]
    placed_label_bboxes_disp = []

    def bboxes_overlap(a, b):
        ax0, ay0, ax1, ay1 = a
        bx0, by0, bx1, by1 = b
        return not (ax1 <= bx0 or bx1 <= ax0 or ay1 <= by0 or by1 <= ay0)

    def fits_without_overlap(disp_bbox):
        # Check against previously placed label bboxes
        return all(not bboxes_overlap(disp_bbox, other) for other in placed_label_bboxes_disp)

    def place_text_with_avoid(ax, x, y, text, color, fontsize=15, pad=2):
        """Try several anchor positions with background box and avoid overlaps.
        Positions are relative to the target (x,y) near the box.
        """
        # Candidate placements: (ha, va, dx, dy)
        # Start near the box corner, then try alternatives.
        candidates = [
            ("left", "top", 0, 0),  # TL
            ("right", "top", 0, 0),  # TR
            ("left", "bottom", 0, 0),  # BL
            ("right", "bottom", 0, 0),  # BR
            ("center", "bottom", 0, -8),  # Above center, slight lift
            ("center", "top", 0, 8),  # Below center
        ]

        fig = ax.figure

        # We may need the renderer to compute extents
        fig.canvas.draw()  # ensure a renderer exists
        renderer = fig.canvas.get_renderer()

        # Try each candidate; if collides, nudge vertically until free or out of image
        for ha, va, dx, dy in candidates:
            tx = x + dx
            ty = y + dy

            # Create a Text, get bbox in display coords
            t = ax.text(
                tx,
                ty,
                text,
                fontsize=fontsize,
                ha=ha,
                va=va,
                bbox={"facecolor": color, "alpha": 0.55, "edgecolor": "none", "pad": pad},
                clip_on=True,
            )
            # Compute bbox in display space
            fig.canvas.draw()
            t.get_window_extent(renderer=renderer).transformed(
                fig.transFigure.inverted(),
            )
            # Convert to display pixels (easier intersection calc)
            bbox_pix = t.get_window_extent(renderer=renderer)
            disp_bbox = (bbox_pix.x0, bbox_pix.y0, bbox_pix.x1, bbox_pix.y1)

            # If overlap, try nudging vertically up to N times
            if not fits_without_overlap(disp_bbox):
                # Nudge loop
                max_tries = 20
                nudged = False
                for n in range(1, max_tries + 1):
                    # Alternate up/down nudges increasing in magnitude
                    delta = 10 * n
                    ty_try = ty - delta if n % 2 else ty + delta
                    t.set_position((tx, ty_try))
                    fig.canvas.draw()
                    bbox_pix2 = t.get_window_extent(renderer=renderer)
                    disp_bbox2 = (
                        bbox_pix2.x0,
                        bbox_pix2.y0,
                        bbox_pix2.x1,
                        bbox_pix2.y1,
                    )
                    # Check within image bounds (in data coords) roughly by asking if its anchor is inside
                    # (We keep clip_on=True as a safety.)
                    if fits_without_overlap(disp_bbox2):
                        placed_label_bboxes_disp.append(disp_bbox2)
                        nudged = True
                        break
                if not nudged:
                    # Couldn't find a spot; keep last position but register bbox to avoid further overlaps.
                    placed_label_bboxes_disp.append(disp_bbox)
                return t
            placed_label_bboxes_disp.append(disp_bbox)
            return t

        # Fallback (shouldn't happen): keep the last attempt
        placed_label_bboxes_disp.append(disp_bbox)
        return t

    def draw_rect(ax, box, color, proba, conformal=False):
        x1, y1, x2, y2 = box
        x1, y1, x2, y2 = clamp_box_to_image(x1, y1, x2, y2)

        ax.add_patch(
            plt.Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                fill=False,
                edgecolor=color,
                linewidth=2,
            ),
        )

        # Prepare label text
        if conformal:
            if isinstance(proba, torch.Tensor):
                proba = proba.detach().cpu().numpy()
            if hasattr(proba, "__len__") and len(proba) <= 13:
                if idx_to_label is not None:
                    txt = ", ".join(f"{idx_to_label[int(cl)]}" for cl in proba)
                else:
                    txt = ", ".join(f"{int(cl)}" for cl in proba)
            else:
                L = len(proba) if hasattr(proba, "__len__") else int(proba)
                txt = f"{L} labels"
            # Prefer top-left corner for conformal sets
            place_text_with_avoid(ax, x1, y1, txt, color)
            return

        # Non-conformal (single class id or prob dist)
        if isinstance(proba, (int, np.integer)) or (
            isinstance(proba, torch.Tensor) and proba.ndim == 0
        ):
            if isinstance(proba, torch.Tensor):
                proba = int(proba.item())
            txt = (
                f"{idx_to_label[proba]}"
                if idx_to_label and proba >= 0
                else (f"{proba}" if proba >= 0 else "conf")
            )
            # Prefer bottom-right corner
            place_text_with_avoid(ax, x2, y2, txt, color)
        else:
            if isinstance(proba, torch.Tensor):
                proba = proba.detach().cpu().numpy()
            cl = int(np.argmax(proba))
            score = float(proba[cl])
            txt = f"{idx_to_label[cl]}: {score:0.2f}" if idx_to_label else f"{cl}: {score:0.2f}"
            # Prefer top-left
            place_text_with_avoid(ax, x1, y1, txt, color)

    # ----- Draw -----

    # Ground-truth in green
    for box, cl in zip(true_boxes, true_cls):
        box = np.array(box, dtype=float)
        draw_rect(ax, box, "green", cl)

    # Predictions in red
    for box, prob in zip(pred_boxes, cls_probas):
        if isinstance(box, torch.Tensor):
            box = box.detach().cpu().numpy()
        draw_rect(ax, box.astype(float), "red", prob)

    # Conformal sets in purple
    if is_conformal:
        for box, conf_cls_i in zip(conf_boxes, conf_cls):
            if isinstance(box, torch.Tensor):
                box = box.detach().cpu().numpy()
            draw_rect(ax, box.astype(float), "purple", conf_cls_i, conformal=True)

    # Arrows from GT to matched prediction (if available)
    if getattr(predictions, "matching", None) is not None:
        for i, true_box in enumerate(true_boxes):
            match_idx = predictions.matching[idx][i][0]
            if match_idx is None or match_idx >= len(pred_boxes):
                continue
            matching_pred_box = pred_boxes[match_idx]
            if isinstance(matching_pred_box, torch.Tensor):
                matching_pred_box = matching_pred_box.detach().cpu().numpy()
            ax.annotate(
                "",
                xy=(true_box[0], true_box[1]),
                xytext=(matching_pred_box[0], matching_pred_box[1]),
                arrowprops={"arrowstyle": "->", "lw": 2, "color": "blue"},
            )

    plt.tight_layout(pad=0)
    if save_as is not None:
        plt.savefig(save_as, bbox_inches="tight", dpi=180)
    plt.show()


def _old_plot_preds(
    idx,
    predictions: ODPredictions,
    conformalized_predictions: ODConformalizedPredictions = None,
    confidence_threshold=None,
    idx_to_label: dict | None = None,
    save_as=None,
):
    """Plot the predictions of an object detection model.

    Args:
    ----
        preds (object): Object containing the predictions.
        idx (int): Index of the image to plot.
        conf_boxes (list): List of confidence boxes.
        conf_cls (list): List of confidence classes.
        confidence_threshold (float, optional): Confidence threshold for filtering predictions. If not provided, the threshold from `preds` will be used. Defaults to None.
        save_as (str, optional): File path to save the plot. Defaults to None.

    """
    is_conformal = conformalized_predictions is not None

    img_path = predictions.image_paths[idx]
    pred_boxes = predictions.pred_boxes[idx]
    true_boxes = predictions.true_boxes[idx]
    true_cls = predictions.true_cls[idx]
    conf = predictions.confidences[idx]
    cls_probas = predictions.pred_cls[idx]

    if is_conformal:
        conf_boxes = conformalized_predictions.conf_boxes[idx]
        conf_cls = conformalized_predictions.conf_cls[idx]

    if confidence_threshold is None and predictions.confidence_threshold is not None:
        confidence_threshold = predictions.confidence_threshold
        print("Using confidence threshold from preds")
    else:
        raise ValueError("Confidence Threshold should be provided")

    keep = conf > confidence_threshold
    pred_boxes = pred_boxes[keep]
    if is_conformal:
        conf_boxes = conf_boxes[keep]
    cls_probas = cls_probas[keep]

    image = Image.open(img_path)
    image_width, image_height = image.size
    image.save("./test.png")
    plt.figure(figsize=(14, 14))
    plt.imshow(image)

    def draw_rect(ax, box, color, proba, conformal=False):
        """Draw a rectangle on the plot.

        Args:
        ----
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
            ),
        )
        # TODO(leo):conf
        if conformal:
            if len(proba) <= 13:
                # Print up to the three labels of the prediction sets
                if isinstance(proba, torch.Tensor):
                    proba = proba.cpu().numpy()
                if idx_to_label is not None:
                    text = ", ".join([f"{idx_to_label[cl]}" for cl in proba])
                else:
                    text = ", ".join([f"{cl}" for cl in proba])
                ax.text(
                    x1,
                    y1,
                    text,
                    fontsize=15,
                    bbox={"facecolor": color, "alpha": 0.5},
                )
            else:
                # Print nb of labels
                text = f"{len(proba)} labels"
                ax.text(
                    x1,
                    y1,
                    text,
                    fontsize=15,
                    bbox={"facecolor": color, "alpha": 0.5},
                )
        elif isinstance(proba, int) or len(proba.shape) == 0:
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
                bbox={"facecolor": color, "alpha": 0.5},
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
                bbox={"facecolor": color, "alpha": 0.5},
            )

    ax = plt.gca()
    for box, cl in zip(true_boxes, true_cls):
        draw_rect(ax, box, "green", cl)

    for box, prob in zip(pred_boxes, cls_probas):
        box = box.detach().cpu().numpy()
        draw_rect(ax, box, "red", prob)

    if is_conformal:
        for box, conf_cls_i in zip(conf_boxes, conf_cls):
            box = box.detach().cpu().numpy()
            draw_rect(ax, box, "purple", conf_cls_i, conformal=True)

    # Draw an arrow between the top-left corners of true and matching predicted box
    for i, true_box in enumerate(true_boxes):
        matching_pred_box = pred_boxes[predictions.matching[idx][i][0]]
        ax.annotate(
            "",
            xy=(true_box[0], true_box[1]),
            xytext=(matching_pred_box[0], matching_pred_box[1]),
            arrowprops={"arrowstyle": "->", "lw": 2, "color": "blue"},
        )

    plt.axis("off")
    if save_as is not None:
        plt.savefig(save_as, bbox_inches="tight")
    plt.show()


def create_pdf_with_plots(
    predictions: ODPredictions,
    conformalized_predictions: ODConformalizedPredictions = None,
    confidence_threshold=None,
    idx_to_label: dict | None = None,
    output_pdf="output.pdf",
):
    """Create a PDF with plots for each image in the predictions.

    Args:
    ----
        predictions (ODPredictions): Object containing the predictions.
        conformalized_predictions (ODConformalizedPredictions, optional): Object containing conformalized predictions. Defaults to None.
        confidence_threshold (float, optional): Confidence threshold for filtering predictions. Defaults to None.
        idx_to_label (dict, optional): Mapping from class indices to labels. Defaults to None.
        output_pdf (str, optional): Path to save the output PDF. Defaults to "output.pdf".

    """
    is_conformal = conformalized_predictions is not None

    if confidence_threshold is None and predictions.confidence_threshold is not None:
        confidence_threshold = predictions.confidence_threshold
    elif confidence_threshold is None:
        raise ValueError("Confidence Threshold should be provided")

    with PdfPages(output_pdf) as pdf:
        for idx in range(len(predictions.image_paths)):
            img_path = predictions.image_paths[idx]
            pred_boxes = predictions.pred_boxes[idx]
            true_boxes = predictions.true_boxes[idx]
            true_cls = predictions.true_cls[idx]
            conf = predictions.confidences[idx]
            cls_probas = predictions.pred_cls[idx]

            if is_conformal:
                conf_boxes = conformalized_predictions.conf_boxes[idx]
                conf_cls = conformalized_predictions.conf_cls[idx]

            keep = conf > confidence_threshold
            pred_boxes = pred_boxes[keep]
            if is_conformal:
                conf_boxes = conf_boxes[keep]
            cls_probas = cls_probas[keep]

            image = Image.open(img_path)
            image_width, image_height = image.size

            plt.figure(figsize=(14, 14))
            plt.imshow(image)

            def draw_rect(
                ax,
                box,
                color,
                proba,
                conformal=False,
                image_width=image_width,
                image_height=image_height,
            ):
                x1, y1, x2, y2 = box
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
                    ),
                )
                if conformal:
                    if len(proba) <= 5:
                        if isinstance(proba, torch.Tensor):
                            proba = proba.cpu().numpy()
                        if idx_to_label is not None:
                            text = ", ".join(
                                [f"{idx_to_label[cl]}" for cl in proba],
                            )
                        else:
                            text = ", ".join([f"{cl}" for cl in proba])
                        ax.text(
                            x1,
                            y1,
                            text,
                            fontsize=15,
                            bbox={"facecolor": color, "alpha": 0.5},
                        )
                    else:
                        text = f"{len(proba)} labels"
                        ax.text(
                            x1,
                            y1,
                            text,
                            fontsize=15,
                            bbox={"facecolor": color, "alpha": 0.5},
                        )
                elif isinstance(proba, int) or len(proba.shape) == 0:
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
                        bbox={"facecolor": color, "alpha": 0.5},
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
                        bbox={"facecolor": color, "alpha": 0.5},
                    )

            ax = plt.gca()
            for box, cl in zip(true_boxes, true_cls):
                draw_rect(ax, box, "green", cl)

            for box, prob in zip(pred_boxes, cls_probas):
                box = box.detach().cpu().numpy()
                draw_rect(ax, box, "red", prob)

            if is_conformal:
                for box, conf_cls_i in zip(conf_boxes, conf_cls):
                    box = box.detach().cpu().numpy()
                    draw_rect(ax, box, "purple", conf_cls_i, conformal=True)

            for i, true_box in enumerate(true_boxes):
                try:
                    matched_idx = predictions.matching[idx][i]
                    if len(matched_idx) == 0:
                        continue
                    matching_pred_box = pred_boxes[matched_idx[0]]
                    ax.annotate(
                        "",
                        xy=(true_box[0], true_box[1]),
                        xytext=(matching_pred_box[0], matching_pred_box[1]),
                        arrowprops={"arrowstyle": "->", "lw": 2, "color": "blue"},
                    )
                except Exception as e:
                    print(e)

            plt.axis("off")
            plt.title(
                f"Image {idx + 1}: {len(true_boxes)} ground truths, {len(pred_boxes)} predictions",
            )
            pdf.savefig(bbox_inches="tight")
            plt.close()


def plot_histograms_predictions(predictions: ODPredictions):
    # Plot three histograms in the same figure, with 20 bins that use lengths3 = [len(x) for x in preds_val.true_boxes] but the same also for confidence and confidence thresholded, and pu the mean length as title of the figure,

    list_true = [len(x) for x in predictions.true_boxes]
    list_pred = [len(x) for x in predictions.confidences]
    list_pred_thresh = [sum(x > predictions.confidence_threshold) for x in predictions.confidence]

    _, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].hist(list_true, bins=20)
    axs[0].set_title(f"True boxes, mean size = {np.mean(list_true):.2f}")
    axs[1].hist(list_pred, bins=20)
    axs[1].set_title(f"Predicted boxes, mean size = {np.mean(list_pred):.2f}")
    axs[2].hist(list_pred_thresh, bins=20)
    axs[2].set_title(
        f"Predicted boxes above threshold, mean size = {np.mean(list_pred_thresh):.2f}",
    )
    plt.show()
