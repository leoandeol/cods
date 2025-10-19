from hashlib import sha256
from typing import Optional

import torch
import torchvision
import tqdm

from cods.base.models import Model
from cods.od.data import ODPredictions
from cods.od.models.utils import bayesod, filter_preds


class ODModel(Model):
    def __init__(
        self,
        model_name: str,
        save_dir_path: str,
        pretrained: bool = True,
        weights: Optional[str] = None,
        device: str = "cpu",
    ):
        """Initializes an instance of the ODModel class.

        Args:
        ----
            model_name (str): The name of the model.
            save_dir_path (str): The path to save the model.
            pretrained (bool, optional): Whether to use pretrained weights. Defaults to True.
            weights (str, optional): The path to the weights file. Defaults to None.
            device (str, optional): The device to use for computation. Defaults to "cpu".

        """
        super().__init__(
            model_name=model_name,
            save_dir_path=save_dir_path,
            pretrained=pretrained,
            weights=weights,
            device=device,
        )

    def build_predictions(
        self,
        dataset,
        dataset_name: str,
        split_name: str,
        batch_size: int,
        shuffle: bool = False,
        verbose: bool = True,
        force_recompute: bool = False,
        deletion_method: str = "nms",
        iou_threshold: float = 0.5,
        filter_preds_by_confidence: Optional[float] = None,
        **kwargs,
    ) -> ODPredictions:
        """Builds predictions for the given dataset.

        Args:
        ----
            dataset: The dataset to build predictions for.
            dataset_name (str): The name of the dataset.
            split_name (str): The name of the split.
            batch_size (int): The batch size for prediction.
            shuffle (bool, optional): Whether to shuffle the dataset. Defaults to False.
            verbose (bool, optional): Prints progress. Defaults to True.
            **kwargs: Additional keyword arguments for the DataLoader.
            #TODO(leo): not up to date

        Returns:
        -------
            ODPredictions: Predictions object to use for prediction set construction.

        """
        string_to_hash = f"{dataset.root}_{dataset_name}_{split_name}_{batch_size}_{shuffle}_{self.model_name}_object_detection_{dataset.image_ids}"
        hash = sha256(string_to_hash.encode()).hexdigest()

        preds = None
        if not force_recompute:
            preds = self._load_preds_if_exists(
                hash,
                # dataset_name=dataset_name,
                # split_name=split_name,
                # task_name="object_detection",
            )
            if preds is not None:
                if verbose:
                    print("Predictions already exist, loading them...")
                if isinstance(preds, ODPredictions):
                    # Make sure the predictions are on the right device
                    if filter_preds_by_confidence is not None:
                        preds = filter_preds(
                            preds,
                            confidence_threshold=filter_preds_by_confidence,
                        )
                    preds.to(self.device)
                    return preds
                raise ValueError(
                    "Predictions file exists but is not of type ODPredictions",
                )
            if verbose:
                print("Predictions do not exist, building them...")
        elif verbose:
            print(
                "Force recompute is set to True, building predictions...",
            )

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            **kwargs,
        )

        # TODO: dangerous for YOLO
        # self.model.eval()

        pbar = enumerate(tqdm.tqdm(dataloader, disable=not verbose))

        all_image_paths = []
        all_image_shapes = []
        all_true_boxes = []
        all_pred_boxes = []
        all_pred_boxes_unc = []
        all_confidences = []
        all_true_cls = []
        all_pred_cls = []
        with torch.no_grad():
            for _, batch in pbar:
                res = self.predict_batch(batch)

                image_paths = res["image_paths"]
                image_shapes = res["image_shapes"]
                true_boxes = res["true_boxes"]
                pred_boxes = res["pred_boxes"]
                confidences = res["confidences"]
                true_cls = res["true_cls"]
                pred_cls = res["pred_cls"]

                pred_boxes, pred_cls, confidences, pred_boxes_unc = self._filter_preds(
                    pred_boxes,
                    pred_cls,
                    confidences,
                    iou_threshold=iou_threshold,
                    method=deletion_method,
                )

                all_image_paths.append(image_paths)
                all_image_shapes.append(image_shapes)
                all_true_boxes.append(true_boxes)
                all_pred_boxes.append(pred_boxes)
                if pred_boxes_unc is not None:
                    all_pred_boxes_unc.append(pred_boxes_unc)
                all_confidences.append(confidences)
                all_true_cls.append(true_cls)
                all_pred_cls.append(pred_cls)

        all_image_paths = ([path for arr_path in all_image_paths for path in arr_path],)

        all_image_shapes = ([shape for arr_shape in all_image_shapes for shape in arr_shape],)

        all_true_boxes = ([box.to(self.device) for arr_box in all_true_boxes for box in arr_box],)

        all_pred_boxes = ([box for arr_box in all_pred_boxes for box in arr_box],)

        if len(all_pred_boxes_unc) > 0:
            all_pred_boxes_unc = (
                [box_unc for arr_box_unc in all_pred_boxes_unc for box_unc in arr_box_unc],
            )
        else:
            all_pred_boxes_unc = None
        all_confidences = (
            [confidence for arr_confidence in all_confidences for confidence in arr_confidence],
        )

        all_true_cls = ([cls.to(self.device) for arr_cls in all_true_cls for cls in arr_cls],)

        all_pred_cls = ([proba for arr_proba in all_pred_cls for proba in arr_proba],)

        preds = ODPredictions(
            dataset_name=dataset_name,
            split_name=split_name,
            image_paths=all_image_paths,
            image_shapes=all_image_shapes,
            true_boxes=all_true_boxes,
            pred_boxes=all_pred_boxes,
            confidences=all_confidences,
            true_cls=all_true_cls,
            pred_cls=all_pred_cls,
            names=dataset.NAMES,
            pred_boxes_uncertainty=all_pred_boxes_unc,
        )
        self._save_preds(preds, hash)

        # Done after saving : we always save and therefore load all predictions without filtering
        if filter_preds_by_confidence is not None:
            preds = filter_preds(
                preds,
                confidence_threshold=filter_preds_by_confidence,
            )

        return preds

    def _filter_preds(
        self,
        pred_boxes,
        pred_cls,
        confidences,
        iou_threshold=0.5,
        method: str = "nms",
    ):
        """Filters the predicted bounding boxes based on the confidence scores and IoU threshold.

        Args:
        ----
            pred_boxes: The predicted bounding boxes.
            pred_cls: The predicted class labels.
            confidences: The confidence scores.
            iou_threshold (float, optional): The IoU threshold for filtering. Defaults to 0.8.
            method (str): the method use to delete redundant boxes, currently supported NMS and BayesOD

        Returns:
        -------
            Tuple: The filtered predicted bounding boxes, predicted class labels, confidence scores and uncertainty if existing.

        """
        new_pred_boxes = []
        new_pred_cls = []
        new_confidences = []
        new_pred_boxes_unc = []
        for i in range(len(pred_boxes)):
            if method.lower() == "nms":
                keep = torchvision.ops.nms(
                    pred_boxes[i],
                    confidences[i],
                    iou_threshold=iou_threshold,
                )
                new_pred_box = pred_boxes[i].index_select(dim=0, index=keep)
                new_pred_cl = pred_cls[i].index_select(dim=0, index=keep)
                new_confidence = confidences[i].index_select(dim=0, index=keep)
                new_pred_boxes.append(new_pred_box)
                new_pred_cls.append(new_pred_cl)
                new_confidences.append(new_confidence)
            elif method.lower() == "bayesod":
                new_pred_boxes, new_confidences, new_pred_cls = bayesod(
                    pred_boxes[i],
                    confidences[i],
                    pred_cls[i],
                    iou_threshold=iou_threshold,
                )
            else:
                raise NotImplementedError(
                    "Not Implemented, method must be one of 'nms', 'bayesod'",
                )

        if len(new_pred_boxes_unc) == 0:
            new_pred_boxes_unc = None

        return (
            new_pred_boxes,
            new_pred_cls,
            new_confidences,
            new_pred_boxes_unc,
        )

    def predict_batch(self, batch: list, **kwargs) -> dict:
        """Predicts the output given a batch of input tensors.

        Args:
        ----
            batch (list): The input batch.

        Returns:
        -------
            dict: The predicted output as a dictionary with the following keys:
                - "image_paths" (list): The paths of the input images
                - "true_boxes" (list): The true bounding boxes of the objects in the images
                - "pred_boxes" (list): The predicted bounding boxes of the objects in the images
                - "confidences" (list): The confidence scores of the predicted bounding boxes
                - "true_cls" (list): The true class labels of the objects in the images
                - "pred_cls" (list): The predicted class labels of the objects in the images

        """
        raise NotImplementedError("Please Implement this method")
