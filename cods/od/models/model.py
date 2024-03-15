from typing import Optional

import torch
import torch.utils.data
import torchvision
import tqdm

from cods.base.models import Model
from cods.od.data import ODPredictions


class ODModel(Model):
    def __init__(
        self,
        model_name: str,
        save_dir_path: str,
        pretrained: bool = True,
        weights: Optional[str] = None,
        device: str = "cuda",
    ):
        """
        Initializes an instance of the ODModel class.

        Args:
            model_name (str): The name of the model.
            save_dir_path (str): The path to save the model.
            pretrained (bool, optional): Whether to use pretrained weights. Defaults to True.
            weights (str, optional): The path to the weights file. Defaults to None.
            device (str, optional): The device to use for computation. Defaults to "cuda".
        """
        super(ODModel, self).__init__(
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
        **kwargs,
    ) -> ODPredictions:
        """
        Builds predictions for the given dataset.

        Args:
            dataset: The dataset to build predictions for.
            dataset_name (str): The name of the dataset.
            split_name (str): The name of the split.
            batch_size (int): The batch size for prediction.
            shuffle (bool, optional): Whether to shuffle the dataset. Defaults to False.
            verbose (bool, optional): Prints progress. Defaults to True.
            **kwargs: Additional keyword arguments for the DataLoader.

        Returns:
            ODPredictions: Predictions object to use for prediction set construction.
        """
        preds = self._load_preds_if_exists(
            dataset_name=dataset_name,
            split_name=split_name,
            task_name="object_detection",
        )
        if preds is not None:
            if verbose:
                print("Predictions already exist, loading them...")
            if isinstance(preds, ODPredictions):
                return preds
            else:
                raise ValueError(
                    f"Predictions already exist, but are of wrong type: {type(preds)}"
                )
        elif verbose:
            print("Predictions do not exist, building them...")

        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=shuffle, **kwargs
        )

        self.model.eval()

        pbar = enumerate(tqdm.tqdm(dataloader, disable=not verbose))

        all_image_paths = []
        all_true_boxes = []
        all_pred_boxes = []
        all_confidences = []
        all_true_cls = []
        all_pred_cls = []
        with torch.no_grad():
            for i, batch in pbar:

                res = self.predict_batch(batch)

                image_paths = res["image_paths"]
                true_boxes = res["true_boxes"]
                pred_boxes = res["pred_boxes"]
                confidences = res["confidences"]
                true_cls = res["true_cls"]
                pred_cls = res["pred_cls"]

                pred_boxes, pred_cls, confidences = self._filter_preds(
                    pred_boxes, pred_cls, confidences
                )

                all_image_paths.append(image_paths)
                all_true_boxes.append(true_boxes)
                all_pred_boxes.append(pred_boxes)
                all_confidences.append(confidences)
                all_true_cls.append(true_cls)
                all_pred_cls.append(pred_cls)

        all_image_paths = list(
            [path for arr_path in all_image_paths for path in arr_path]
        )
        all_true_boxes = list([box for arr_box in all_true_boxes for box in arr_box])
        all_pred_boxes = list([box for arr_box in all_pred_boxes for box in arr_box])
        all_confidences = list(
            [
                confidence
                for arr_confidence in all_confidences
                for confidence in arr_confidence
            ]
        )
        all_true_cls = list([cls for arr_cls in all_true_cls for cls in arr_cls])
        all_pred_cls = list(
            [proba for arr_proba in all_pred_cls for proba in arr_proba]
        )

        preds = ODPredictions(
            dataset_name=dataset_name,
            split_name=split_name,
            image_paths=all_image_paths,
            true_boxes=all_true_boxes,
            pred_boxes=all_pred_boxes,
            confidences=all_confidences,
            true_cls=all_true_cls,
            pred_cls=all_pred_cls,
        )
        self._save_preds(preds)
        return preds

    def _filter_preds(self, pred_boxes, pred_cls, confidences, iou_threshold=0.8):
        """
        Filters the predicted bounding boxes based on the confidence scores and IoU threshold.

        Args:
            pred_boxes: The predicted bounding boxes.
            pred_cls: The predicted class labels.
            confidences: The confidence scores.
            iou_threshold (float, optional): The IoU threshold for filtering. Defaults to 0.8.

        Returns:
            Tuple: The filtered predicted bounding boxes, predicted class labels, and confidence scores.
        """
        new_pred_boxes = []
        new_pred_cls = []
        new_confidences = []
        for i in range(len(pred_boxes)):
            keep = torchvision.ops.nms(
                pred_boxes[i], confidences[i], iou_threshold=iou_threshold
            )
            new_pred_box = pred_boxes[i].index_select(dim=0, index=keep)
            new_pred_cl = pred_cls[i].index_select(dim=0, index=keep)
            new_confidence = confidences[i].index_select(dim=0, index=keep)
            new_pred_boxes.append(new_pred_box)
            new_pred_cls.append(new_pred_cl)
            new_confidences.append(new_confidence)

        return new_pred_boxes, new_pred_cls, new_confidences

    def predict_batch(self, batch: list, **kwargs) -> dict:
        """
        Predicts the output given a batch of input tensors.

        Args:
            batch (list): The input batch.

        Returns:
            dict: The predicted output as a dictionary with the following keys:
                - "image_paths" (list): The paths of the input images
                - "true_boxes" (list): The true bounding boxes of the objects in the images
                - "pred_boxes" (list): The predicted bounding boxes of the objects in the images
                - "confidences" (list): The confidence scores of the predicted bounding boxes
                - "true_cls" (list): The true class labels of the objects in the images
                - "pred_cls" (list): The predicted class labels of the objects in the images
        """
        raise NotImplementedError("Please Implement this method")
