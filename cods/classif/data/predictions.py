from typing import Dict, List, Union

import torch

from cods.base.data import Predictions


class ClassificationPredictions(Predictions):
    """Container for predictions from a classification model.

    Stores image paths, true and predicted class labels, and class index mapping for a classification task.
    """

    def __init__(
        self,
        dataset_name: str,
        split_name: str,
        image_paths: List[str],
        idx_to_cls: Union[Dict[int, str], None],
        true_cls: torch.Tensor,
        pred_cls: torch.Tensor,
    ):
        """Initialize ClassificationPredictions.

        Args:
        ----
            dataset_name (str): Name of the dataset.
            split_name (str): Name of the data split (e.g., 'train', 'val').
            image_paths (List[str]): List of image file paths.
            idx_to_cls (dict or None): Mapping from class indices to class names.
            true_cls (torch.Tensor): Ground truth class labels (N,).
            pred_cls (torch.Tensor): Model predictions (N, num_classes), before softmax.

        """
        super().__init__(dataset_name, split_name, task_name="classification")
        self.image_paths = image_paths
        self.true_cls = true_cls  # tensor: N
        self.pred_cls = pred_cls  # tensor: Nx1, before softmax
        self.idx_to_cls = idx_to_cls

        self.n_classes = len(self.pred_cls[0])

    def __len__(self):
        """Return the number of samples in the predictions."""
        return len(self.true_cls)

    def __str__(self):
        """Return a string summary of the predictions object."""
        return f"ClassificationPredictions_len={len(self)}"

    def split(self, splits_names: list, splits_ratios: list):
        """Split predictions into multiple splits.

        Args:
        ----
            splits_names (list): List of names for each split.
            splits_ratios (list): List of ratios for each split. Must sum to 1.

        Returns:
        -------
            list: List of ClassificationPredictions objects, one for each split.

        """
        assert sum(splits_ratios) == 1
        assert len(splits_names) == len(splits_ratios)
        n = len(self)
        splits = []
        for i in range(len(splits_names)):
            start = int(sum(splits_ratios[:i]) * n)
            end = int(sum(splits_ratios[: i + 1]) * n)
            splits.append(
                ClassificationPredictions(
                    dataset_name=self.dataset_name,
                    split_name=splits_names[i],
                    image_paths=self.image_paths[start:end],
                    true_cls=self.true_cls[start:end],
                    pred_cls=self.pred_cls[start:end],
                    idx_to_cls=self.idx_to_cls,
                ),
            )
        return splits
