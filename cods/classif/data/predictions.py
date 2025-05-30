from typing import Dict, List

import torch

from cods.base.data import Predictions


class ClassificationPredictions(Predictions):
    """Predictions for classification tasks"""

    def __init__(
        self,
        dataset_name: str,
        split_name: str,
        image_paths: List[str],
        idx_to_cls: Dict[int, str],
        true_cls: torch.Tensor,
        pred_cls: List[torch.Tensor],
    ):
        super().__init__(dataset_name, split_name, task_name="classification")
        self.image_paths = image_paths
        self.true_cls = true_cls  # tensor: N
        self.pred_cls = pred_cls  # tensor: Nx1, before softmax
        self.idx_to_cls = idx_to_cls

        self.n_classes = len(self.pred_cls[0])

    def __len__(self):
        return len(self.true_cls)

    def __str__(self):
        return f"ClassificationPredictions_len={len(self)}"

    def split(self, splits_names: list, splits_ratios: list):
        """Split predictions into multiple splits

        Args:
            splits_names (list): list of names for each split
            splits_ratios (list): list of ratios for each split

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
