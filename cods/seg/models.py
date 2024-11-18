from logging import getLogger

import torch

from cods.base.model import Model
from cods.seg.data import SegmentationPredictions

logger = getLogger("cods")


class SegmentationModel(Model):
    def __init__(
        self,
        model_name: str,
        save_dir_path: str,
        pretrained=True,
        weights=None,
        device="cpu",
    ):
        super().__init__(model_name, save_dir_path, pretrained, weights, device)
        
        

    def build_predictions(
        self, dataloader: torch.utils.data.DataLoader, verbose=True, **kwargs
    ) -> SegmentationPredictions:
        raise NotImplementedError("Please Implement this method")
