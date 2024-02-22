from typing import Any, Optional, Union

import torch
from tqdm import tqdm

from cods.base.models import Model
from cods.classif.data import ClassificationPredictions


class ClassificationModel(Model):
    def __init__(
        self,
        model: torch.nn.Module,
        model_name: str,
        pretrained: bool = True,
        weights: Union[str, None] = None,
        device: str = "cuda",
        save: bool = True,
        save_dir_path: Union[str, None] = None,
    ):
        super().__init__(
            model_name=model_name,
            save_dir_path=save_dir_path,
            pretrained=pretrained,
            weights=weights,
            device=device,
        )
        self.model = model.to(device)
        self.model.eval()

    def build_predictions(
        self,
        dataset,
        dataset_name: str,
        split_name: str,
        batch_size: int,
        shuffle: bool = False,
        verbose: bool = True,
        **kwargs,
    ) -> ClassificationPredictions:
        preds = self._load_preds_if_exists(
            dataset_name=dataset_name, split_name=split_name, task_name="classification"
        )
        if preds is not None:
            if verbose:
                print("Predictions already exist, loading them...")
            if isinstance(preds, ClassificationPredictions):
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
        true_cls = []
        pred_cls = []
        if verbose:
            print("Building predictions...")
        ids = []
        with torch.no_grad():
            for i, data in enumerate(tqdm(dataloader, disable=not verbose)):
                id, images, labels = data
                images = images.to(self.device)
                labels = labels.to(self.device)
                ids.extend(id)
                preds = self.model(images)
                true_cls.extend(labels)
                pred_cls.extend(preds)
        predictions = {}
        predictions["dataset_name"] = dataset_name
        predictions["split_name"] = split_name
        paths = ids  # dataset.get_paths(ids)
        predictions["image_paths"] = paths
        predictions["idx_to_cls"] = dataset.idx_to_cls
        predictions["pred_cls"] = torch.stack(predictions["pred_cls"])
        predictions["true_cls"] = torch.stack(predictions["true_cls"])
        preds = ClassificationPredictions(**predictions)
        self._save_preds(preds)
        return preds
