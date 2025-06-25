"""Model wrapper for conformal classification tasks."""

import torch
from tqdm import tqdm

from cods.base.models import Model
from cods.classif.data import ClassificationPredictions


class ClassificationModel(Model):
    """Model wrapper for classification tasks with prediction saving/loading."""

    def __init__(
        self,
        model,
        model_name,
        pretrained=True,
        weights=None,
        device="cpu",
        save=True,
        save_dir_path=None,
    ):
        """Initialize the ClassificationModel.

        Args:
        ----
            model: The underlying PyTorch model.
            model_name (str): Name of the model.
            pretrained (bool, optional): Whether to use pretrained weights. Defaults to True.
            weights (optional): Model weights. Defaults to None.
            device (str, optional): Device to use. Defaults to 'cpu'.
            save (bool, optional): Whether to save predictions. Defaults to True.
            save_dir_path (str, optional): Directory to save predictions. Defaults to None.

        """
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
        """Build predictions for the given dataset and save/load as needed.

        Args:
        ----
            dataset: Dataset to build predictions for.
            dataset_name (str): Name of the dataset.
            split_name (str): Name of the data split.
            batch_size (int): Batch size for prediction.
            shuffle (bool, optional): Whether to shuffle the data. Defaults to False.
            verbose (bool, optional): Whether to print progress. Defaults to True.
            **kwargs: Additional arguments for DataLoader.

        Returns:
        -------
            ClassificationPredictions: Predictions object for the dataset.

        """
        preds = self._load_preds_if_exists(
            dataset_name=dataset_name,
            split_name=split_name,
            task_name="classification",
        )
        if preds is not None:
            if verbose:
                print("Predictions already exist, loading them...")
            return preds
        if verbose:
            print("Predictions do not exist, building them...")

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            **kwargs,
        )
        true_cls = []
        pred_cls = []
        if verbose:
            print("Building predictions...")
        ids = []
        with torch.no_grad():
            for _i, data in enumerate(tqdm(dataloader, disable=not verbose)):
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
