import os
import pickle
from typing import Union

import torch

from cods.base.data import Predictions


class Model:
    def __init__(
        self,
        model_name: str,
        save_dir_path: Union[str, None],
        pretrained: bool = True,
        weights: Union[str, None] = None,
        device: str = "cuda",
    ):
        self.model_name = model_name
        self.pretrained = pretrained
        self.weights = weights
        self.device = device
        if save_dir_path is None:
            save_dir_path = "./saved_predictions"
        self.save_dir_path = save_dir_path

    def build_predictions(
        self, dataloader: torch.utils.data.DataLoader, verbose=True, **kwargs
    ) -> Predictions:
        raise NotImplementedError("Please Implement this method")

    "./saved_predictions/model_name/dataset_name/split_name/predictions.pkl"

    def _save_preds(self, predictions: Predictions):
        """Save predictions to file

        Args:
            predictions (Predictions): predictions object
            path (str): path to file
        """
        path = f"{self.save_dir_path}/{self.model_name}/{predictions.dataset_name}"
        path = (
            f"{path}/{predictions.split_name}/predictions_{predictions.task_name}.pkl"
        )
        # create directory if it doesn't exist
        dir_path = os.path.dirname(path)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        # check if file exists
        print(f"Saving predictions to {path}")
        if os.path.exists(path):
            print(f"File {path} already exists, overwriting it")
        with open(path, "wb") as f:
            pickle.dump(predictions, f)

    def _load_preds_if_exists(
        self,
        dataset_name: str,
        split_name: str,
        task_name: str,
    ) -> Union[Predictions, None]:
        """Load predictions if they exist, else return None

        Args:
            path (str): path to predictions file

        Returns:
            Predictions: predictions object
        """
        path = f"{self.save_dir_path}/{self.model_name}/{dataset_name}/{split_name}/predictions_{task_name}.pkl"
        if not os.path.exists(path):
            print(f"File {path} does not exist")
            return None
        with open(path, "rb") as f:
            print(f"Loading predictions from {path}")
            return pickle.load(f)
