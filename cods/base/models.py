import os
import pickle
from logging import getLogger
from typing import Optional

import torch

from cods.base.data import Predictions

logger = getLogger("cods")


class Model:
    def __init__(
        self,
        model_name: str,
        save_dir_path: str,
        pretrained=True,
        weights=None,
        device="cpu",
    ):
        self.model_name = model_name
        self.model = None  # TODO: add model loading
        self.pretrained = pretrained
        self.weights = weights
        self.device = device
        if save_dir_path is None:
            save_dir_path = "./saved_predictions"
        self.save_dir_path = save_dir_path
        logger.info(f"Model {model_name} initialized")

    def build_predictions(
        self,
        dataloader: torch.utils.data.DataLoader,
        verbose=True,
        **kwargs,
    ) -> Predictions:
        raise NotImplementedError("Please Implement this method")

    def _save_preds(self, predictions: Predictions, hash: str, verbose=True):
        """Save predictions to file

        Args:
            predictions (Predictions): predictions object
            path (str): path to file

        """
        path = f"{self.save_dir_path}/{hash}.pkl"
        # create directory if it doesn't exist
        dir_path = os.path.dirname(path)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        # check if file exists
        if verbose:
            logger.info(f"Saving predictions to {path}")
        if os.path.exists(path):
            logger.warning(f"File {path} already exists, overwriting it")
        with open(path, "wb") as f:
            pickle.dump(predictions, f)

    def _load_preds_if_exists(
        self,
        hash: str,
        verbose: bool = True,
    ) -> Optional[Predictions]:
        """Load predictions if they exist, else return None

        Args:
            path (str): path to predictions file

        Returns:
            Predictions: predictions object

        """
        path = f"{self.save_dir_path}/{hash}.pkl"
        if not os.path.exists(path):
            logger.error(f"File {path} does not exist")
            return None
        with open(path, "rb") as f:
            if verbose:
                logger.info(f"Loading predictions from {path}")
            return pickle.load(f)
