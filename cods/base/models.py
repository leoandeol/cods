"""Base model class for machine learning models in the cods library."""

import os
import pickle
from logging import getLogger
from typing import Optional

import torch

from cods.base.data import Predictions

logger = getLogger("cods")


class Model:
    """Abstract base class for models in the cods library."""

    def __init__(
        self,
        model_name: str,
        save_dir_path: str,
        pretrained=True,
        weights=None,
        device="cpu",
    ):
        """Initialize the Model base class.

        Args:
        ----
            model_name (str): Name of the model.
            save_dir_path (str): Directory path to save predictions.
            pretrained (bool, optional): Whether to use pretrained weights. Defaults to True.
            weights (optional): Model weights. Defaults to None.
            device (str, optional): Device to use. Defaults to 'cpu'.

        """
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
        """Build predictions for the given dataloader.

        Args:
        ----
            dataloader (torch.utils.data.DataLoader): DataLoader to generate predictions from.
            verbose (bool, optional): Whether to print progress. Defaults to True.
            **kwargs: Additional arguments.

        Returns:
        -------
            Predictions: Predictions object.

        Raises:
        ------
            NotImplementedError: If not implemented in subclass.

        """
        raise NotImplementedError("Please Implement this method")

    def _save_preds(self, predictions: Predictions, hash: str, verbose=True):
        """Save predictions to file.

        Args:
        ----
            predictions (Predictions): Predictions object to save.
            hash (str): Hash string for filename.
            verbose (bool, optional): Whether to print progress. Defaults to True.

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
        """Load predictions from file if they exist, else return None.

        Args:
        ----
            hash (str): Hash string for filename.
            verbose (bool, optional): Whether to print progress. Defaults to True.

        Returns:
        -------
            Optional[Predictions]: Loaded predictions object, or None if not found.

        """
        path = f"{self.save_dir_path}/{hash}.pkl"
        if not os.path.exists(path):
            logger.error(f"File {path} does not exist")
            return None
        with open(path, "rb") as f:
            if verbose:
                logger.info(f"Loading predictions from {path}")
            return pickle.load(f)
