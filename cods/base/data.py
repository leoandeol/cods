"""Base data structures for predictions, parameters, and conformalized results."""

from time import time_ns
from typing import Union


class Predictions:
    """Abstract base class for predictions.

    Attributes
    ----------
        unique_id (int): Unique ID of the predictions.
        dataset_name (str): Name of the dataset.
        split_name (str): Name of the split.
        task_name (str): Name of the task.

    """

    def __init__(
        self,
        dataset_name: str,
        split_name: str,
        task_name: str,
        unique_id: Union[int, None] = None,
    ):
        """Initialize a new instance of the Predictions class.

        Args:
        ----
            dataset_name (str): Name of the dataset.
            split_name (str): Name of the split.
            task_name (str): Name of the task.
            unique_id (Optional[int], optional): Unique ID of the predictions. If None, a timestamp is used.

        """
        if unique_id is None:
            unique_id = time_ns()
        self.unique_id = unique_id
        self.dataset_name = dataset_name
        self.split_name = split_name
        self.task_name = task_name


class Parameters:
    """Abstract base class for parameters.

    Attributes
    ----------
        predictions_id (int): Unique ID of the predictions.
        unique_id (int): Unique ID of the parameters.

    """

    def __init__(self, predictions_id: int, unique_id: Union[int, None] = None):
        """Initialize a new instance of the Parameters class.

        Args:
        ----
            predictions_id (int): Unique ID of the predictions.
            unique_id (Optional[int], optional): Unique ID of the parameters. If None, a timestamp is used.

        """
        self.predictions_id = predictions_id
        if unique_id is None:
            unique_id = time_ns()
        self.unique_id = unique_id


class ConformalizedPredictions:
    """Abstract base class for conformalized prediction results.

    Attributes
    ----------
        predictions_id (int): Unique ID of the predictions.
        parameters_id (int): Unique ID of the parameters.
        unique_id (int): Unique ID of the conformalized predictions.

    """

    def __init__(
        self,
        predictions_id: int,
        parameters_id: int,
        unique_id: Union[int, None] = None,
    ):
        """Initialize a new instance of the ConformalizedPredictions class.

        Args:
        ----
            predictions_id (int): Unique ID of the predictions.
            parameters_id (int): Unique ID of the parameters.
            unique_id (Optional[int], optional): Unique ID of the conformalized predictions. If None, a timestamp is used.

        """
        self.predictions_id = predictions_id
        self.parameters_id = parameters_id
        if unique_id is None:
            unique_id = time_ns()
        self.unique_id = unique_id


class Results:
    """Abstract base class for results.

    Attributes
    ----------
        predictions_id (int): Unique ID of the predictions.
        parameters_id (int): Unique ID of the parameters.
        conformalized_id (int): Unique ID of the conformalized predictions.

    """

    def __init__(
        self,
        predictions_id: int,
        parameters_id: int,
        conformalized_id: int,
    ):
        """Initialize a new instance of the Results class.

        Args:
        ----
            predictions_id (int): Unique ID of the predictions.
            parameters_id (int): Unique ID of the parameters.
            conformalized_id (int): Unique ID of the conformalized predictions.

        """
        self.predictions_id = predictions_id
        self.parameters_id = parameters_id
        self.conformalized_id = conformalized_id
