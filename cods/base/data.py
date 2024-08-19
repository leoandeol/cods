from time import time_ns
from typing import Optional


class Predictions:
    """
    Abstract class for predictions
    """

    def __init__(
        self,
        dataset_name: str,
        split_name: str,
        task_name: str,
        unique_id: Optional[int] = None,
    ):
        """
        Initializes a new instance of the Predictions class.
        Args:
            unique_id (int): The unique ID of the predictions.
            dataset_name (str): The name of the dataset.
            split_name (str): The name of the split.
            task_name (str): The name of the task.
        """
        if unique_id is None:
            unique_id = time_ns()
        self.unique_id = unique_id
        self.dataset_name = dataset_name
        self.split_name = split_name
        self.task_name = task_name


class Parameters:
    """Abstract class for parameters"""

    def __init__(self, predictions_id: int, unique_id: Optional[int] = None):
        """
        Initializes a new instance of the Parameters class.

        Parameters:
        predictions_id (int): The unique ID of the predictions.
        unique_id (int): The unique ID of the parameters.
        """
        self.predictions_id = predictions_id
        if unique_id is None:
            unique_id = time_ns()
        self.unique_id = unique_id


class ConformalizedPredictions:
    """Abstract class for results"""

    def __init__(
        self,
        predictions_id: int,
        parameters_id: int,
        unique_id: Optional[int] = None,
    ):
        """
        Initializes a new instance of the Data class.

        Parameters:
        predictions_id (int): The unique ID of the predictions.
        parameters_id (int): The unique ID of the parameters.
        """
        self.predictions_id = predictions_id
        self.parameters_id = parameters_id
        if unique_id is None:
            unique_id = time_ns()
        self.unique_id = unique_id


class Results:
    """Abstract class for results"""

    def __init__(
        self, predictions_id: int, parameters_id: int, conformalized_id: int
    ):
        """
        Initializes a new instance of the Data class.

        Parameters:
        predictions_id (int): The unique ID of the predictions.
        parameters_id (int): The unique ID of the parameters.
        conformalized_id (int): The unique ID of the conformalized predictions.
        """
        self.predictions_id = predictions_id
        self.parameters_id = parameters_id
        self.conformalized_id = conformalized_id
