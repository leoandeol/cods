class Predictions:
    """Abstract class for predictions"""

    def __init__(self, dataset_name, split_name, task_name):
        self.dataset_name = dataset_name
        self.split_name = split_name
        self.task_name = task_name
