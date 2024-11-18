from cods.base.data import Predictions

class SegmentationPredictions(Predictions):
    def __init__(self, predictions):
        self.predictions = predictions

    def __len__(self):
        return len(self.predictions)