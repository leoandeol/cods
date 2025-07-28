import torch

from cods.base.data import ConformalizedPredictions, Predictions


# Non-Conformity Score for classical CP
class NCScore:
    def __init__(self):
        pass


# TODO: perhaps rename to something better than loss, to avoid confusion with usual losses
class Loss:
    def __init__(self):
        pass

    def __call__(
        self,
        predictions: Predictions,
        conformalized_predictions: ConformalizedPredictions,
        **kwargs,
    ) -> torch.Tensor:
        raise NotImplementedError
