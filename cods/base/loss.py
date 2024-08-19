import torch

from cods.base.data import ConformalizedPredictions, Predictions


# Non-Conformity Score for classical CP
class NCScore:
    def __init__(self):
        pass


# todo: perhaps rename to something better than loss, to avoid confusion with usual losses
class Loss:
    def __init__(self):
        pass

    def __call__(
        self,
        predictions: Predictions,
        conformalized_predictions: ConformalizedPredictions,
        **kwargs,
    ) -> torch.Tensor:
        raise NotImplementedError()


# class MultiLoss(Loss):
#     def __init__(self):
#         pass

#     def __call__(self, **kwargs):
#         raise NotImplementedError()


# class BonferroniMultiLoss(Loss):
#     def __init__(self):
#         pass

#     def __call__(self, **kwargs):
#         raise NotImplementedError()


# class HMPMultiLoss(Loss):
#     def __init__(self):
#         pass

#     def __call__(self, **kwargs):
#         print("WARNING: ASYMPOTITC GUARANTEES ONLY")
#         raise NotImplementedError()
