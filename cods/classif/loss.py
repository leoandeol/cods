import torch

from cods.base.loss import Loss


class ClassificationLoss(Loss):
    def __init__(self, upper_bound: float, **kwargs):
        super().__init__()
        self.upper_bound = upper_bound

    def __call__(self, true_cls: torch.Tensor, conf_cls: torch.Tensor, **kwargs):
        raise NotImplementedError("ClassifLoss is an abstract class.")

    def get_set(self, pred_cls: torch.Tensor, lbd: float):
        raise NotImplementedError("ClassifLoss is an abstract class.")


# TODO: split in two classes: one for the loss and one for the set construction


class LACLoss(ClassificationLoss):
    def __init__(self, upper_bound: float = 1, **kwargs):
        super().__init__(upper_bound=upper_bound)

    def __call__(self, true_cls: torch.Tensor, conf_cls: torch.Tensor):
        """
        Computes the LAC loss.
        :param true_cls: true class of the sample
        :param conf_cls: conformalized prediction of the sample
        :return: LAC loss
        """
        return 1 - torch.isin(true_cls, conf_cls).float()

    def get_set(self, pred_cls: torch.Tensor, lbd: float):
        return torch.where(pred_cls >= 1 - lbd)[0]


CLASSIFICATION_LOSSES = {"lac": LACLoss}
