import torch

from cods.base.loss import NCScore


class ClassifNCScore(NCScore):
    """Abstract class for classification non-conformity score functions"""

    def __init__(self, n_classes, **kwargs):
        super().__init__()
        self.n_classes = n_classes

    def __call__(self, **kwargs):
        raise NotImplementedError("ClassifNCScore is an abstract class.")

    def get_set(self, pred_cls, quantile):
        ys = list(range(self.n_classes))
        pred_set = list(
            [y for y in ys if self._score_function(pred_cls, y) <= quantile]
        )
        pred_set = torch.tensor(pred_set)
        return pred_set


class LACNCScore(ClassifNCScore):
    def __init__(self, n_classes: int, **kwargs):
        super().__init__(n_classes=n_classes)

    def __call__(self, pred_cls: torch.Tensor, y: int, **kwargs):
        return 1 - pred_cls[y]

    def get_set(self, pred_cls, quantile):
        return torch.where(pred_cls >= 1 - quantile)[0]


class APSNCScore(ClassifNCScore):
    def __init__(self, n_classes, **kwargs):
        super().__init__(n_classes=n_classes)

    def __call__(self, pred_cls: torch.Tensor, y: int, **kwargs):
        values, indices = torch.sort(pred_cls, descending=True)
        cumsum = torch.cumsum(values, dim=0)
        return cumsum[torch.where(indices == y)[0]]

    def get_set(self, pred_cls: torch.Tensor, quantile: float):
        values, indices = torch.sort(pred_cls, descending=True)
        cumsum = torch.cumsum(values, dim=0)
        try:
            idxs = torch.where(cumsum >= quantile)[0]
            if len(idxs)>0:
                k = idxs[0]
            else:
                k = 0
        except:
            print(pred_cls.sum(), pred_cls.size())
            print(torch.where(cumsum >= quantile))
            print(cumsum)
        return indices[: k + 1]
