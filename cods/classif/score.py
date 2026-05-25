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
        pred_set = [y for y in ys if self._score_function(pred_cls, y) <= quantile]
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
        # Ensure input is a probability distribution
        if not torch.isclose(pred_cls.sum(), torch.tensor(1.0), atol=1e-3):
            raise ValueError(
                f"Input pred_cls should be a probability vector, but sums to {pred_cls.sum().item()}",
            )

        values, indices = torch.sort(pred_cls, descending=True)
        cumsum = torch.cumsum(values, dim=0)
        match = torch.where(indices == y)[0]
        if len(match) == 0:
            raise ValueError(f"Label {y} not found in indices.")

        # Randomized nonconformity score for calibration
        u = torch.rand(1).item() * values[match[0]]
        score = cumsum[match[0]] - values[match[0]] + u
        final_score, _ = torch.max(score, 0)
        return final_score

    def get_set(
        self,
        pred_cls: torch.Tensor,
        quantile: float,
    ):
        # TODO: tmp fix
        pred_cls /= pred_cls.sum()
        # In many cases the probabilities don't sum to 1
        # This case is handled in the else : defaults to the full prediction set.
        values, indices = torch.sort(pred_cls, descending=True)
        cumsum = torch.cumsum(values, dim=0)
        idxs = torch.where(cumsum > quantile)[0]
        if len(idxs) > 0:
            k = idxs[0]
        else:
            k = len(cumsum) - 1
        return indices[: k + 1]
