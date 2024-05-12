from typing import Any, Optional

import torch

from cods.base.cp import Conformalizer
from cods.classif.data import ClassificationPredictions
from cods.classif.score import APSNCScore, ClassifNCScore, LACNCScore


class ClassificationConformalizer(Conformalizer):
    ACCEPTED_METHODS = {"lac": LACNCScore, "aps": APSNCScore}
    ACCEPTED_PREPROCESS = {"softmax": torch.softmax}

    def __init__(self, method="lac", preprocess="softmax"):
        if method not in self.ACCEPTED_METHODS.keys() and not isinstance(
            method, ClassifNCScore
        ):
            raise ValueError(
                f"method '{method}' not accepted, must be one of {self.ACCEPTED_METHODS} or a ClassifNCScore"
            )
        if preprocess not in self.ACCEPTED_PREPROCESS.keys():
            raise ValueError(
                f"preprocess '{preprocess}' not accepted, must be one of {self.ACCEPTED_PREPROCESS}"
            )

        self.method = method
        if isinstance(method, ClassifNCScore):
            self._score_function = method

        self.preprocess = preprocess
        self.f_preprocess = self.ACCEPTED_PREPROCESS[preprocess]
        self._score_function: Optional[Any] = None
        self._quantile: Optional[Any] = None
        self._n_classes: Optional[Any] = None

    def calibrate(
        self,
        preds: ClassificationPredictions,
        alpha: float = 0.1,
        verbose: bool = True,
        lbd_minus: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self._n_classes = preds.n_classes
        if self._score_function is None:
            self._score_function = self.ACCEPTED_METHODS[self.method](self._n_classes)
        scores = []

        n = 0

        for i, true_cls in enumerate(preds.true_cls):
            pred_cls = self.f_preprocess(preds.pred_cls[i], -1)
            score = self._score_function(pred_cls, true_cls)
            n += 1
            scores.append(score)

        scores = torch.stack(scores).ravel()
        self._scores = scores

        if lbd_minus:
            print("Using lbd_minus")
            quantile = torch.quantile(
                scores,
                1 - (alpha * (n + 1) / n),
                interpolation="higher",
            )
        else:
            print("Using lbd_plus")
            quantile = torch.quantile(
                scores,
                (1 - alpha) * ((n + 1) / n),
                interpolation="higher",
            )
        self._quantile = quantile

        if verbose:
            print(f"Calibrated quantile: {quantile}")

        return quantile, scores

    def conformalize(self, preds: ClassificationPredictions) -> list:
        if self._quantile is None:
            raise ValueError("Conformalizer must be calibrated before conformalizing.")

        conf_cls = []
        for pred_cls in preds.pred_cls:
            pred_cls = self.f_preprocess(pred_cls, -1)

            ys = self._score_function.get_set(
                pred_cls=pred_cls, quantile=self._quantile
            )
            conf_cls.append(ys)

        return conf_cls

    def evaluate(self, preds: ClassificationPredictions, conf_cls: list, verbose=True):
        if self._quantile is None:
            raise ValueError("Conformalizer must be calibrated before evaluating.")
        if conf_cls is None:
            raise ValueError("Predictions must be conformalized before evaluating.")
        losses = []
        set_sizes = []
        for i, true_cls_i in enumerate(preds.true_cls):
            true_cls_i = true_cls_i.cuda()
            conf_cls_i = conf_cls[i]
            loss = torch.isin(true_cls_i, conf_cls_i).float()
            losses.append(loss)
            set_size = torch.tensor(float(len(conf_cls_i)))
            set_sizes.append(set_size)
        losses = torch.stack(losses).ravel()
        set_sizes = torch.stack(set_sizes).ravel()
        if verbose:
            print(
                f"Coverage: {torch.mean(losses)}, Avg. set size: {torch.mean(set_sizes)}"
            )
        return losses, set_sizes
