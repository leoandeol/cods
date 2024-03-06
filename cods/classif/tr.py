from typing import Callable, Union

import torch

from cods.base.tr import ToleranceRegion
from cods.classif.data import ClassificationPredictions
from cods.classif.loss import CLASSIFICATION_LOSSES, ClassificationLoss


class ClassificationToleranceRegion(ToleranceRegion):
    ACCEPTED_PREPROCESS = {"softmax": torch.softmax}

    def __init__(
        self,
        loss: Union[str, ClassificationLoss] = "lac",
        inequality: Union[str, Callable] = "binomial_inverse_cdf",
        optimizer: str = "binary_search",
        preprocess: Union[str, Callable] = "softmax",
        optimizer_args: dict = {},
    ):
        super().__init__(
            inequality=inequality, optimizer=optimizer, optimizer_args=optimizer_args
        )
        self.ACCEPTED_LOSSES = CLASSIFICATION_LOSSES
        self.lbd = None
        if isinstance(preprocess, str):
            if preprocess not in self.ACCEPTED_PREPROCESS.keys():
                raise ValueError(
                    f"preprocess '{preprocess}' not accepted, must be one of {self.ACCEPTED_PREPROCESS.keys()}"
                )
            self.preprocess = preprocess
            self.f_preprocess = self.ACCEPTED_PREPROCESS[preprocess]
        elif isinstance(preprocess, Callable):
            self.preprocess_name = preprocess.__name__
            self.f_preprocess = preprocess
        else:
            raise ValueError(
                "preprocess must be a string or a callable function, got {type(preprocess)}"
            )

        if isinstance(loss, str):
            if loss not in self.ACCEPTED_LOSSES:
                raise ValueError(
                    f"Loss {loss} not supported. Choose from {self.ACCEPTED_LOSSES}."
                )
            self.loss_name = loss
            self.loss = self.ACCEPTED_LOSSES[loss]()
        elif isinstance(loss, ClassificationLoss):
            self.loss_name = loss.__class__.__name__
            self.loss = loss
        else:
            raise ValueError(
                f"loss must be a string or a ClassificationLoss instance, got {loss}"
            )

    def calibrate(
        self,
        predictions: ClassificationPredictions,
        alpha=0.1,
        delta=0.1,
        steps=13,
        bounds=[0, 1],
        verbose=True,
        objectness_threshold=0.8,
    ):
        if self.lbd is not None:
            print("Replacing previously computed lambda")
        self._n_classes = predictions.n_classes
        risk_function = self._get_risk_function(
            predictions=predictions,
            alpha=alpha,
            delta=delta,
            objectness_threshold=objectness_threshold,
            # matching=matching,
        )

        lbd = self.optimizer.optimize(
            risk_function=risk_function,
            alpha=alpha,
            bounds=bounds,
            steps=steps,
            verbose=verbose,
        )
        self.lbd = lbd
        self.confidence_threshold = objectness_threshold
        return lbd

    def _get_risk_function(
        self,
        predictions: ClassificationPredictions,
        alpha: float,
        delta: float,
        objectness_threshold: float,
        **kwargs,
    ) -> Callable:
        n = len(predictions.true_cls)

        def risk_function(lbd):
            risk = []
            for i, true_cls in enumerate(predictions.true_cls):
                # for j, true_cls in enumerate(true_cls_img):
                # pred_cls = predictions.cls_prob[i][matching[i][j]]
                pred_cls = self.f_preprocess(predictions.pred_cls[i], -1)
                conf_set = self.loss.get_set(pred_cls=pred_cls, lbd=lbd)
                score = self.loss(true_cls=true_cls, conf_cls=conf_set)
                risk.append(score)
            risk = torch.stack(risk)
            risk = torch.mean(risk)
            corrected_risk = self._correct_risk(
                risk=risk,
                n=n,
                delta=delta,
            )
            return corrected_risk

        return risk_function

    def _correct_risk(
        self,
        risk: torch.Tensor,
        n: int,
        delta: float,
    ) -> torch.Tensor:
        # TODO: fix
        return self.f_inequality(
            Rhat=risk,
            n=torch.tensor(n, dtype=torch.float).cuda(),
            delta=torch.tensor(delta, dtype=torch.float).cuda(),
        )

    def conformalize(
        self, predictions: ClassificationPredictions, verbose: bool = True, **kwargs
    ) -> list:
        if self.lbd is None:
            raise ValueError("Conformalizer must be calibrated before conformalizing.")
        conf_cls = []
        for pred_cls in predictions.pred_cls:
            pred_cls = self.f_preprocess(pred_cls, -1)
            ys = self.loss.get_set(pred_cls=pred_cls, lbd=self.lbd)
            conf_cls.append(ys)
        return conf_cls

    def evaluate(
        self, preds: ClassificationPredictions, conf_cls: list, verbose=True, **kwargs
    ):
        if self.lbd is None:
            raise ValueError("Conformalizer must be calibrated before evaluating.")
        losses = []
        set_sizes = []
        for i, true_cls in enumerate(preds.true_cls):
            conf_cls_i = conf_cls[i]
            loss = torch.tensor(float(true_cls in conf_cls_i))
            losses.append(loss)
            set_size = torch.tensor(float(len(conf_cls_i)))
            set_sizes.append(set_size)
        losses = torch.stack(losses)
        set_sizes = torch.stack(set_sizes)
        if verbose:
            print(
                f"Coverage: {torch.mean(losses)}, Avg. set size: {torch.mean(set_sizes)}"
            )
        return losses, set_sizes
