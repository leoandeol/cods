from collections.abc import Callable
from logging import getLogger

import numpy as np
import torch
from tqdm import tqdm

from cods.base.optim import Optimizer
from cods.od.data import ODPredictions
from cods.od.loss import ODLoss
from cods.od.utils import apply_margins, match_predictions_to_true_boxes

logger = getLogger("cods")


# TODO(leo): only image level currently
class FirstStepMonotonizingOptimizer(Optimizer):
    def __init__(self):
        pass

    def _correct_risk(
        self,
        risk: torch.Tensor,
        n: int,
        B: float,
    ) -> torch.Tensor:
        """Correct the risk using the number of predictions and the upper bound.

        Parameters
        ----------
        - risk (torch.Tensor): The risk tensor.
        - n (int): The number of predictions.
        - B (float): The upper bound.

        Returns
        -------
        - corrected_risk (torch.Tensor): The corrected risk tensor.

        """
        return (torch.sum(torch.stack(risk)) + B) / (n + 1)

    def optimize(
        self,
        predictions: ODPredictions,
        confidence_loss: ODLoss,
        localization_loss: ODLoss,
        classification_loss: ODLoss,
        matching_function,
        alpha: float,
        device: str,
        B: float = 1,
        init_lambda: float = 1,
        verbose: bool = False,
    ):
        true_boxes = predictions.true_boxes
        pred_boxes = predictions.pred_boxes
        true_cls = predictions.true_cls
        pred_cls = predictions.pred_cls
        image_shapes = predictions.image_shapes
        confidences = predictions.confidences

        stacked_confidences = torch.concatenate(
            confidences,
        )
        confidence_image_idx = torch.concatenate(
            [
                torch.ones_like(x, dtype=int) * i
                for i, x in enumerate(confidences)
            ],
        )

        sorted_stacked_confidences, indices = torch.sort(stacked_confidences)
        sorted_stacked_confidences[:-1] = sorted_stacked_confidences[
            1:
        ].clone()
        sorted_stacked_confidences[-1] = 1.0  # last one is always 1.0
        sorted_confidence_image_indices = confidence_image_idx[indices]

        lambda_conf = init_lambda

        confidence_losses = []
        localization_losses = []
        classification_losses = []

        # Step 0: Initialize the risk

        match_predictions_to_true_boxes(
            predictions,
            distance_function=matching_function,
            verbose=False,
            overload_confidence_threshold=1 - lambda_conf,
        )

        # TODO(leo):parallelize?
        # Step 1: Compute the risk
        for i in tqdm(range(len(predictions)), disable=not verbose):
            true_boxes_i = true_boxes[i]
            pred_boxes_i = pred_boxes[i]
            confidences_i = confidences[i]
            true_cls_i = true_cls[i]
            pred_cls_i = pred_cls[i]
            image_shape = image_shapes[i]

            matching_i = predictions.matching[i]

            pred_boxes_i = pred_boxes_i[confidences_i >= 1 - lambda_conf]
            pred_cls_i = [
                x
                for x, c in zip(pred_cls_i, confidences_i)
                if c >= 1 - lambda_conf
            ]
            pred_cls_i = (
                torch.stack(pred_cls_i)
                if len(pred_cls_i) > 0
                else torch.tensor([]).float().to(device)
            )

            # no confidence filtering here because lambda_conf = 1 for this first loop
            confidence_loss_i = confidence_loss(
                true_boxes_i,
                true_cls_i,
                pred_boxes_i,
                pred_cls_i,
            )

            tmp_matched_boxes_i = [
                (
                    torch.stack([pred_boxes_i[m] for m in matching_i[j]])[0]
                    if len(matching_i[j]) > 0
                    else torch.tensor([]).float().to(device)
                )
                for j in range(len(true_boxes_i))
            ]
            matched_pred_boxes_i = (
                torch.stack(tmp_matched_boxes_i)
                if len(tmp_matched_boxes_i) > 0
                else torch.tensor([]).float().to(device)
            )
            matched_pred_cls_i = [
                (
                    torch.stack([pred_cls_i[m] for m in matching_i[j]])[
                        0
                    ]  # TODO zero here ?
                    if len(matching_i[j]) > 0
                    else torch.tensor([]).float().to(device)
                )
                for j in range(len(true_boxes_i))
            ]
            margin = np.concatenate((image_shape, image_shape))
            matched_conf_boxes_i = apply_margins(
                [matched_pred_boxes_i],
                margin,
                mode="additive",  # TODO fix this
            )[0]

            n_classes = len(predictions.pred_cls[0][0].squeeze())
            matched_conf_cls_i = [
                torch.arange(n_classes)[None, ...].to(device)
                for _ in range(len(matched_pred_cls_i))
            ]

            localization_loss_i = localization_loss(
                true_boxes_i,
                true_cls_i,
                matched_conf_boxes_i,
                matched_conf_cls_i,
            )
            classification_loss_i = classification_loss(
                true_boxes_i,
                true_cls_i,
                matched_conf_boxes_i,
                matched_conf_cls_i,
            )

            confidence_losses.append(confidence_loss_i)
            localization_losses.append(localization_loss_i)
            classification_losses.append(classification_loss_i)

        # TODO(leo): image level
        confidence_risk = self._correct_risk(
            confidence_losses,
            len(predictions),
            B,
        )
        localization_risk = self._correct_risk(
            localization_losses,
            len(predictions),
            B,
        )
        classification_risk = self._correct_risk(
            classification_losses,
            len(predictions),
            B,
        )

        # ------- LOGGING -------
        _log_raw_confidence_losses = confidence_losses.copy()
        _log_raw_localization_losses = localization_losses.copy()
        _log_raw_classification_losses = classification_losses.copy()

        max_risk = torch.max(
            torch.stack(
                [confidence_risk, localization_risk, classification_risk],
            ),
        )
        logger.info(f"First risk: {max_risk.detach().cpu().numpy()}")
        if max_risk.detach().cpu().numpy() > alpha:
            # Debug: all three risks to see why there isn't any solution
            logger.debug(f"Confidence risk: {confidence_risk}")
            logger.debug(f"Localization risk: {localization_risk}")
            logger.debug(f"Classification risk: {classification_risk}")
            logger.debug(f"Max risk: {max_risk} > {alpha}. No solution found.")
            logger.warning(
                "There does not exist any solution satisfying the constraints.",
            )
            return 1.0
        logger.debug(
            f"Risk after 1st epoch is {max_risk.detach().cpu().numpy()} < {alpha}",
        )

        previous_lbd = lambda_conf

        pbar = tqdm(
            list(
                zip(
                    sorted_confidence_image_indices,
                    sorted_stacked_confidences,
                ),
            ),
            disable=not verbose,
        )

        self.all_risks_raw = [max_risk.detach().cpu().numpy()]
        self.all_risks_raw_conf = [confidence_risk.detach().cpu().numpy()]
        self.all_risks_raw_loc = [localization_risk.detach().cpu().numpy()]
        self.all_risks_raw_cls = [classification_risk.detach().cpu().numpy()]
        self.all_risks_mon = [max_risk.detach().cpu().numpy()]
        self.all_risks_mon_conf = [confidence_risk.detach().cpu().numpy()]
        self.all_risks_mon_loc = [localization_risk.detach().cpu().numpy()]
        self.all_risks_mon_cls = [classification_risk.detach().cpu().numpy()]
        self.all_lbds = [lambda_conf]

        # Step 2: Update one loss at a time
        for image_id, conf_score in pbar:
            previous_lbd = lambda_conf
            lambda_conf = 1 - conf_score.cpu().numpy().item()
            if lambda_conf > init_lambda:
                continue

            i = image_id
            true_boxes_i = true_boxes[i]
            pred_boxes_i = pred_boxes[i]
            confidences_i = confidences[i]
            true_cls_i = true_cls[i]
            pred_cls_i = pred_cls[i]
            image_shape = image_shapes[i]

            pred_boxes_i = pred_boxes_i[confidences_i >= 1 - lambda_conf]
            pred_cls_i = [
                x
                for x, c in zip(pred_cls_i, confidences_i)
                if c >= 1 - lambda_conf
            ]
            pred_cls_i = (
                torch.stack(pred_cls_i)
                if len(pred_cls_i) > 0
                else torch.tensor([]).float().to(device)
            )

            confidence_loss_i = confidence_loss(
                true_boxes_i,
                true_cls_i,
                pred_boxes_i,
                pred_cls_i,
            )

            matching_i = match_predictions_to_true_boxes(
                predictions,
                distance_function=matching_function,
                verbose=False,
                overload_confidence_threshold=1 - lambda_conf,
                idx=i,
            )

            predictions.matching[i] = matching_i

            tmp_matched_boxes_i = [
                (
                    torch.stack([pred_boxes_i[m] for m in matching_i[j]])[0]
                    if len(matching_i[j]) > 0
                    else torch.tensor([]).float().to(device)
                )
                for j in range(len(true_boxes_i))
            ]
            matched_pred_boxes_i = (
                torch.stack(tmp_matched_boxes_i)
                if len(tmp_matched_boxes_i) > 0
                else torch.tensor([]).float().to(device)
            )
            matched_pred_cls_i = [
                (
                    torch.stack([pred_cls_i[m] for m in matching_i[j]])[0]
                    if len(matching_i[j]) > 0
                    else torch.tensor([]).float().to(device)
                )
                for j in range(len(true_boxes_i))
            ]

            margin = np.concatenate((image_shape, image_shape))
            matched_conf_boxes_i = apply_margins(
                [matched_pred_boxes_i],
                margin,
                mode="additive",  # TODO: fix this
            )[0]

            n_classes = len(predictions.pred_cls[0][0].squeeze())
            matched_conf_cls_i = [
                torch.arange(n_classes)[None, ...].to(device)
                for _ in range(len(matched_pred_cls_i))
            ]

            localization_loss_i = localization_loss(
                true_boxes_i,
                true_cls_i,
                matched_conf_boxes_i,
                matched_conf_cls_i,
            )
            classification_loss_i = classification_loss(
                true_boxes_i,
                true_cls_i,
                matched_conf_boxes_i,
                matched_conf_cls_i,
            )

            _log_raw_confidence_losses[i] = confidence_loss_i.detach().clone()
            _log_raw_localization_losses[i] = (
                localization_loss_i.detach().clone()
            )
            _log_raw_classification_losses[i] = (
                classification_loss_i.detach().clone()
            )

            confidence_losses[i] = max(
                confidence_loss_i.detach().clone(),
                confidence_losses[i].clone(),
            )
            localization_losses[i] = max(
                localization_loss_i.detach().clone(),
                localization_losses[i].clone(),
            )
            classification_losses[i] = max(
                classification_loss_i.detach().clone(),
                classification_losses[i].clone(),
            )

            confidence_risk = self._correct_risk(
                confidence_losses,
                len(predictions),
                B,
            )
            localization_risk = self._correct_risk(
                localization_losses,
                len(predictions),
                B,
            )
            classification_risk = self._correct_risk(
                classification_losses,
                len(predictions),
                B,
            )

            max_risk = torch.max(
                torch.stack(
                    [confidence_risk, localization_risk, classification_risk],
                ),
            )
            _log_raw_confidence_risk = self._correct_risk(
                _log_raw_confidence_losses,
                len(predictions),
                B,
            )
            _log_raw_localization_risk = self._correct_risk(
                _log_raw_localization_losses,
                len(predictions),
                B,
            )
            _log_raw_classification_risk = self._correct_risk(
                _log_raw_classification_losses,
                len(predictions),
                B,
            )
            _log_raw_max_risk = torch.max(
                torch.stack(
                    [
                        _log_raw_confidence_risk,
                        _log_raw_localization_risk,
                        _log_raw_classification_risk,
                    ],
                ),
            )
            self.all_lbds.append(lambda_conf)
            self.all_risks_raw.append(_log_raw_max_risk.detach().cpu().numpy())
            self.all_risks_raw_conf.append(
                _log_raw_confidence_risk.detach().cpu().numpy(),
            )
            self.all_risks_raw_loc.append(
                _log_raw_localization_risk.detach().cpu().numpy(),
            )
            self.all_risks_raw_cls.append(
                _log_raw_classification_risk.detach().cpu().numpy(),
            )

            self.all_risks_mon.append(max_risk.detach().cpu().numpy())
            self.all_risks_mon_conf.append(
                confidence_risk.detach().cpu().numpy()
                if isinstance(confidence_risk, torch.Tensor)
                else confidence_risk,
            )
            self.all_risks_mon_loc.append(
                localization_risk.detach().cpu().numpy()
                if isinstance(localization_risk, torch.Tensor)
                else localization_risk,
            )
            self.all_risks_mon_cls.append(
                classification_risk.detach().cpu().numpy()
                if isinstance(classification_risk, torch.Tensor)
                else classification_risk,
            )

            pbar.set_description(
                f"λ={lambda_conf}. Corrected Risk = {max_risk.detach().cpu().numpy():.4f}",
            )

            if max_risk.detach().cpu().numpy() > alpha:
                logger.info(
                    f"Solution Found: {previous_lbd} with risk {max_risk}",
                )

                print("--------------------------------------------------")
                print("Lambdas")
                print(f"\tprevious_lbd = {previous_lbd}")
                print(f"\tLast Lambda = {lambda_conf}")
                print(f"\tOther previous lbd = {self.all_lbds[-2]}")
                print(f"\tOther current lbd = {self.all_lbds[-1]}")
                print("All risks raw (precomputed):")
                confidence_risk_raw = self.all_risks_raw_conf[-2]
                localization_risk_raw = self.all_risks_raw_loc[-2]
                classification_risk_raw = self.all_risks_raw_cls[-2]
                max_risk_raw = self.all_risks_raw[-2]
                print(f"\tConfidence Risk: {confidence_risk_raw}")
                print(f"\tLocalization Risk: {localization_risk_raw}")
                print(f"\tClassification Risk: {classification_risk_raw}")
                print(f"\tMax Risk: {max_risk_raw}")
                print("All risks monotonized (precomputed):")
                confidence_risk_mon = self.all_risks_mon_conf[-2]
                localization_risk_mon = self.all_risks_mon_loc[-2]
                classification_risk_mon = self.all_risks_mon_cls[-2]
                max_risk_mon = self.all_risks_mon[-2]
                print(f"\tConfidence Risk: {confidence_risk_mon}")
                print(f"\tLocalization Risk: {localization_risk_mon}")
                print(f"\tClassification Risk: {classification_risk_mon}")
                print(f"\tMax Risk: {max_risk_mon}")
                print("Confidence risk (recomputed):")
                conf_losses = []
                for i in range(len(predictions)):
                    true_boxes_i = true_boxes[i]
                    pred_boxes_i = pred_boxes[i]
                    pred_cls_i = pred_cls[i]
                    confidences_i = confidences[i]
                    true_cls_i = true_cls[i]

                    matching_i = predictions.matching[i]

                    pred_boxes_i = pred_boxes_i[
                        confidences_i >= 1 - previous_lbd
                    ]
                    pred_cls_i = [
                        x
                        for x, c in zip(pred_cls_i, confidences_i)
                        if c >= 1 - previous_lbd
                    ]
                    confidence_loss_i = confidence_loss(
                        true_boxes_i,
                        true_cls_i,
                        pred_boxes_i,
                        pred_cls_i,
                    )

                    conf_losses.append(confidence_loss_i)
                conf_losses = torch.stack(conf_losses)
                print(f"\tConfidence Risk: {torch.mean(conf_losses)}")
                confidence_losses = torch.stack(confidence_losses)
                print("Comparison of the two :")
                print(
                    f"\t (isclose) {torch.isclose(conf_losses, confidence_losses).float().mean()}",
                )
                print(
                    f"\t (eq) {torch.eq(conf_losses, confidence_losses).float().mean()}",
                )
                # now get the indices of where the losses differ, and print the image id as well as the two losses, for about 20 images
                diff_indices = torch.where(
                    torch.ne(conf_losses, confidence_losses),
                )[0]
                for i in diff_indices[:10]:
                    print(
                        f"\tImage {i} loss: {conf_losses[i]} (eval) vs {confidence_losses[i]} (opti)",
                    )
                    print(
                        f"\tImage {i} confidence: {predictions.confidences[i]}",
                    )
                    # print number of ground truths
                    print(
                        f"\tImage {i} number of ground truths: {len(predictions.true_boxes[i])}",
                    )
                    print(
                        f"\tImage {i} number of predictions: {len(predictions.pred_boxes[i][predictions.confidences[i] >= 1 - previous_lbd])}",
                    )
                print("--------------------------------------------------")
                return previous_lbd
        return lambda_conf


class SecondStepMonotonizingOptimizer(Optimizer):
    def __init__(self):
        pass

    def _correct_risk(
        self,
        risk: torch.Tensor,
        n: int,
        B: float,
    ) -> torch.Tensor:
        """Correct the risk using the number of predictions and the upper bound.

        Parameters
        ----------
        - risk (torch.Tensor): The risk tensor.
        - n (int): The number of predictions.
        - B (float): The upper bound.

        Returns
        -------
        - corrected_risk (torch.Tensor): The corrected risk tensor.

        """
        return (n / (n + 1)) * risk + B / (n + 1)

    def evaluate_risk(
        self,
        lbd,
        loss,
        final_lbd_conf,
        predictions,
        build_predictions,
        matching_function,
    ):
        all_risks_raw = []
        all_risks_mon = []
        all_lbds_cnf = []
        all_lbds_loc = []

        true_boxes = predictions.true_boxes
        pred_boxes = predictions.pred_boxes
        true_cls = predictions.true_cls
        pred_cls = predictions.pred_cls
        confidences = predictions.confidences
        device = predictions.true_boxes[0].device

        stacked_confidences = torch.concatenate(confidences)
        confidence_image_idx = torch.concatenate(
            [
                torch.ones_like(x, dtype=torch.int) * i
                for i, x in enumerate(confidences)
            ],
        )

        sorted_stacked_confidences, indices = torch.sort(stacked_confidences)
        sorted_stacked_confidences[:-1] = sorted_stacked_confidences[
            1:
        ].clone()
        sorted_stacked_confidences[-1] = 1.0  # last one is always 1.0
        sorted_confidence_image_indices = confidence_image_idx[indices]

        lambda_conf = 1

        match_predictions_to_true_boxes(
            predictions,
            distance_function=matching_function,
            verbose=False,
            overload_confidence_threshold=1 - lambda_conf,
        )

        losses = []

        # TODO(leo): parallelize?
        # Step 1: Compute the risk
        for i in range(len(predictions)):
            true_boxes_i = true_boxes[i]
            pred_boxes_i = pred_boxes[i]
            true_cls_i = true_cls[i]
            pred_cls_i = pred_cls[i]
            confidences_i = confidences[i]
            matching_i = predictions.matching[i]

            pred_boxes_i = pred_boxes_i[confidences_i >= 1 - lambda_conf]
            pred_cls_i = [
                x
                for x, c in zip(pred_cls_i, confidences_i)
                if c >= 1 - lambda_conf
            ]
            pred_cls_i = (
                torch.stack(pred_cls_i)
                if len(pred_cls_i) > 0
                else torch.tensor([]).float().to(device)
            )

            tmp_matched_boxes_i = [
                (
                    torch.stack([pred_boxes_i[m] for m in matching_i[j]])[0]
                    if len(matching_i[j]) > 0
                    else torch.tensor([]).float().to(device)
                )
                for j in range(len(true_boxes_i))
            ]
            matched_pred_boxes_i = (
                torch.stack(tmp_matched_boxes_i)
                if len(tmp_matched_boxes_i) > 0
                else torch.tensor([]).float().to(device)
            )
            matched_pred_cls_i = [
                (
                    torch.stack([pred_cls_i[m] for m in matching_i[j]])[0]
                    if len(matching_i[j]) > 0
                    else torch.tensor([]).float().to(device)
                )
                for j in range(len(true_boxes_i))
            ]

            matched_conf_boxes_i, matched_conf_cls_i = build_predictions(
                matched_pred_boxes_i,
                matched_pred_cls_i,
                lbd,
            )

            loss_i = loss(
                true_boxes_i,
                true_cls_i,
                matched_conf_boxes_i,
                matched_conf_cls_i,
            )

            losses.append(loss_i)

        # ------- Start Logging -------
        _log_risk = torch.mean(torch.stack(losses))
        _log_losses = losses.copy()
        all_risks_raw.append(_log_risk.detach().cpu().numpy())
        all_risks_mon.append(_log_risk.detach().cpu().numpy())
        all_lbds_cnf.append(lambda_conf)
        all_lbds_loc.append(lbd)
        # ------- End Logging -------

        # Step 2: Update one loss at a time
        for image_id, conf_score in zip(
            sorted_confidence_image_indices,
            sorted_stacked_confidences,
        ):
            lambda_conf = 1 - conf_score.cpu().numpy().item()

            i = image_id
            true_boxes_i = true_boxes[i]
            pred_boxes_i = pred_boxes[i]
            true_cls_i = true_cls[i]
            pred_cls_i = pred_cls[i]
            confidences_i = confidences[i]

            matching_i = match_predictions_to_true_boxes(
                predictions,
                distance_function=matching_function,
                verbose=False,
                overload_confidence_threshold=1 - lambda_conf,
                idx=i,
            )

            predictions.matching[i] = matching_i

            pred_boxes_i = pred_boxes_i[confidences_i >= 1 - lambda_conf]
            pred_cls_i = [
                x
                for x, c in zip(pred_cls_i, confidences_i)
                if c >= 1 - lambda_conf
            ]
            pred_cls_i = (
                torch.stack(pred_cls_i)
                if len(pred_cls_i) > 0
                else torch.tensor([]).float().to(device)
            )

            # TODO: currently only support matching to a single box
            tmp_matched_boxes_i = [
                (
                    torch.stack([pred_boxes_i[m] for m in matching_i[j]])[0]
                    if len(matching_i[j]) > 0
                    else torch.tensor([]).float().to(device)
                )
                for j in range(len(true_boxes_i))
            ]
            matched_pred_boxes_i = (
                torch.stack(tmp_matched_boxes_i)
                if len(tmp_matched_boxes_i) > 0
                else torch.tensor([]).float().to(device)
            )
            matched_pred_cls_i = [
                (
                    torch.stack([pred_cls_i[m] for m in matching_i[j]])[0]
                    if len(matching_i[j]) > 0
                    else torch.tensor([]).float().to(device)
                )
                for j in range(len(true_boxes_i))
            ]
            matched_conf_boxes_i, matched_conf_cls_i = build_predictions(
                matched_pred_boxes_i,
                matched_pred_cls_i,
                lbd,
            )

            loss_i = loss(
                true_boxes_i,
                true_cls_i,
                matched_conf_boxes_i,
                matched_conf_cls_i,
            )

            _log_losses[i] = loss_i.clone()
            old_loss_i = losses[i].clone()
            loss_i = max(old_loss_i, loss_i)
            losses[i] = loss_i

            # ------- Start Logging -------
            _log_mon_risk = torch.mean(torch.stack(losses))
            _log_raw_risk = torch.mean(torch.stack(_log_losses))
            all_risks_raw.append(_log_raw_risk.detach().cpu().numpy())
            all_risks_mon.append(_log_mon_risk.detach().cpu().numpy())
            all_lbds_cnf.append(lambda_conf)
            all_lbds_loc.append(lbd)
            # ------- End Logging -------

            # Stopping condition: when we reached desired lbd_conf
            if final_lbd_conf >= lambda_conf:
                return (
                    torch.mean(torch.stack(losses)).detach().cpu().numpy(),
                    all_risks_raw,
                    all_risks_mon,
                    all_lbds_cnf,
                    all_lbds_loc,
                )
        return None, all_risks_raw, all_risks_mon, all_lbds_cnf, all_lbds_loc

    def optimize(
        self,
        predictions: ODPredictions,
        build_predictions: Callable,
        loss: ODLoss,
        matching_function: str,
        alpha: float,
        device: str,
        overload_confidence_threshold: float | None = None,
        B: float = 1,
        lower_bound: float = 0,
        upper_bound: float = 1,
        steps: int = 13,
        epsilon: float = 1e-10,
        *,
        verbose: bool = False,
    ):
        if overload_confidence_threshold is None:
            if predictions.confidence_threshold is None:
                raise ValueError(
                    "confidence_threshold must be set in the predictions or in the conformalizer",
                )
            confidence_threshold = predictions.confidence_threshold
            if isinstance(confidence_threshold, torch.Tensor):
                confidence_threshold = confidence_threshold.item()
            logger.info(
                f"Using predictions' confidence threshold: {confidence_threshold:.4f}",
            )
        else:
            logger.info(
                f"Using overload confidence threshold: {overload_confidence_threshold:.4f}",
            )
            confidence_threshold = overload_confidence_threshold

        ##### FINISH MODIFYING THE LOGGING OUTPUT AND COLLECTION

        self.all_risks_raw = []
        self.all_risks_mon = []
        self.all_lbds_cnf = []
        self.all_lbds_loc = []

        lambda_conf = 1 - confidence_threshold

        # Step 0: Initialize the risk

        left, right = lower_bound, upper_bound

        good_lbds = []

        pbar = tqdm(range(steps), disable=not verbose)

        for _ in pbar:
            lbd = (left + right) / 2
            # Evaluating the risk in this lbd, requires to remonotonize the loss in this lbd_loc/cls wrt the lbd_cnf
            risk, all_risks_raw, all_risks_mon, all_lbds_cnf, all_lbds_loc = (
                self.evaluate_risk(
                    lbd,
                    loss,
                    lambda_conf,
                    predictions,
                    build_predictions,
                    matching_function,
                )
            )
            self.all_risks_raw.append(all_risks_raw)
            self.all_risks_mon.append(all_risks_mon)
            self.all_lbds_cnf.append(all_lbds_cnf)
            self.all_lbds_loc.append(all_lbds_loc)

            corrected_risk = self._correct_risk(risk, len(predictions), B)

            corrected_risk = corrected_risk.item()  # detach().cpu().item()

            pbar.set_description(
                f"[{left:.2f}, {right:.2f}] -> λ={lbd}. Corrected Risk = {corrected_risk:.3f}",
            )

            if corrected_risk <= alpha:
                right = lbd
                good_lbds.append(lbd)
                if corrected_risk >= alpha - epsilon:
                    break
            else:  # corrected_risk > alpha
                left = lbd

        if len(good_lbds) == 0:
            raise ValueError("No good lambda found")

        return good_lbds[-1]
