from logging import getLogger
from typing import List

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
        bounds: List[float] = [0, 1],  # deprecated
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
            [x for x in confidences],  # .squeeze()
        )
        confidence_image_idx = torch.concatenate(
            [
                torch.ones_like(x, dtype=int) * i
                for i, x in enumerate(confidences)
            ],
        )

        sorted_stacked_confidences, indices = torch.sort(stacked_confidences)
        sorted_confidence_image_indices = confidence_image_idx[indices]
        sorted_confidence_image_indices[1:] = sorted_confidence_image_indices[
            0:-1
        ].clone()
        # We let the first be, it should occur no change anyways ?
        # sorted_confidence_image_indices[0] = ???

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
        for i in tqdm(range(len(predictions))):
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
            # print(matched_pred_boxes_i.shape)
            matched_pred_cls_i = list(
                [
                    (
                        torch.stack([pred_cls_i[m] for m in matching_i[j]])[
                            0
                        ]  # TODO zero here ?
                        if len(matching_i[j]) > 0
                        else torch.tensor([]).float().to(device)
                    )
                    for j in range(len(true_boxes_i))
                ],
            )
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

            # if matched_conf_boxes_i.size() == 0:
            #     matched_conf_boxes_i = torch.tensor([]).float().to(device)

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

        # confidence_risk = torch.mean(torch.stack(confidence_losses))
        # localization_risk = torch.mean(torch.stack(localization_losses))
        # classification_risk = torch.mean(torch.stack(classification_losses))

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

        max_risk = torch.max(
            torch.stack(
                [confidence_risk, localization_risk, classification_risk],
            ),
        )
        print(f"First risk: {max_risk.detach().cpu().numpy()}")
        if max_risk.detach().cpu().numpy() > alpha:
            # Debug: all three risks to see why there isn't any solution
            logger.debug(f"Confidence risk: {confidence_risk}")
            logger.debug(f"Localization risk: {localization_risk}")
            logger.debug(f"Classification risk: {classification_risk}")
            logger.debug(f"Max risk: {max_risk} > {alpha}. No solution found.")
            raise ValueError(
                "There does not exist any solution satisfying the constraints.",
            )
        logger.debug(
            f"Risk after 1st epoch is {max_risk.detach().cpu().numpy()} < {alpha}",
        )

        previous_lbd = lambda_conf
        previous_risk = max_risk

        pbar = tqdm(
            list(
                zip(
                    sorted_confidence_image_indices,
                    sorted_stacked_confidences,
                ),
            ),
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
            previous_risk = max_risk
            lambda_conf = 1 - conf_score.cpu().numpy()  # - 1e-7  # Test
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

            # if i in [14, 74, 199, 213, 225, 234]:
            #     print("--------------------------------------------------")
            #     print(f"Image {i}")
            #     print(
            #         f"Confidence Loss: {confidence_loss_i.detach().cpu().numpy()}"
            #     )  # .tolist:.4f}")
            #     print(f"Number of ground truths: {len(true_boxes_i)}")
            #     print(f"Number of predictions: {len(pred_boxes_i)}")
            #     print(f"Confidences ({confidences_i.dtype}):")
            #     for c in confidences_i:
            #         print(f"\t{c}")
            #     print(f"Lambda and confidence: {lambda_conf} and {conf_score}")

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
            matched_pred_cls_i = list(
                [
                    (
                        torch.stack([pred_cls_i[m] for m in matching_i[j]])[0]
                        if len(matching_i[j]) > 0
                        else torch.tensor([]).float().to(device)
                    )
                    for j in range(len(true_boxes_i))
                ],
            )

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
            # print(
            #     true_boxes_i.shape,
            #     true_cls_i.shape,
            #     matched_conf_boxes_i[0].shape,
            #     matched_conf_cls_i[0].shape,
            # )
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

            # if confidence_loss_i.detach().cpu().numpy()[0] != confidence_losses[i].detach().cpu().numpy()[0]:
            #     print(f"Confidence Loss: {confidence_loss_i.detach().cpu().numpy()}")#.tolist:.4f}")
            #     print(f"Localization Loss: {localization_loss_i.detach().cpu().numpy()}")#:.4f}")
            #     print(f"Classification Loss: {classification_loss_i.detach().cpu().numpy()}")#:.4f}")

            confidence_losses[i] = confidence_loss_i.detach().clone()
            localization_losses[i] = localization_loss_i.detach().clone()
            classification_losses[i] = classification_loss_i.detach().clone()

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
            # idddd = torch.argmax(
            #     torch.stack(
            #         [confidence_risk, localization_risk, classification_risk]
            #     )
            # )
            # logger.info(f"maximizing risk : {idddd} where 0 is confidence, 1 is localization and 2 is classification")

            self.all_lbds.append(lambda_conf)
            # _tmp_max_risk =
            self.all_risks_raw.append(max_risk.detach().cpu().numpy())
            self.all_risks_raw_conf.append(
                confidence_risk.detach().cpu().numpy(),
            )
            self.all_risks_raw_loc.append(
                localization_risk.detach().cpu().numpy(),
            )
            self.all_risks_raw_cls.append(
                classification_risk.detach().cpu().numpy(),
            )

            # Monotonization
            if max_risk < previous_risk:
                max_risk = previous_risk
                confidence_risk = np.max(
                    [
                        self.all_risks_mon_conf[-1],
                        confidence_risk.detach().cpu().numpy(),
                    ],
                )
                localization_risk = np.max(
                    [
                        self.all_risks_mon_loc[-1],
                        localization_risk.detach().cpu().numpy(),
                    ],
                )
                classification_risk = np.max(
                    [
                        self.all_risks_mon_cls[-1],
                        classification_risk.detach().cpu().numpy(),
                    ],
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
        true_boxes = predictions.true_boxes
        pred_boxes = predictions.pred_boxes
        true_cls = predictions.true_cls
        pred_cls = predictions.pred_cls
        image_shapes = predictions.image_shapes
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
        sorted_confidence_image_indices = confidence_image_idx[indices]
        sorted_confidence_image_indices[1:] = sorted_confidence_image_indices[
            0:-1
        ].clone()
        # We let the first be, it should occur no change anyways ?
        # sorted_confidence_image_indices[0] = ???

        lambda_conf = 1

        match_predictions_to_true_boxes(
            predictions,
            distance_function=matching_function,
            verbose=False,
            overload_confidence_threshold=1 - lambda_conf,
        )

        losses = []

        # TODO(leo):parallelize?
        # Step 1: Compute the risk
        for i in range(len(predictions)):
            true_boxes_i = true_boxes[i]
            pred_boxes_i = pred_boxes[i]
            true_cls_i = true_cls[i]
            pred_cls_i = pred_cls[i]
            matching_i = predictions.matching[i]

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
            # print(matched_pred_boxes_i.shape)
            matched_pred_cls_i = list(
                [
                    (
                        torch.stack([pred_cls_i[m] for m in matching_i[j]])[0]
                        if len(matching_i[j]) > 0
                        else torch.tensor([]).float().to(device)
                    )
                    for j in range(len(true_boxes_i))
                ],
            )
            # if len(matched_pred_boxes_i.shape)==1:
            #     matched_pred_boxes_i = matched_pred_boxes_i[None,...]
            # if len(matched_pred_cls_i.shape)==1:
            #     matched_pred_cls_i = matched_pred_cls_i[None,...]

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

        risk = torch.mean(torch.stack(losses))

        n_losses = len(losses)

        # Step 2: Update one loss at a time
        for image_id, conf_score in zip(
            sorted_confidence_image_indices,
            sorted_stacked_confidences,
        ):
            previous_risk = risk
            lambda_conf = 1 - conf_score.cpu().numpy()

            i = image_id
            true_boxes_i = true_boxes[i]
            pred_boxes_i = pred_boxes[i]
            true_cls_i = true_cls[i]
            pred_cls_i = pred_cls[i]

            matching_i = match_predictions_to_true_boxes(
                predictions,
                distance_function=matching_function,
                verbose=False,
                overload_confidence_threshold=1 - lambda_conf,
                idx=i,
            )

            predictions.matching[i] = matching_i

            # TODO: currently only support matching to a single box
            tmp_matched_boxes_i = [
                (
                    torch.stack([pred_boxes_i[m] for m in matching_i[j]])[0]
                    if len(matching_i[j]) > 0
                    else torch.tensor([]).float().to(device)
                )
                for j in range(len(true_boxes_i))
            ]
            # print([x.shape for x in tmp_matched_boxes_i])
            matched_pred_boxes_i = (
                torch.stack(tmp_matched_boxes_i)
                if len(tmp_matched_boxes_i) > 0
                else torch.tensor([]).float().to(device)
            )
            matched_pred_cls_i = list(
                [
                    (
                        torch.stack([pred_cls_i[m] for m in matching_i[j]])[0]
                        if len(matching_i[j]) > 0
                        else torch.tensor([]).float().to(device)
                    )
                    for j in range(len(true_boxes_i))
                ],
            )

            # print(matched_pred_boxes_i.shape)
            # print(matched_pred_boxes_i)

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

            old_loss_i = losses[i]
            losses[i] = loss_i

            # risk = torch.mean(torch.stack(losses))
            # Faster
            risk = risk + (loss_i - old_loss_i) / n_losses

            self.all_risks_raw.append(risk.detach().cpu().numpy())

            risk = max(risk, previous_risk)

            self.all_risks_mon.append(risk.detach().cpu().numpy())

            self.all_lbds_cnf.append(lambda_conf)
            self.all_lbds_loc.append(lbd)

            # Stopping condition: when we reached desired lbd_conf
            if final_lbd_conf >= lambda_conf:
                return risk

    def optimize(
        self,
        predictions: ODPredictions,
        build_predictions,
        loss: ODLoss,
        matching_function,
        alpha: float,
        device: str,
        B: float = 1,
        bounds: List[float] = [0, 1],
        steps=13,
        epsilon=1e-10,
        verbose: bool = False,
    ):
        self.all_risks_raw = []
        self.all_risks_mon = []
        self.all_lbds_cnf = []
        self.all_lbds_loc = []

        lambda_conf = 1 - predictions.confidence_threshold

        # Step 0: Initialize the risk

        match_predictions_to_true_boxes(
            predictions,
            distance_function=matching_function,
            verbose=False,
            overload_confidence_threshold=1 - lambda_conf,
        )

        left, right = bounds
        lbd = (left + right) / 2

        good_lbds = []

        pbar = tqdm(range(steps), disable=not verbose)

        for step in pbar:
            # Evaluating the risk in this lbd, requires to remonotonize the loss in this lbd_loc/cls wrt the lbd_cnf
            risk = self.evaluate_risk(
                lbd,
                loss,
                lambda_conf,
                predictions,
                build_predictions,
                matching_function,
            )

            corrected_risk = self._correct_risk(risk, len(predictions), B)

            corrected_risk = corrected_risk.detach().cpu().numpy().item()

            pbar.set_description(
                f"[{left:.2f}, {right:.2f}] -> λ={lbd}. Corrected Risk = {corrected_risk:.2f}",
            )

            if risk <= alpha:
                good_lbds.append(lbd)
                if risk >= alpha - epsilon:
                    break

            if corrected_risk > alpha:
                left = lbd
            else:
                right = lbd
            lbd = (left + right) / 2

        if len(good_lbds) == 0:
            raise ValueError("No good lambda found")

        return good_lbds[-1]
