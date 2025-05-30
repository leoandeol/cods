import numpy as np
import torch

from cods.base.loss import NCScore


class ObjectnessNCScore(NCScore):
    """ObjectnessNCScore is a class that calculates the score for objectness prediction.

    Args:
        kwargs: Additional keyword arguments.

    Attributes:
        None

    Methods:
        __call__(self, n_gt, confidence): Calculates the score based on the number of ground truth objects and confidence values.

    """

    def __init__(self, **kwargs):
        super().__init__()

    def __call__(self, n_gt: int, confidence: torch.Tensor) -> torch.Tensor:
        """Calculates the score based on the number of ground truth objects and confidence values.

        Args:
            n_gt (int): Number of ground truth objects.
            confidence (torch.Tensor): Confidence values.

        Returns:
            torch.Tensor: The calculated score.

        """
        sorted_values, indices = torch.sort(confidence)
        return 1 - sorted_values[-n_gt]


class ODNCScore(NCScore):
    """ODNCScore is an abstract class for calculating the score in object detection tasks.

    Args:
        kwargs: Additional keyword arguments.

    Attributes:
        None

    Methods:
        __call__(self, pred_boxes, true_box, **kwargs): Calculates the score based on predicted boxes and true box.
        get_set(self, pred_boxes, quantile): Returns the set of boxes based on predicted boxes and quantile.
        apply_margins(self, pred_boxes): Applies margins to the predicted boxes.

    """

    def __init__(self, **kwargs):
        super().__init__()

    def __call__(
        self,
        pred_boxes: torch.Tensor,
        true_box: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """Calculates the score based on predicted boxes and true box.

        Args:
            pred_boxes (torch.Tensor): Predicted boxes.
            true_box (torch.Tensor): True box.

        Returns:
            torch.Tensor: The calculated score.

        """
        raise NotImplementedError("ODNCScore is an abstract class.")

    def get_set(
        self,
        pred_boxes: torch.Tensor,
        quantile: float,
    ) -> torch.Tensor:
        """Returns the set of boxes based on predicted boxes and quantile.

        Args:
            pred_boxes (torch.Tensor): Predicted boxes.
            quantile (float): Quantile value.

        Returns:
            torch.Tensor: The set of boxes.

        """
        return self.apply_margins(pred_boxes, quantile)

    def apply_margins(self, pred_boxes: torch.Tensor) -> torch.Tensor:
        """Applies margins to the predicted boxes.

        Args:
            pred_boxes (torch.Tensor): Predicted boxes.

        Returns:
            torch.Tensor: The predicted boxes with applied margins.

        """
        raise NotImplementedError("ODNCScore is an abstract class.")


class MinAdditiveSignedAssymetricHausdorffNCScore(ODNCScore):
    """MinAdditiveSignedAssymetricHausdorffNCScore is a class that calculates the score using the minimum additive signed asymmetric Hausdorff distance.

    Args:
        image_shape (torch.Tensor, optional): The shape of the image. Defaults to None.

    Attributes:
        image_shape (torch.Tensor): The shape of the image.

    Methods:
        __call__(self, pred_boxes, true_box): Calculates the score based on predicted boxes and true box.
        apply_margins(self, pred_boxes, quantile): Applies margins to the predicted boxes based on quantile.

    """

    def __init__(self, image_shape: torch.Tensor):
        super().__init__()
        self.image_shape = image_shape

    def __call__(
        self,
        pred_boxes: torch.Tensor,
        true_box: torch.Tensor,
    ) -> torch.Tensor:
        """Calculates the score based on predicted boxes and true box.

        Args:
            pred_boxes (torch.Tensor): Predicted boxes.
            true_box (torch.Tensor): True box.

        Returns:
            torch.Tensor: The calculated score.

        """
        device = pred_boxes.device
        matching_scores = []
        scores = []
        if len(pred_boxes) == 0:
            print(
                "Warning: no predicted boxes found. It should not happen too often",
            )
        for pred_box in pred_boxes:
            score_x1 = pred_box[0] - true_box[0]
            score_y1 = pred_box[1] - true_box[1]
            score_x2 = true_box[2] - pred_box[2]
            score_y2 = true_box[3] - pred_box[3]
            score = torch.stack(
                (score_x1, score_y1, score_x2, score_y2),
                axis=-1,
            )
            scores.append(score)
            max_score = torch.max(score).item()
            matching_scores.append(max_score)
        if len(matching_scores) == 0:
            print(
                "Warning: no matching boxes found. It should not happen too often",
            )
            return torch.ones(4).to(device) * self.image_shape
        idx = np.argmin(matching_scores).item()
        return scores[idx]

    def apply_margins(
        self,
        pred_boxes: torch.Tensor,
        quantile: float,
    ) -> torch.Tensor:
        """Applies margins to the predicted boxes based on quantile.

        Args:
            pred_boxes (torch.Tensor): Predicted boxes.
            quantile (float): Quantile value.

        Returns:
            torch.Tensor: The predicted boxes with applied margins.

        """
        device = pred_boxes.device
        n = len(pred_boxes)
        new_boxes = [None] * n
        Qst = torch.FloatTensor([quantile]).to(device)
        for i in range(n):
            new_boxes[i] = pred_boxes[i] + torch.mul(
                torch.FloatTensor([[-1, -1, 1, 1]]).to(device),
                Qst,
            )
        return new_boxes


class UnionAdditiveSignedAssymetricHausdorffNCScore(ODNCScore):
    """UnionAdditiveSignedAssymetricHausdorffNCScore is a class that calculates the score using the union additive signed asymmetric Hausdorff distance.

    Args:
        None

    Attributes:
        None

    Methods:
        apply_margins(self, pred_boxes, quantile): Applies margins to the predicted boxes based on quantile.

    """

    def __init__(self):
        super().__init__()

    def __call__(self, pred_boxes: torch.Tensor, true_box: torch.Tensor):
        """Calculates the score based on predicted boxes and true box.

        Args:
            pred_boxes (torch.Tensor): Predicted boxes.
            true_box (torch.Tensor): True box.

        Raises:
            NotImplementedError: This method is not implemented yet. Use Min instead.

        """
        raise NotImplementedError("Not implemented yet. Use Min instead.")

    def apply_margins(
        self,
        pred_boxes: torch.Tensor,
        quantile: float,
    ) -> torch.Tensor:
        """Applies margins to the predicted boxes based on quantile.

        Args:
            pred_boxes (torch.Tensor): Predicted boxes.
            quantile (float): Quantile value.

        Returns:
            torch.Tensor: The predicted boxes with applied margins.

        """
        device = pred_boxes.device
        n = len(pred_boxes)
        new_boxes = [None] * n
        Qst = torch.FloatTensor([quantile]).to(device)
        for i in range(n):
            new_boxes[i] = pred_boxes[i] + torch.mul(
                torch.FloatTensor([[-1, -1, 1, 1]]).to(device),
                Qst,
            )
        return new_boxes


class MinMultiplicativeSignedAssymetricHausdorffNCScore(ODNCScore):
    """MinMultiplicativeSignedAssymetricHausdorffNCScore is a class that calculates the score using the minimum multiplicative signed asymmetric Hausdorff distance.

    Args:
        None

    Attributes:
        None

    Methods:
        __call__(self, pred_boxes, true_box): Calculates the score based on predicted boxes and true box.
        apply_margins(self, pred_boxes, quantile): Applies margins to the predicted boxes based on quantile.

    """

    def __init__(self):
        super().__init__()

    def __call__(
        self,
        pred_boxes: torch.Tensor,
        true_box: torch.Tensor,
    ) -> torch.Tensor:
        """Calculates the score based on predicted boxes and true box.

        Args:
            pred_boxes (torch.Tensor): Predicted boxes.
            true_box (torch.Tensor): True box.

        Returns:
            torch.Tensor: The calculated score.

        """
        areas = []
        scores = []
        for pred_box in pred_boxes:
            w = pred_box[2] - pred_box[0]
            h = pred_box[3] - pred_box[1]
            score_x1 = (pred_box[0] - true_box[0]) / w
            score_y1 = (pred_box[1] - true_box[1]) / h
            score_x2 = (true_box[2] - pred_box[2]) / w
            score_y2 = (true_box[3] - pred_box[3]) / h
            scores.append(
                torch.stack((score_x1, score_y1, score_x2, score_y2), axis=-1),
            )
            areas.append(torch.max(scores[-1]).item())
        if len(areas) == 0:
            return torch.ones(4) * float("inf")
        idx = np.argmin(areas).item()
        return scores[idx]

    def apply_margins(
        self,
        pred_boxes: torch.Tensor,
        quantile: float,
    ) -> torch.Tensor:
        """Applies margins to the predicted boxes based on quantile.

        Args:
            pred_boxes (torch.Tensor): Predicted boxes.
            quantile (float): Quantile value.

        Returns:
            torch.Tensor: The predicted boxes with applied margins.

        """
        device = pred_boxes.device
        n = len(pred_boxes)
        new_boxes = [None] * n
        Qst = torch.FloatTensor([quantile]).to(device)
        for i in range(n):
            w = pred_boxes[i][:, 2] - pred_boxes[i][:, 0]
            h = pred_boxes[i][:, 3] - pred_boxes[i][:, 1]
            new_boxes[i] = pred_boxes[i] + torch.mul(
                torch.stack((-w, -h, w, h), axis=-1),
                Qst,
            )
        return new_boxes


class UnionMultiplicativeSignedAssymetricHausdorffNCScore(ODNCScore):
    """UnionMultiplicativeSignedAssymetricHausdorffNCScore is a class that calculates the score using the union multiplicative signed asymmetric Hausdorff distance.

    Args:
        None

    Attributes:
        None

    Methods:
        apply_margins(self, pred_boxes, quantile): Applies margins to the predicted boxes based on quantile.

    """

    def __init__(self):
        super().__init__()

    def __call__(self, pred_boxes: torch.Tensor, true_box: torch.Tensor):
        """Calculates the score based on predicted boxes and true box.

        Args:
            pred_boxes (torch.Tensor): Predicted boxes.
            true_box (torch.Tensor): True box.

        Raises:
            NotImplementedError: This method is not implemented yet. Use Min instead.

        """
        raise NotImplementedError("Not implemented yet. Use Min instead.")

    def apply_margins(
        self,
        pred_boxes: torch.Tensor,
        quantile: float,
    ) -> torch.Tensor:
        """Applies margins to the predicted boxes based on quantile.

        Args:
            pred_boxes (torch.Tensor): Predicted boxes.
            quantile (float): Quantile value.

        Returns:
            torch.Tensor: The predicted boxes with applied margins.

        """
        device = pred_boxes.device
        n = len(pred_boxes)
        new_boxes = [None] * n
        Qst = torch.FloatTensor([quantile]).to(device)
        for i in range(n):
            w = pred_boxes[i][:, 2] - pred_boxes[i][:, 0]
            h = pred_boxes[i][:, 3] - pred_boxes[i][:, 1]
            new_boxes[i] = pred_boxes[i] + torch.mul(
                torch.stack((-w, -h, w, h), axis=-1),
                Qst,
            )
        return new_boxes
