import torch
import torch.nn as nn
from torchvision.ops import box_iou


class ResizeChannels(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        self.num_channels = num_channels

    # for if 1 channel, repeat 3 times, if 3 channels, don't change the image
    def forward(self, image):
        if image.shape[0] == 1:
            return image.repeat(3, 1, 1)
        else:
            return image


def bayesod(
    pred_boxes: torch.Tensor,
    confidences: torch.Tensor,
    pred_cls: torch.Tensor,
    iou_threshold: float,
):
    """_summary_

    Args:
        pred_boxes (torch.Tensor): _description_
        confidences (torch.Tensor): _description_
        pred_cls (torch.Tensor): _description_
        iou_threshold (float): _description_
    """

    ious = box_iou(pred_boxes, pred_boxes)

    ious_overlap = ious > iou_threshold
    
    
