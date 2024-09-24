import torch.nn as nn


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
