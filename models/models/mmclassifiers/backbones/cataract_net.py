from mmcls.models.builder import BACKBONES
import torch.nn as nn


def default_unet_features():
    nb_features = [[16, 32, 32, 32], [32, 32, 32, 32, 32, 16, 16]]  # encoder  # decoder
    return nb_features


class ConvBlock(nn.Module):
    """
    Specific convolutional block followed by relu for unet.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3)
        self.activation = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        out = self.conv(x)
        out = self.activation(out)
        out = self.pool(out)
        return out

@BACKBONES.register_module()
class CataractNet(nn.Module):
    """
    A unet architecture. Layer features can be specified directly as a list of encoder and decoder
    features or as a single integer along with a number of unet levels. The default network features
    per layer (when no options are specified) are:

        encoder: [16, 32, 32, 32]
        decoder: [32, 32, 32, 32, 32, 16, 16]
    """

    arch = [32, 32, 64, 128]

    def __init__(
        self,
        in_channels=3,
    ):
        super().__init__()

        self.layers = nn.ModuleList()

        for num_features in self.arch:
            self.layers.append(ConvBlock(in_channels, num_features))
            in_channels = num_features

    def forward(self, x):

        for layer in self.layers:
            x = layer(x)

        return x.flatten(1)
