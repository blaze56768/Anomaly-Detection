"""
src.models.modules.autoencoders.convolutional.encoder.py

This module defines the convolutional encoder.

.. _Google Python Style Guide:
   http://google.github.io/styleguide/pyguide.html

"""
__author__ = "Luis Felipe Villa Arenas"
__copyright__ = "Deutsche Telekom"

import torch
import torch.nn as nn


class ConvEncoder(nn.Module):
    """ 1D Convolution autoencoder.
    """
    def __init__(self, hparams: dict):
        """Init  1-D ConvolutionalEncoder.
        Args:
            hparams (dict): Configuration parameters.
        """
        super(ConvEncoder, self).__init__()
        self.dims_expand = hparams["dims_expand"]
        self.flatten = nn.Flatten(start_dim=1)
        self.dropout = nn.Dropout(p=hparams["dropout_prob"])
        self.convblocks = self.blocks_init(hparams["features_dim"],
                                           hparams["num_conv_blocks"])

        self.linear_2 = nn.Linear(hparams["features_dim"] * hparams["series_dim"],
                                hparams["inter_dim"],
                                bias=True)
        self.linear = nn.Linear(hparams["inter_dim"],
                                hparams["latent_dim"],
                                bias=True)

        
        # weight init
        torch.nn.init.xavier_uniform(self.linear.weight)

    def blocks_init(self, features, layers):
        in_channels = [self.dims_expand[layer] * features for layer in range(layers)]
        out_channels = [self.dims_expand[layer + 1] * features for layer in range(layers)]
        return nn.ModuleList([ConvBlock(in_channels[i], out_channels[i]) for i in range(layers)])

    def forward(self, x):
        x = x.permute(0, 2, 1)

        for convblock in self.convblocks:
            x = convblock(x)

        x = self.flatten(x)
        x = self.dropout(x)
        x = self.linear_2(x)
        x = self.linear(x)
        return x


class ConvBlock(nn.Module):
    """Convolutional block.

    This module creates a 1D convolutional block with the following layers:
    - 1D convolution
    - 1D batchnorm
    - Relu No linearity
    - 1D maxpool
    """
    def __init__(self, in_channels, out_channels):
        """Init ConvBlock parameters.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
        """
        super(ConvBlock, self).__init__()
        self.conv1d = nn.Conv1d(in_channels=in_channels,
                                out_channels=out_channels,
                                kernel_size=3,
                                stride=1,
                                padding=1)
        torch.nn.init.xavier_uniform(self.conv1d.weight)
        self.batchnorm1d = nn.BatchNorm1d(out_channels)
        self.maxpool1d = nn.MaxPool1d(2, stride=2)

    def forward(self, x):
        x = self.conv1d(x)
        x = self.batchnorm1d(x)
        x = torch.relu(x)
        x = self.maxpool1d(x)
        return x
