"""
src.models.modules.autoencoders.convolutional.decoder

This module defines the convolutional decoder.

.. _Google Python Style Guide:
   http://google.github.io/styleguide/pyguide.html

"""
__author__ = "Luis Felipe Villa Arenas"
__copyright__ = "Deutsche Telekom"

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvDecoder(nn.Module):
    """ 1D Convolution autoencoder.
    """
    def __init__(self, hparams: dict):
        """Init  1-D ConvolutionalEncoder.
        Args:
            hparams (dict): Configuration parameters.
            weight (tensor): Embeding weight.
        """
        super(ConvDecoder, self).__init__()
        self.dims_expand = hparams["dims_expand"]
        self.features_dim = hparams["features_dim"]
        self.latent_dim = hparams["latent_dim"]
        self.num_conv_blocks = hparams["num_conv_blocks"]

        self.inverse_convblocks = self.blocks_init(hparams["features_dim"],
                                                   hparams["num_conv_blocks"])

        self.linear_2 = nn.Linear(hparams["latent_dim"],
                                hparams["inter_dim"],
                                bias=True)

        self.linear = nn.Linear(hparams["inter_dim"],
                                hparams["features_dim"] * hparams["series_dim"],
                                bias=True)

    def blocks_init(self, features, layers):
        in_channels = [self.dims_expand[layer + 1] * features for layer in reversed(range(layers))]
        out_channels = [self.dims_expand[layer] * features for layer in reversed(range(layers))]
        blocks_list = list()
        for i in range(layers):
            if i + 1 == layers:
                last_block = True
            else:
                last_block = False
            blocks_list.append(InverseConvBlock(in_channels[i],
                                                out_channels[i],
                                                last_block))
        return nn.ModuleList(blocks_list)

    def forward(self, x):
        x = self.linear_2(x)
        x = self.linear(x)
        x = x.view(x.shape[0], self.dims_expand[self.num_conv_blocks] * self.features_dim, -1)

        for inverse_convblock in self.inverse_convblocks:
            x = inverse_convblock(x)

        x = x.permute(0, 2, 1)
        return x


class InverseConvBlock(nn.Module):
    """Inverse convolutional block.

    This module creates a 1D convolutional block with the following layers:
    - 1D inverse convolution
    - 1D batchnorm
    - Relu No linearity
    """
    def __init__(self, in_channels, out_channels, last_block=False):
        """Init ConvBlock parameters.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            last_block (boolean): Flag to identify last block.
        """
        super(InverseConvBlock, self).__init__()
        self.conv1d_trans = nn.ConvTranspose1d(in_channels=in_channels,
                                               out_channels=out_channels,
                                               kernel_size=2,
                                               stride=2)
        torch.nn.init.xavier_uniform(self.conv1d_trans.weight)
        self.last_block = last_block
        if self.last_block is False:
            self.batchnorm1d = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        x = self.conv1d_trans(x)
        if self.last_block is False:
            x = self.batchnorm1d(x)
            x = torch.relu(x)
        else:
            x = torch.tanh(x)
        return x
