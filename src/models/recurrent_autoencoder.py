"""
src.models.recurrent_autoencoder.py

This module defines the recurrent autoencoder.

.. _Google Python Style Guide:
   http://google.github.io/styleguide/pyguide.html

"""
__author__ = "Luis Felipe Villa Arenas"
__copyright__ = "Deutsche Telekom"

import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl

from src.models.modules.autoencoders.recurrent.encoder import RecurrentEncoder
from src.models.modules.autoencoders.recurrent.decoder import RecurrentDecoder


class RecurrentAutoEncoder(pl.LightningModule):
    """ Recurrent Autoencoder.
    """
    def __init__(self,
                 lr: 0.001,
                 latent_dim: 500,
                 series_dim: 32,
                 features_dim: 184,
                 num_layers: 1,
                 weight_decay: 0.05,):
        """Init  RecurrentAutoencoder.

        The deafult parameters can be set using the file config.model.rnn-autoencoder.yaml

        Args:
            lr (float): learning rate.
            latent_dim (int): Latent dimmesion size.
            series_dim (int): Time series length.
            features_dim (int): Number of features.
            num_layers (int): Number of stack RNN layers.
            weight_decay (float): Weight decay
        """
        super(RecurrentAutoEncoder, self).__init__()

        self.save_hyperparameters()
        self.encoder = RecurrentEncoder(self.hparams)
        self.decoder = RecurrentDecoder(self.hparams)

        # loss function
        self.criterion = nn.MSELoss()

    def forward(self, x):
        embedding = self.encoder(x)
        output = self.decoder(embedding)
        return output

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(),
                                 lr=self.hparams.lr,
                                 weight_decay=self.hparams.weight_decay
                                 )

    def step(self, batch):
        input, labels, weights = batch
        input_hat = self(input)
        loss = self.criterion(input, input_hat)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.step(batch)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.step(batch)
        return {"loss": loss}

    def validation_epoch_end(self, outputs):
        avg_val_loss = torch.tensor([x["loss"] for x in outputs]).mean()
        self.log("val/loss", avg_val_loss)

    def test_step(self, batch, batch_idx):
        loss = self.step(batch)
        return {"loss": loss}

    def test_epoch_end(self, outputs):
        avg_test_loss = torch.tensor([x["loss"] for x in outputs]).mean()
        self.log("test/loss", avg_test_loss)
