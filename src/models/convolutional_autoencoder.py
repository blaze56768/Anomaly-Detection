"""src.models.convolutional_autoencoder.py

This module defines the convolutional autoencoder.

.. _Google Python Style Guide:
   http://google.github.io/styleguide/pyguide.html

"""

__author__ = "Luis Felipe Villa Arenas"
__copyright__ = "Deutsche Telekom"


import torch
import torch.nn as nn
import pytorch_lightning as pl

from src.models.modules.autoencoders.convolutional.encoder import ConvEncoder
from src.models.modules.autoencoders.convolutional.decoder import ConvDecoder


class ConvNetAutoEncoder(pl.LightningModule):
    """
    Convolutional Autoencoder.
    """
    def __init__(self,
                 lr = 0.001,
                 latent_dim= 1024,
                 series_dim= 32,
                 features_dim= 160,
                 num_category= 54,
                 num_numerical= 106,
                 num_conv_blocks= 2,
                 dims_expand= list([1, 2, 4, 8, 16]),
                 dropout_prob= 0.1,
                 weight_decay= 0.05,
                 **kwargs):
        """Init Autoencoder parameters.

        The deafult parametera can be set using the file config.model.conv-autoenc.yaml

        Args:
            lr (float): learning rate.
            latent_dim (int): Latent dimmesion size.
            series_dim (int): Time series length.
            features_dim (int): Number of features.
            num_category (int): Number of categorical features,
            num_numerical (int): Number of numerical features, 
            num_conv_blocks (int): Number of convolutional layers.
            dims_expand (list[int]): List of available dimession expansion.
            dropout_prob (float): Probability dropout layers.
            weight_decay (float): Weight decay

        """
        super(ConvNetAutoEncoder, self).__init__()

        self.save_hyperparameters()

        # model
        self.encoder = ConvEncoder(self.hparams)
        self.decoder = ConvDecoder(self.hparams)

        # loss function
        self.criterion_mse = torch.nn.MSELoss()

    def forward(self, x):
        embeddings = self.encoder(x)
        return self.decoder(embeddings)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(),
                                 lr=self.hparams.lr,
                                 weight_decay=self.hparams.weight_decay)
    def split(self, x):
        return torch.split(x, [self.hparams.num_numerical, self.hparams.num_category], dim=2)

    def reconstruction_loss(self, input, reconstruction):
        loss = self.criterion_mse(input, reconstruction)
        # Numerical and Categorical loss for logging
        input_recover_numerical, input_recover_categorical = self.split(reconstruction)
        input_numerical, input_categorical = self.split(input)
        categorical_loss = self.criterion_mse(input_categorical,input_recover_categorical)
        numerical_loss = self.criterion_mse(input_numerical, input_recover_numerical)
        return loss, numerical_loss, categorical_loss
    
    def step(self, batch):
        input, labels = batch
        # Autoencoder
        representations = self.encoder(input)
        reconstruction = self.decoder(representations)
        # Reconstruction loss
        loss, numerical_loss, categorical_loss = self.reconstruction_loss(input, reconstruction)
        return loss, numerical_loss, categorical_loss
    
    def training_step(self, batch, batch_idx):
        loss, numerical_loss, categorical_loss = self.step(batch)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train/numerical_loss", numerical_loss)
        self.log("train/categorical_loss", categorical_loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, numerical_loss, categorical_loss = self.step(batch)
        return {"loss": loss, "numerical_loss": numerical_loss, "categorical_loss": categorical_loss}

    def validation_epoch_end(self, outputs):
        avg_val_loss = torch.tensor([x["loss"] for x in outputs]).mean()
        avg_val_numerical_loss = torch.tensor([x["numerical_loss"] for x in outputs]).mean()
        avg_val_categorical_loss = torch.tensor([x["categorical_loss"] for x in outputs]).mean()
        self.log("val/loss", avg_val_loss)
        self.log("val/numerical_loss", avg_val_numerical_loss)
        self.log("val/categorical_loss", avg_val_categorical_loss)

    def test_step(self, batch, batch_idx):
        loss, numerical_loss, categorical_loss = self.step(batch)
        return {"loss": loss, "numerical_loss": numerical_loss, "categorical_loss": categorical_loss}

    def test_epoch_end(self, outputs):
        avg_test_loss = torch.tensor([x["loss"] for x in outputs]).mean()
        avg_test_numerical_loss = torch.tensor([x["numerical_loss"] for x in outputs]).mean()
        avg_test_categorical_loss = torch.tensor([x["categorical_loss"] for x in outputs]).mean()
        self.log("test/loss", avg_test_loss)
        self.log("test/numerical_loss", avg_test_numerical_loss)
        self.log("test/categorical_loss", avg_test_categorical_loss)
