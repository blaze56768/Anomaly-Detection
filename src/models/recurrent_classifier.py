"""
src.models.4ecurrent_classifier.py

This module uses a pretrain encoder combined with MLP to classify abnormalities.

.. _Google Python Style Guide:
   http://google.github.io/styleguide/pyguide.html

"""

__author__ = "Luis Felipe Villa Arenas"
__copyright__ = "Deutsche Telekom"


import pytorch_lightning as pl

import torch
import torch.nn as nn
import torchmetrics

from src.models.recurrent_autoencoder import RecurrentAutoEncoder


class ReplacementRecurrentClassifier(pl.LightningModule):
    """
    Recurrent Classifier.
    """
    def __init__(self,
                 lr: 0.001,
                 latent_dim: 500,
                 series_dim: 32,
                 features_dim: 184,
                 num_layers: 1,
                 weight_decay: 0.05,
                 linear_reduction: 0.1,
                 dropout_prob: 0.5,
                 output_dim: 1,
                 freeze_encoder: True,
                 name_labels_logger,
                 checkpoint: None,
                 **kwargs):
        """Init Recurrent classifier parameters.

        The deafult parameters can be set using the file config.model.conv-classifier.yaml

        Args:
            lr (float): learning rate.
            latent_dim (int): Latent dimmesion size.
            series_dim (int): Time series length.
            features_dim (int): Number of features.
            num_layers (int): Number of stacked recurrent layers.
            weight_decay (float): Weight decay
            linear_reduction (int): Ratio between the latent and inter linear layer.
            dropout_prob (float): Probability dropout layers.
            output_dim (int): Output dimmesion
            freeze_encoder (boolean): Flag to freeze teh weights of pretrain model.
            name_labels_logger (list[str]): label names used in logger eg. WandB.
            checkpoint (dir): Pretrain model's Checkpoint directory.
        """
        super(ReplacementRecurrentClassifier, self).__init__()

        # Hyperparameters
        self.save_hyperparameters()

        # Init the pretrained LightningModule
        autoencoder = RecurrentAutoEncoder(lr=self.hparams.lr,
                                           latent_dim=self.hparams.latent_dim,
                                           series_dim=self.hparams.series_dim,
                                           features_dim=self.hparams.features_dim,
                                           num_layers=self.hparams.num_layers,
                                           weight_decay=self.hparams.weight_decay)

        autoencoder.load_from_checkpoint(self.hparams.checkpoint)

        # Freeze Pretrain encoder
        if self.hparams.freeze_encoder:
            autoencoder.freeze()
        self.encoder = autoencoder.encoder

        # Init Layers
        self.dropout = nn.Dropout(p=self.hparams.dropout_prob)

        self.hparams.inter_dim = int(self.hparams.linear_reduction * self.hparams.latent_dim)

        self.input_linear = nn.Linear(self.hparams.latent_dim,
                                      self.hparams.inter_dim)

        self.output_linear = nn.Linear(self.hparams.latent_dim,
                                       self.hparams.output_dim)

        # Criterion
        self.criterion = nn.BCELoss()

    def forward(self, x):
        representations = self.encoder(x)
        #x = torch.relu((self.dropout(self.input_linear(representations))))
        x = torch.sigmoid((self.output_linear(self.dropout(representations))))
        return x

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(),
                                 lr=self.hparams.lr,
                                 weight_decay=self.hparams.weight_decay
                                 )

    def metrics_logger_custom(self, predictions, targets, prefix):
        prefix = prefix + '/'
        outputs = dict()
        for idx, l in enumerate(self.hparams.name_labels_logger):
            preds = predictions[:, idx]
            target = targets[:, idx].int()
            outputs[prefix + l + '/accuracy'] = torchmetrics.functional.accuracy(preds, target)
            outputs[prefix + l + '/precision'] = torchmetrics.functional.precision(preds, target)
            outputs[prefix + l + '/recall'] = torchmetrics.functional.recall(preds, target)
            outputs[prefix + l + '/f1'] = torchmetrics.functional.f1(preds, target)
        return outputs

    def training_step(self, batch, batch_idx):
        input, labels, weights = batch
        predictions = self(input)
        self.criterion.weight = weights
        loss = self.criterion(predictions, labels)
        outputs = self.metrics_logger_custom(predictions, labels, 'train')
        self.log('train/loss_step', loss)
        return {'loss': loss, 'metrics': outputs}

    def training_epoch_end(self, outputs):
        for _, metric in enumerate(outputs[0]['metrics'].keys()):
            self.log(metric, torch.tensor([x['metrics'][metric] for x in outputs]).mean())

    def validation_step(self, batch, batch_idx):
        input, labels, weights = batch
        predictions = self(input)
        self.criterion.weight = weights
        loss = self.criterion(predictions, labels)
        outputs = self.metrics_logger_custom(predictions, labels, 'val')
        outputs['val/loss'] = loss
        return outputs

    def validation_epoch_end(self, outputs):
        for _, metric in enumerate(outputs[0].keys()):
            self.log(metric, torch.tensor([x[metric] for x in outputs]).mean())

    def test_step(self, batch, batch_idx):
        input, labels, weights = batch
        predictions = self(input)
        outputs = self.metrics_logger_custom(predictions, labels, 'test')
        return outputs

    def test_epoch_end(self, outputs):
        for _, metric in enumerate(outputs[0].keys()):
            self.log(metric, torch.tensor([x[metric] for x in outputs]).mean())
