"""
src.models.transformer_classifier.py

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

from src.models.transformer_autoencoder import TransformerAutoEncoder


class ReplacementTransformerClassifier(pl.LightningModule):
    """
    Convolutional Classifier.
    """
    def __init__(self,
                 init_lr = 0.001,
                 lr_decay = 0.03, 
                 num_category = 125, 
                 num_numerical = 75, 
                 seq_len = 32, 
                 embed_dim = 200, 
                 num_heads = 2, 
                 dropout = .1, 
                 feedforward_dim = 400, 
                 emphasis = .75,
                 numerical_weight_loss = 0.8,
                 categorical_weight_loss = 0.2,                 
                 mask_weight_loss = 2,
                 swap_category = 0.5,
                 swap_numerical = 0.2,
                 inter_dim = 200,
                 output_dim = 2,
                 freeze_encoder =  True,
                 name_labels_logger = None,
                 checkpoint =  None,
                 **kwargs):
        """Init Convolutional classifier parameters.

        The deafult parameters can be set using the file config.model.conv-classifier.yaml

        Args:
            init_lr (float): Initial learning rate.
            lr_decay (float): Learning rate decay used by the scheduler
            num_category (int): Number of categorical features.
            num_numerical (init): Number of numerical features.
            seq_len (int): Sequence length.
            embed_dim (init): Embedding size.
            num_heads (init): Number of heads.
            dropout (float): Dropout probability.
            feedforward_dim (int): Dimension position-wise feedforward net.
            emphasis (float): Masked position emphasis weight.
            numerical_weight_loss (float): Weights for numerical loss.
            categorical_weight_loss (float): Weights for categorical loss.
            mask_weight_loss (float): Weights for mask loss.
            swap_category (float): Noiseswap probability for categorical features.
            swap_numerical (float): Noiseswap probability for numerical features.
            output_dim (int): Output dimmesion
            freeze_encoder (boolean): Flag to freeze teh weights of pretrain model.
            name_labels_logger (list[str]): label names used in logger eg. WandB.
            checkpoint (dir): Pretrain model's Checkpoint directory.
        """
        super(ReplacementTransformerClassifier, self).__init__()

        # Hyperparameters
        self.save_hyperparameters()

        # Init the pretrained LightningModule
        self.autoencoder = TransformerAutoEncoder(init_lr=self.hparams.init_lr,
                                                  lr_decay=self.hparams.lr_decay, 
                                                  num_category=self.hparams.num_category, 
                                                  num_numerical=self.hparams.num_numerical, 
                                                  seq_len=self.hparams.seq_len, 
                                                  embed_dim=self.hparams.embed_dim, 
                                                  num_heads=self.hparams.num_heads, 
                                                  dropout=self.hparams.dropout, 
                                                  feedforward_dim=self.hparams.feedforward_dim, 
                                                  emphasis=self.hparams.emphasis,
                                                  numerical_weight_loss=self.hparams.numerical_weight_loss,
                                                  categorical_weight_loss=self.hparams.categorical_weight_loss,
                                                  mask_weight_loss=self.hparams.mask_weight_loss,
                                                  swap_category=self.hparams.swap_category,
                                                  swap_numerical=self.hparams.swap_numerical)

        self.autoencoder.load_from_checkpoint(self.hparams.checkpoint)

        # Freeze Pretrain encoder
        if self.hparams.freeze_encoder:
            self.autoencoder.freeze()

        # Init Layers
        self.dropout = nn.Dropout(p=self.hparams.dropout)

        self.input_linear = nn.Linear(self.hparams.embed_dim * self.hparams.seq_len,
                                      self.hparams.inter_dim)

        self.output_linear = nn.Linear(self.hparams.inter_dim,
                                       self.hparams.output_dim)

        # Criterion
        self.criterion = nn.BCELoss()

    def forward(self, x):
        # Feature extractor
        first_features, intermin_features, last_features = self.autoencoder(x)
        # Unroll input and features
        #input_unroll = self.sequence_unroll(x)
        last_features_unroll = self.sequence_unroll(last_features)
        # Enriching Input
        #input_expanded = torch.cat([last_features_unroll, input_unroll], dim=1)
        # Classifier
        x = torch.relu((self.dropout(self.input_linear(last_features_unroll))))
        x = torch.sigmoid((self.output_linear(x)))
        return x

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(),
                                      lr=self.hparams.init_lr
                                      )
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 
                                                           gamma=self.hparams.lr_decay
                                                           )
        return [optimizer], [scheduler]

    def sequence_unroll(self, x):
        batch_size = x.shape[0]
        x = x.reshape((batch_size, -1))
        return x

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
        input, labels = batch
        predictions = self(input)
        loss = self.criterion(predictions, labels)
        outputs = self.metrics_logger_custom(predictions, labels, 'train')
        self.log('train/loss_step', loss)
        return {'loss': loss, 'metrics': outputs}

    def training_epoch_end(self, outputs):
        for _, metric in enumerate(outputs[0]['metrics'].keys()):
            self.log(metric, torch.tensor([x['metrics'][metric] for x in outputs]).mean())

    def validation_step(self, batch, batch_idx):
        input, labels = batch
        predictions = self(input)
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
