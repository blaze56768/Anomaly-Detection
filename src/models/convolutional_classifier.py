"""
src.models.convolutional_classifier.py

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

from src.models.convolutional_autoencoder import ConvNetAutoEncoder


class ReplacementConvClassifier(pl.LightningModule):
    """
    Convolutional Classifier.
    """
    def __init__(self,
                 lr= 0.001,
                 latent_dim= 1024,
                 series_dim= 32,
                 features_dim= 160,
                 num_category= 54,
                 num_numerical= 106,
                 num_conv_blocks= 2,
                 dims_expand= [1,2,4,8,16],
                 dropout_prob= 0.1,
                 output_dim= 2,
                 weight_decay= 0.05,
                 inter_dim = 54,
                 freeze_encoder= True,
                 name_labels_logger = None,
                 weights_class= None,
                 checkpoint = None,
                 **kwargs):
        """Init Convolutional classifier parameters.

        The deafult parameters can be set using the file config.model.conv-classifier.yaml

        Args:
            lr (float): learning rate.
            latent_dim (int): Latent dimmesion size.
            series_dim (int): Time series length.
            features_dim (int): Number of features.
            num_category (int): Number of categorical features,
            num_numerical (int): Number of numerical features, 
            categorical_weight_loss (float): Weights for categorical loss.
            numerical_weight_loss (float): Weights for numerical loss.
            num_conv_blocks (int): Number of convolutional layers.
            dims_expand (list[int]): List of available dimession expansion.
            dropout_prob (float): Probability dropout layers.
            weight_decay (float): Weight decay
            linear_reduction (int): Ratio between the latent and inter linear layer.
            output_dim (int): Output dimmesion
            freeze_encoder (boolean): Flag to freeze teh weights of pretrain model.
            name_labels_logger (list[str]): label names used in logger eg. WandB.
            checkpoint (dir): Pretrain model's Checkpoint directory.
        """
        super(ReplacementConvClassifier, self).__init__()

        # Hyperparameters
        self.save_hyperparameters()

        # Init the pretrained LightningModule
        autoencoder = ConvNetAutoEncoder(lr=self.hparams.lr,
                                         inter_dim=self.hparams.inter_dim,
                                         latent_dim=self.hparams.latent_dim,
                                         series_dim=self.hparams.series_dim,
                                         features_dim=self.hparams.features_dim,
                                         num_category= self.hparams.num_category,
                                         num_numerical= self.hparams.num_numerical,
                                         num_conv_blocks=self.hparams.num_conv_blocks,
                                         dims_expand=self.hparams.dims_expand,
                                         dropout_prob=self.hparams.dropout_prob,
                                         weight_decay=self.hparams.weight_decay)

        autoencoder.load_from_checkpoint(self.hparams.checkpoint)

        # Freeze Pretrain encoder
        if self.hparams.freeze_encoder:
            autoencoder.freeze()
        self.encoder = autoencoder.encoder

        # Init Layers
        self.dropout = nn.Dropout(p=self.hparams.dropout_prob)

        self.input_linear = nn.Linear(self.hparams.latent_dim,
                                      self.hparams.inter_dim)

        self.output_linear = nn.Linear(self.hparams.inter_dim,
                                       self.hparams.output_dim)

        # Criterion
        #self.criterion = nn.BCELoss() #nn.L1Loss()
        self.criterion = nn.MSELoss(reduce=False)
        self.weights = torch.tensor([self.hparams.weights_class]).type(torch.FloatTensor)
        
        # self.criterion = nn.BCELoss(weight=self.weights,reduce=False)

    def forward(self, x):
        representations = self.encoder(x)
        x = torch.relu((self.dropout(self.input_linear(representations))))
        x = torch.sigmoid((self.output_linear(x)))
        return x
    
    def split(self, x):
        return torch.split(x, [self.hparams.num_numerical, self.hparams.num_category], dim=2)
    
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

    def weighted_loss(self, predictions,labels):
          unreduced_loss = self.criterion(predictions, labels)
          label_weights = labels*self.weights + (1-labels)*(1-self.weights)
          weighted_unreduced_loss = unreduced_loss*label_weights
          loss = weighted_unreduced_loss.mean()
          return loss


    def training_step(self, batch, batch_idx):
        input, labels  = batch
        #input_noise = self.add_noise(input, self.hparams.noise_factor)
        predictions = self(input)
        loss = self.weighted_loss(predictions, labels) #self.criterion(predictions, labels) #
        outputs = self.metrics_logger_custom(predictions, labels, 'train')
        self.log('train/loss_step', loss)
        return {'loss': loss, 'metrics': outputs}

    def training_epoch_end(self, outputs):
        for _, metric in enumerate(outputs[0]['metrics'].keys()):
            self.log(metric, torch.tensor([x['metrics'][metric] for x in outputs]).mean())

    def validation_step(self, batch, batch_idx):
        input, labels = batch
        predictions = self(input)
        loss = self.weighted_loss(predictions, labels) #self.criterion(predictions, labels)
        outputs = self.metrics_logger_custom(predictions, labels, 'val')
        outputs['val/loss'] = loss
        return outputs

    def validation_epoch_end(self, outputs):
        for _, metric in enumerate(outputs[0].keys()):
            self.log(metric, torch.tensor([x[metric] for x in outputs]).mean())

    def test_step(self, batch, batch_idx):
        input, labels = batch
        predictions = self(input)
        outputs = self.metrics_logger_custom(predictions, labels, 'test')
        return outputs

    def test_epoch_end(self, outputs):
        for _, metric in enumerate(outputs[0].keys()):
            self.log(metric, torch.tensor([x[metric] for x in outputs]).mean())
