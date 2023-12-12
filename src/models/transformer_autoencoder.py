"""src.models.transformer_autoencoder.py

This module defines the tranformer autoencoder.

.. _Google Python Style Guide:
   http://google.github.io/styleguide/pyguide.html

"""

__author__ = "Luis Felipe Villa Arenas"
__copyright__ = "Deutsche Telekom"


import torch
import torch.nn as nn
import pytorch_lightning as pl

from src.models.modules.autoencoders.transformer.block import TransformerEncoder
from src.models.modules.autoencoders.transformer.noiser import SwapNoiseMasker


class TransformerAutoEncoder(pl.LightningModule):
    """
    Transformer Autoencoder.
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
                 **kwargs):
        """Init Autoencoder parameters.

        The deafult parametera can be set using the file config.model.conv-autoenc.yaml

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
        """
        super(TransformerAutoEncoder, self).__init__()

        self.save_hyperparameters()
        
        # Init Linear Layers
        self.linear_mask_predictor = torch.nn.Linear(in_features=self.hparams.embed_dim * self.hparams.seq_len, 
                                              out_features=self.hparams.embed_dim * self.hparams.seq_len)
        
        self.linear_reconstructor = torch.nn.Linear(in_features=2 * self.hparams.embed_dim * self.hparams.seq_len, 
                                             out_features=self.hparams.embed_dim * self.hparams.seq_len)

        # Init Swap Noiser 
        self.noise_maker = SwapNoiseMasker(self.hparams)

        # Init Tranformer layers
        self.encoder_low = TransformerEncoder(self.hparams)                            
        self.encoder_medium = TransformerEncoder(self.hparams)
        self.encoder_high = TransformerEncoder(self.hparams)

        # loss function
        self.criterion_mse = torch.nn.functional.mse_loss
        self.criterion_bce_logits = torch.nn.functional.binary_cross_entropy_with_logits

    def forward(self, x):
        first_features = self.encoder_low(x)
        intermin_features = self.encoder_medium(first_features)
        last_features = self.encoder_high(intermin_features)
        return (first_features, intermin_features, last_features)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(),
                                      lr=self.hparams.init_lr
                                      )
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 
                                                           gamma=self.hparams.lr_decay
                                                           )
        return [optimizer], [scheduler]

    def step(self, batch):
        input, labels, weights = batch

        # Corrupting Input
        input_corrputed, mask = self.noise_maker.apply(input)
    
        # Output features from transformer layers
        (_, _, last_features) = self.forward(input_corrputed)
        
        # Unroll features
        last_features_unroll = self.sequence_unroll(last_features)
        
        # Prediction mask and reconstruction
        predicted_mask_unroll = self.linear_mask_predictor(last_features_unroll)
        reconstruction_unroll = self.linear_reconstructor(torch.cat([last_features_unroll, predicted_mask_unroll], dim=1))

        # Roll mask prediction and reconstruction
        reconstruction = self.sequence_roll(reconstruction_unroll)
        predicted_mask = self.sequence_roll(predicted_mask_unroll)

        # Losses Reconstruction and mask
        numerical_loss, categorical_loss = self.reconstruction_loss(input, reconstruction, mask)                                 
        mask_loss = self.mask_loss(mask, predicted_mask)
       
        return numerical_loss, categorical_loss, mask_loss
    
    def reconstruction_loss(self, input, reconstruction, mask):
        input_recover_numerical, input_recover_categorical = self.split(reconstruction)
        input_numerical, input_categorical = self.split(input)
        mask_emphasis = mask * self.hparams.emphasis + (1 - mask) * (1 - self.hparams.emphasis)
        weights_numerical, weights_categorical = self.split(mask_emphasis)
        categorical_loss = torch.mul(weights_categorical, self.criterion_bce_logits(input_recover_categorical,
                                                                                    input_categorical,
                                                                                    reduction='none'))
        numerical_loss = torch.mul(weights_numerical, self.criterion_mse(input_recover_numerical,
                                                                        input_numerical,
                                                                        reduction='none'))
        return numerical_loss.mean(), categorical_loss.mean()
    
    def apply_weights(self, numerical_loss, categorical_loss, mask_loss):
        numerical_loss = self.hparams.numerical_weight_loss * numerical_loss
        categorical_loss = self.hparams.categorical_weight_loss * categorical_loss
        mask_loss = self.hparams.mask_weight_loss * mask_loss
        return numerical_loss, categorical_loss, mask_loss

    def mask_loss(self, mask, predicted_mask):
        mask_loss = self.criterion_bce_logits(predicted_mask,
                                              mask,
                                              reduction='mean')
        return mask_loss

    def sequence_roll(self, x):
        batch_size = x.shape[0]
        x = x.reshape((batch_size, self.hparams.seq_len, self.hparams.embed_dim))
        return x

    def sequence_unroll(self, x):
        batch_size = x.shape[0]
        x = x.reshape((batch_size, -1))
        return x

    def split(self, x):
        return torch.split(x, [self.hparams.num_numerical, self.hparams.num_category], dim=2)

    def feature(self, x):
        attn_outs = self.forward(x)
        return torch.cat([self.sequence_unroll(x) for x in attn_outs], dim=1)

    def training_step(self, batch, batch_idx):
        numerical_loss, categorical_loss, mask_loss = self.step(batch)
        numerical_loss_weight, categorical_loss_weight, mask_loss_weight = self.apply_weights(numerical_loss,
                                                                                              categorical_loss,
                                                                                              mask_loss)
        loss = numerical_loss_weight + categorical_loss_weight + mask_loss_weight
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train/numerical_loss", numerical_loss)
        self.log("train/categorical_loss", categorical_loss)
        self.log("train/mask_loss", mask_loss)
        return loss

    def validation_step(self, batch, batch_idx):
        numerical_loss, categorical_loss, mask_loss = self.step(batch)
        numerical_loss_weight, categorical_loss_weight, mask_loss_weight = self.apply_weights(numerical_loss,
                                                                                              categorical_loss,
                                                                                              mask_loss)                                                                                    
        loss = numerical_loss_weight + categorical_loss_weight + mask_loss_weight
        return {"loss": loss, "numerical_loss":numerical_loss, "categorical_loss":categorical_loss, "mask_loss":mask_loss}

    def validation_epoch_end(self, outputs):
        avg_val_loss = torch.tensor([x["loss"] for x in outputs]).mean()
        avg_val_numerical_loss = torch.tensor([x["numerical_loss"] for x in outputs]).mean()
        avg_val_categorical_loss = torch.tensor([x["categorical_loss"] for x in outputs]).mean()
        avg_val_mask_loss = torch.tensor([x["mask_loss"] for x in outputs]).mean()
        self.log("val/loss", avg_val_loss)
        self.log("val/numerical_loss", avg_val_numerical_loss)
        self.log("val/categorical_loss", avg_val_categorical_loss)
        self.log("val/mask_loss", avg_val_mask_loss)

    def test_step(self, batch, batch_idx):
        numerical_loss, categorical_loss, mask_loss = self.step(batch)
        numerical_loss_weight, categorical_loss_weight, mask_loss_weight = self.apply_weights(numerical_loss,
                                                                                              categorical_loss,
                                                                                              mask_loss)
        loss = numerical_loss_weight + categorical_loss_weight + mask_loss_weight
        return {"loss": loss, "numerical_loss":numerical_loss, "categorical_loss":categorical_loss, "mask_loss":mask_loss}

    def test_epoch_end(self, outputs):
        avg_test_loss = torch.tensor([x["loss"] for x in outputs]).mean()
        avg_test_numerical_loss = torch.tensor([x["numerical_loss"] for x in outputs]).mean()
        avg_test_categorical_loss = torch.tensor([x["categorical_loss"] for x in outputs]).mean()
        avg_test_mask_loss = torch.tensor([x["mask_loss"] for x in outputs]).mean()
        self.log("test/loss", avg_test_loss)
        self.log("test/numerical_loss", avg_test_numerical_loss)
        self.log("test/categorical_loss", avg_test_categorical_loss)
        self.log("test/mask_loss", avg_test_mask_loss)
