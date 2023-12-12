"""
datamodule.py

This module defines the data module and dataset class.

.. _Google Python Style Guide:
   http://google.github.io/styleguide/pyguide.html

"""
__author__ = "Luis Felipe Villa Arenas"
__copyright__ = "Deutsche Telekom"

import os
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from pytorch_lightning import LightningDataModule
import pandas as pd

from src.datamodules.datasets.replacement_dataset import ReplacementDataset


class ReplacementDataModule(LightningDataModule):
    """Data Module for CPE Replacement.
    """

    def __init__(self,
                 data_dir: str = "data/preprocessed",
                 train_val_split: float = 0.9,
                 batch_size: int = 32,
                 num_workers: int = 0,
                 pin_memory: bool = False,
                 features_dim: int = 184,
                 series_dim: int = 32,
                 name_labels: list = ["label_cpe_replacement"],
                 sampler = False,
                 weights_minority_class: float = 1.0,
                 features_numerical: list = [],
                 features_categorical: list = [], 
                 **kwargs,):
        """Init data module.

        The deafult parametera can be set using the file config.datamodule.cpe-replacement.yaml

        Args:
            data_dir (str): Preprocessed data directory.
            train_val_split (float): Dataset share assing for training.
            batch_size (int): Batch size,
            num_workers (int): Number of parallel workers in a GPU env.
            pin_memory (bool): Pin memory enable for GOU env,
            series_dim (int): Time series length.
            features_dim (int): Number of features.
            name_labels (list[str]): Label names used in training.

        """
        super().__init__()

        self.data_dir = data_dir
        self.train_val_split = train_val_split
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.features_dim = features_dim
        self.series_dim = series_dim
        self.name_labels = name_labels
        self.features_numerical = features_numerical
        self.features_categorical = features_categorical
        self.sampler = sampler
        self.weights_minority_class = weights_minority_class

    def setup(self, stage=None):
        # Setup function is key when handling dataloaders inside the class since
        # this function is triggered when training or testing the model by passing the
        # right datasets (training, validation and testing)
        if not os.path.exists(self.data_dir + "/torch"):
            os.makedirs(self.data_dir + '/torch')
            train_dataset = ReplacementDataset(self.data_dir + "/train.parquet",
                                               self.features_dim,
                                               self.series_dim,
                                               self.name_labels,
                                               self.features_numerical,
                                               self.features_categorical)
            torch.save(train_dataset, self.data_dir + '/torch/train.pt')

            test_dataset = ReplacementDataset(self.data_dir + "/test.parquet",
                                              self.features_dim,
                                              self.series_dim,
                                              self.name_labels,
                                              self.features_numerical,
                                              self.features_categorical)
            torch.save(test_dataset, self.data_dir + '/torch/test.pt')

        if stage == 'fit' or stage is None:
            """dataset = torch.load(self.data_dir + '/torch/train.pt')
            ds_size = dataset.__len__()
            train_size = int(self.train_val_split * ds_size)
            val_size = ds_size - train_size
            split = [train_size, val_size]
            self.train_dataset, self.val_dataset = torch.utils.data.random_split(dataset, split)"""
            self.train_dataset = torch.load(self.data_dir + '/torch/train.pt')
            self.val_dataset = torch.load(self.data_dir + '/torch/test.pt')

        if stage == 'test' or stage is None:
            self.test_dataset = torch.load(self.data_dir + '/torch/test.pt')
        
        if self.sampler == True:
            self.train_sampler = self.get_sampler(self.train_dataset)

    def get_sampler(self, dataset):
        synthetic_class = list()
        indices = list()
        for idx, (input, labels) in enumerate(dataset):            
            synthetic_class.append(labels[0].item())
            indices.append(idx)
        
        df_class = pd.DataFrame(data={'indices': indices, 'synthetic_class': synthetic_class})
        
        df_class_stats = df_class.groupby(['synthetic_class']).count().reset_index()
        df_class_stats['weights'] = len(indices) / df_class_stats['indices']
        print('Samples weights for Balance Random Sampler: \n', df_class_stats.head())

        df_class_stats = pd.DataFrame(data={'synthetic_class': [0.0, 1.0 ], 
                                            'weights': [1.0, self.weights_minority_class]})
        print('Samples weights for Custom Random Sampler: \n', df_class_stats.head())

        df_class = df_class.merge(df_class_stats[['synthetic_class', 'weights']],
                                  how='left',
                                  on='synthetic_class')

        samples_weights = df_class['weights'].to_list()
        sampler = WeightedRandomSampler(weights=samples_weights,
                                        num_samples=len(samples_weights),
                                        replacement=True)
        
        return sampler

    def train_dataloader(self):
        # Called when training the model, for balance sampler add in args
        # sampler=self.train_sampler, otherwise shuffle=True
        if self.sampler == True:
            dataloader = DataLoader(self.train_dataset,
                                    batch_size=self.batch_size,
                                    sampler=self.train_sampler,
                                    num_workers=self.num_workers)
        else:
            dataloader = DataLoader(self.train_dataset,
                                    batch_size=self.batch_size,
                                    shuffle=True,
                                    num_workers=self.num_workers)
        return dataloader
        

    def val_dataloader(self):
        # Called when evaluating the model (for each "n" steps or "n" epochs)
        return DataLoader(self.val_dataset,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=self.num_workers)

    def test_dataloader(self):
        # Called when testing the model by calling: Trainer.test()
        return DataLoader(self.test_dataset,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=self.num_workers)
