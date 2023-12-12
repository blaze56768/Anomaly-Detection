"""
replacement_datamodule.py

This module defines the data module and dataset class.

.. _Google Python Style Guide:
   http://google.github.io/styleguide/pyguide.html

"""
__author__ = "Luis Felipe Villa Arenas"
__copyright__ = "Deutsche Telekom"

from omegaconf import OmegaConf

import torch
from torch.utils.data import Dataset

from pyspark.sql import SparkSession
import pyspark.sql.functions as F


class ReplacementDataset(Dataset):
    """
    Replacement dataset.
    """
    def __init__(self,
                 data_dir,
                 features_dim,
                 series_dim,
                 name_labels,
                 features_numerical,
                 features_categorical):
        """Init paramaters for dataset building.

        Args:
            data_dir (str): Path to data.
            features_dim (int): Number of Features.
            series_dim (int): Number of elements in the time series.
            name_labels (list[str]): List of label names.

        """
        super(ReplacementDataset, self).__init__()

        self.data_dir = data_dir
        self.features_dim = features_dim
        self.series_dim = series_dim
        self.name_labels = list(name_labels)
        self.features_numerical = list(features_numerical)
        self.features_categorical = list(features_categorical)

        self.data = self.read_data()
        self.asset_ids = self.get_asset_ids()

    def spark_init(self):
        """Initialization of the Spark Session

        Returns:
            spark (object): Spark Session.
        """
        spark = SparkSession.builder \
                            .master("local[*]") \
                            .appName("cpe-replacement") \
                            .config("spark.driver.memory", "2g") \
                            .getOrCreate()
        return spark

    def read_data(self):
        """Load Spark dataframe and transformed to pandas
        dataframe.

        Returns:
            dataframe: data.
        """
        spark = self.spark_init()
        df = spark.read.parquet(self.data_dir)
        #df = df.filter((F.col('label_minor_physical_thermal_damage')==1)|(F.col('label_no_problem')==1))
        df = df.filter((F.col('cpetype_SpeedportPlus')==1)) #|(F.col('label_healthy')==0)
        #df = df.filter(F.col('label_healthy')==0)
        data = df.toPandas()        
        return data

    def get_asset_ids(self):
        """Get unique asset ids.ßßß

        Returns:
            list: asset ids.
        """
        return self.data.assetid.unique()

    def __len__(self):
        """Dataset length.
        Returns:
            int: Dataset length.
        """
        return len(self.asset_ids)

    def __getitem__(self, idx: int):
        """Get item.
        Args:
            idx (int): Index.
        Returns:
            {
                str: User ID,
                list: [batch, num_click_docs, seq_len],
                list: [batch, num_candidate_docs, seq_len],
                bool: candidate docs label (0 or 1)
            }
        """
        # Filter by assetid
        item = self.data.loc[self.data.assetid == self.asset_ids[idx]].copy()

        # Sort by date
        item = item.sort_values('datum')
        seq_len = item.shape[0]
        seq_id = [i/seq_len for i in range(seq_len)]
        item['seq_id'] = seq_id

        # Labels
        labels = item[self.name_labels].values[0]
        labels = torch.tensor(labels).type(torch.FloatTensor)

        # Input
        input = item[['seq_id']+ self.features_numerical + self.features_categorical].values
        input = torch.tensor(input).type(torch.FloatTensor)

        # Padder days
        padder_sequence = torch.zeros(self.series_dim - input.shape[0],
                                      input.shape[1])
        input = torch.cat([padder_sequence, input], dim=0)

        # Padder features
        padder_features = torch.zeros(self.series_dim,
                                      self.features_dim - input.shape[1])
        input = torch.cat([padder_features, input], dim=1)

        return input, labels

