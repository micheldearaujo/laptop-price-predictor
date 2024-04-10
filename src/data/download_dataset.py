"""
    Purpose: 
      This script is responsibe only for downloading the RAW laptop prices dataset from the github repository.
        
      Out table:  lp_raw_dataset
"""

import sys
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pyspark.sql import SparkSession

sys.path.insert(0,'.')

spark = SparkSession.builder.getOrCreate()

def data_extraction_orchestration(save_to_table: bool = True) -> pd.DataFrame:
    """
    Perform data extraction orchestration.

    This function reads laptop price data from a CSV file, converts it to a Spark DataFrame, and optionally saves it as a table.

    Parameters:
        save_to_table (bool, optional): If True, the data will be saved as a table. Defaults to True.

    Returns:
        pd.DataFrame: DataFrame containing the laptop price data.
    """

    prices_df = pd.read_csv("https://raw.githubusercontent.com/micheldearaujo/datasets/main/laptop_price/laptop_price.csv", sep=',', encoding='latin1')
    prices_df = spark.createDataFrame(prices_df)

    if save_to_table:
        
        #prices_df.write.mode("overwrite").saveAsTable("lp_raw_dataset")
        prices_df.createOrReplaceGlobalTempView("lp_raw_dataset")

    return prices_df

