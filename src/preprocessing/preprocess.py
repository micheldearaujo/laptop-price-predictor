"""
    Purpose: 
      This script is responsibe for performing data cleaning in the raw laptop prices dataset.

        In table:   lp_raw_dataset
        Out table:  lp_interim_clean_prices
"""

import sys
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pyspark.sql import SparkSession

sys.path.insert(0,'.')

spark = SparkSession.builder.getOrCreate()

def convert_to_gigabytes(string):
    """
    Converts Hard Drive string memory to float.
    """
    number = float(string[:-2])

    if "TB" in string:
        number = number * 1024
    
    return number


def clean_screen_resolution_column(raw_prices_pd: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the 'ScreenResolution' column in the DataFrame.

    This function extracts screen width and height from the 'ScreenResolution' column and creates new columns for them.

    Parameters:
        raw_prices_pd (pd.DataFrame): The input DataFrame containing raw laptop price data.

    Returns:
        pd.DataFrame: The DataFrame with the 'ScreenResolution' column cleaned and new columns for screen width and height added.
    """
    raw_prices_pd["ScreenResolution"] = raw_prices_pd["ScreenResolution"].str.split(" ").apply(lambda x: x[-1])
    raw_prices_pd["ScreenWidth"] =  raw_prices_pd["ScreenResolution"].str.split("x").apply(lambda x: x[0]).astype("int")
    raw_prices_pd["ScreenHeight"] =  raw_prices_pd["ScreenResolution"].str.split("x").apply(lambda x: x[1]).astype("int")

    return raw_prices_pd


def clean_cpu_columns(raw_prices_pd: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the CPU-related columns in the DataFrame.

    This function extracts CPU brand and frequency information from the 'Cpu' column and creates new columns for them.

    Parameters:
        raw_prices_pd (pd.DataFrame): The input DataFrame containing raw laptop price data.

    Returns:
        pd.DataFrame: The DataFrame with CPU-related columns cleaned and new columns added for CPU brand and frequency.
    """
    raw_prices_pd["CPU_BRAND"] =  raw_prices_pd["Cpu"].str.split(" ").apply(lambda x: x[0])
    raw_prices_pd["CPU_FREQUENCY"] =  raw_prices_pd["Cpu"].str.split(" ").apply(lambda x: x[-1])
    raw_prices_pd["CPU_FREQUENCY"] =  raw_prices_pd["CPU_FREQUENCY"].apply(lambda x: x[:-3]).astype("float")
    
    return raw_prices_pd


def clean_memory_columns(raw_prices_pd: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the memory-related columns in the DataFrame.

    This function extracts RAM size and memory type information from the 'Ram' and 'Memory' columns and creates new columns for them.

    Parameters:
        raw_prices_pd (pd.DataFrame): The input DataFrame containing raw laptop price data.

    Returns:
        pd.DataFrame: The DataFrame with memory-related columns cleaned and new columns added for RAM size and memory type.
    """
    raw_prices_pd["Ram"] =  raw_prices_pd["Ram"].apply(lambda x: x[:-2]).astype("int")
    raw_prices_pd["Memory_Size"] = raw_prices_pd["Memory"].str.split(" ").apply(lambda x: x[0])
    raw_prices_pd["Memory_Size"] = raw_prices_pd["Memory_Size"].apply(lambda x: convert_to_gigabytes(x))
    raw_prices_pd["Memory_Type"] = raw_prices_pd["Memory"].str.split(" ").apply(lambda x: x[1])
    
    return raw_prices_pd


def drop_residual_columns(raw_prices_pd: pd.DataFrame) -> pd.DataFrame:
    """
    Drop residual columns from the DataFrame.

    This function drops residual columns that are no longer needed in the DataFrame.

    Parameters:
        raw_prices_pd (pd.DataFrame): The input DataFrame containing raw laptop price data.

    Returns:
        pd.DataFrame: The DataFrame with residual columns dropped.
    """
    clean_prices = raw_prices_pd.drop(["laptop_ID", "Product", "ScreenResolution", "Cpu", "Memory", "Gpu"], axis=1)

    return clean_prices

def preprocessing_orchestration(save_to_table: bool = True) -> pd.DataFrame:
    """
    Perform preprocessing orchestration for laptop price data.

    This function orchestrates the preprocessing steps for laptop price data. It reads raw data from a Spark table,
    performs cleaning operations including standardizing column names, handling categorical variables, and dropping residual columns,
    and optionally saves the preprocessed data as a Spark table.

    Parameters:
        raw_prices_pd (pd.DataFrame): The input DataFrame containing raw laptop price data.
        save_to_table (bool, optional): If True, the preprocessed data will be saved as a Spark table. Defaults to True.
        one_hot (bool, optional): If True, perform one-hot encoding for categorical variables. Defaults to True.

    Returns:
        pd.DataFrame: The preprocessed DataFrame containing laptop price data.
    """

    raw_prices_pd = spark.sql("select * from global_temp.lp_raw_dataset").toPandas()

    raw_prices_pd = clean_screen_resolution_column(raw_prices_pd)
    raw_prices_pd = clean_cpu_columns(raw_prices_pd)
    raw_prices_pd = clean_memory_columns(raw_prices_pd)
    raw_prices_pd["Weight"] = raw_prices_pd["Weight"].apply(lambda x: x[:-2]).astype("float")
    raw_prices_pd["GPU_BRAND"] = raw_prices_pd["Gpu"].str.split(" ").apply(lambda x: x[0])
    clean_prices_pd = drop_residual_columns(raw_prices_pd)

    for column in clean_prices_pd.columns:
        clean_prices_pd = clean_prices_pd.rename(columns={column: column.replace(' ', '_')})

    clean_prices_pys = spark.createDataFrame(clean_prices_pd)

    if save_to_table:
        
        #clean_prices_pys.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable("lp_interim_clean_prices")
        clean_prices_pys.createOrReplaceGlobalTempView("lp_interim_clean_prices")

    return clean_prices_pd

