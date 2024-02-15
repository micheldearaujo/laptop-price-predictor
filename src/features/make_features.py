"""
    Purpose: 
      This script is responsibe for performing data cleaning in the raw laptop prices dataset.

        In table:   lp_interim_clean_prices
        Out table:  lp_processed_modeling
"""

import sys
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pyspark.sql import SparkSession

sys.path.insert(0,'.')

from src.config import companies_to_agg, columns_to_dummy 

spark = SparkSession.builder.getOrCreate()

def select_top_features(clean_dataframe: pd.DataFrame):
    """
    Select the top 20 features based on Linear Correlation

    Args:
        clean_dataframe (pd.DataFrame): The cleaned dataframe, out of the build_features function.

    Returns:
        pd.DataFrame: Cleaned Dataframe only with the top performing features.
    """

    top_20_features = [
        'company_MSI',
        'CPU_BRAND_AMD',
        'CPU_BRAND_Intel',
        'GPU_BRAND_Intel',
        'GPU_BRAND_AMD',
        'company_Acer',
        'Weight',
        'Memory_Type_Flash',
        'typeName_Workstation',
        'typeName_Ultrabook',
        'GPU_BRAND_Nvidia',
        'typeName_Gaming',
        'Memory_Type_HDD',
        'CPU_FREQUENCY',
        'Memory_Type_SSD',
        'typeName_Notebook',
        'ScreenHeight',
        'ScreenWidth',
        'Ram',
        'Price_euros'
    ]

    model_dataframe = clean_dataframe[top_20_features]

    return model_dataframe


def dummization_cat_columns(raw_prices_pd: pd.DataFrame, columns_to_dummy: list) -> pd.DataFrame:
    """
    Perform one-hot encoding for categorical columns in the DataFrame.

    This function performs one-hot encoding for specified categorical columns in the DataFrame.

    Parameters:
        raw_prices_pd (pd.DataFrame): The input DataFrame containing raw laptop price data.
        columns_to_dummy (list): A list of column names to perform one-hot encoding.

    Returns:
        pd.DataFrame: The DataFrame with categorical columns one-hot encoded.
    """
    raw_prices_pd = raw_prices_pd.join(pd.get_dummies(raw_prices_pd["Company"], prefix="company", dtype="int"))
    raw_prices_pd = raw_prices_pd.join(pd.get_dummies(raw_prices_pd["TypeName"], prefix="typeName", dtype="int"))
    raw_prices_pd = raw_prices_pd.join(pd.get_dummies(raw_prices_pd["CPU_BRAND"], prefix="CPU_BRAND", dtype="int"))
    raw_prices_pd = raw_prices_pd.join(pd.get_dummies(raw_prices_pd["GPU_BRAND"], prefix="GPU_BRAND", dtype="int"))
    raw_prices_pd = raw_prices_pd.join(pd.get_dummies(raw_prices_pd["OpSys"], prefix="OpSys", dtype="int"))
    raw_prices_pd = raw_prices_pd.join(pd.get_dummies(raw_prices_pd["Memory_Type"], prefix="Memory_Type", dtype="int"))

    raw_prices_pd = raw_prices_pd.drop(columns=columns_to_dummy)

    return raw_prices_pd


def aggregate_small_companies(clean_prices_pd:pd.DataFrame, companies_to_agg: list) -> pd.DataFrame:
    """
    Aggregate small companies in the DataFrame.

    This function aggregates small companies in the DataFrame by replacing their names with 'Other'.

    Parameters:
        clean_prices_pd (pd.DataFrame): The input DataFrame containing clean laptop price data.
        companies_to_agg (list): A list of company names to be aggregated.

    Returns:
        pd.DataFrame: The DataFrame with small companies aggregated.
    """
    clean_prices_pd["Company"] = clean_prices_pd["Company"].apply(lambda x: "Other" if x in companies_to_agg else x)

    return clean_prices_pd

def transform_string_columns_to_cat(clean_prices_pd: pd.DataFrame):

    string_columns = clean_prices_pd.select_dtypes(include='object').columns

    for column in string_columns:
        clean_prices_pd[column] = clean_prices_pd[column].astype("category")

    return clean_prices_pd


def feature_engineering_orchestration(save_to_table: bool = True, one_hot: bool = True):
    """
    Perform feature engineering orchestration for laptop price data.

    This function orchestrates the feature engineering steps for laptop price data. It reads clean data from a Spark table,
    performs feature engineering operations including aggregating small companies, one-hot encoding categorical variables,
    and selecting top features, and optionally saves the processed data as a Spark table.

    Parameters:
        save_to_table (bool, optional): If True, the processed data will be saved as a Spark table. Defaults to True.
        one_hot (bool, optional): If True, perform one-hot encoding for categorical variables. Defaults to True.

    Returns:
        pd.DataFrame: The processed DataFrame containing features for modeling.
    """
    clean_prices_pd = spark.sql("select * from lp_interim_clean_prices").toPandas()

    clean_prices_pd = aggregate_small_companies(clean_prices_pd, companies_to_agg)

    if one_hot:
        clean_prices_pd = dummization_cat_columns(clean_prices_pd, columns_to_dummy)

    else:
        clean_prices_pd = transform_string_columns_to_cat(clean_prices_pd)

    for column in clean_prices_pd.columns:
        clean_prices_pd = clean_prices_pd.rename(columns={column: column.replace(" ", "_")})

    clean_prices_pys = spark.createDataFrame(clean_prices_pd)

    if save_to_table:
        if one_hot:
            typer = "_dummy"
        else:
            typer = ""

        #spark.createDataFrame(clean_prices_pd).write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(f"lp_processed_modeling{typer}")
        clean_prices_pys.createOrReplaceTempView(f"lp_processed_modeling{typer}")

    return clean_prices_pd
