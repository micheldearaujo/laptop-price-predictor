"""
    Authors: Nina Thoni, Clarissa Chevalier, Ciro Paiva
    Purpose: Predict PoS' Consumer Occasion Type, Channel and Subchannel.
"""

import sys
import os

sys.path.append(os.path.abspath(".."))

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

import pandas as pd

import mlflow
import mlflow.sklearn
from pyspark.sql import SparkSession

spark = SparkSession.builder.getOrCreate()

from src import config


def model_path_finder(scope: str, model_name: str) -> dict:
    """
    Generate a dictionary of model URIs for different models based on the provided scope.

    Args:
        scope: The scope for which model URIs are required.

    Returns:
        dict: A dictionary mapping model names to their corresponding URIs.
    """
    model_uri = {
        "laptop": f"models:/{model_name}-{scope}/latest",
    }

    return model_uri


def _predict(X_pred: pd.DataFrame, model_uri: str, typer: str) -> object:
    """
    Make predictions using a loaded model and return results.

    Args:
        X_pred: Input data for making predictions.
        model_uri: The URI of the loaded model.
        le: The label encoder object used for encoding and decoding classes.
        typer: A string indicating the type or purpose of predictions.

    Returns:
        pd.DataFrame: A DataFrame containing prediction results, including class probabilities and the predicted class.
    """

    model = mlflow.sklearn.load_model(model_uri)

    y_pred = model.predict(X_pred)
    #y_pred = le.inverse_transform(y_pred)
    y_pred = pd.Series(y_pred, index=X_pred.index, name=typer)

    return y_pred


def all_models_predict(scope: str) -> tuple:
    """
    Make predictions for consumer_occasion, channel, and subchannel using a set of loaded machine learning models.

    Args:
        scope: A string indicating the scope or context of the predictions.

    Returns:
        dict: A dictionary containing prediction results for different models.
    """
    inference_df = spark.sql(f"select * from global_temp.lp_processed_modeling{scope}_inference").toPandas()
    X_pred = inference_df.drop(columns=[config.MODEL_CONFIG['TARGET']])
    predictions = {}

    model_uri_dict = model_path_finder(scope, config.MODEL_CONFIG['MODEL_NAME'])
    logging.warning("starting model predict")

    # predict consumer occasion
    inference_df["predicted"] = _predict(
        X_pred,
        model_uri_dict[f"laptop"],
        scope
    )

    spark.createDataFrame(inference_df).createOrReplaceGlobalTempView(f"lp_predicted{scope}")

    return inference_df