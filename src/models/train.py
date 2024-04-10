"""
    Authors: 
    Purpose: Train and select the best performing model for predicting the laptop
    prices.
"""

import sys
import os

sys.path.append(os.path.abspath(".."))

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

import pandas as pd
import numpy as np

import mlflow
import mlflow.sklearn
from mlflow.tracking.client import MlflowClient
from mlflow.models import infer_signature, set_signature
from pyspark.sql import SparkSession

spark = SparkSession.builder.getOrCreate()

from sklearn.model_selection import (
    train_test_split,
    RandomizedSearchCV,
    cross_val_score,
)

from sklearn.model_selection import KFold
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error
)

from src import config

def model_selection(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    y_train: pd.Series,
    y_val: pd.Series,
) -> tuple:
    """
    Select the best model from a set of candidate models using RMSE on the validation dataset.

    Args:
        X_train: Feature dataset for training.
        X_val: Feature dataset for validation.
        y_train: Target variable for training.
        y_val: Target variable for validation.

    Returns:
        Tuple[object, str]: A tuple containing the best model (as an object) and the name of the best model.
    """

    logger.warning("beggining model selection stage")

    models = dict()
    results = dict()

    for item in config.GRIDS.keys():

        model_ = config.MODELS[item]
        model_.fit(X_train, y_train)

        y_pred = model_.predict(X_val)

        models[item] = model_
        results[item] = np.sqrt(mean_squared_error(y_val, y_pred))

    best_key = min(results, key=results.get)

    logger.warning(f"best model is {best_key}: RMSE {results[best_key]}")
    return models[best_key], best_key


def cross_val(model_: object, X_train: pd.DataFrame, y_train: pd.Series) -> np.ndarray:
    """
    Perform k-fold cross-validation to assess the model's performance.

    Args:
        model_: The machine learning model to evaluate.
        X_train: Feature dataset for training.
        y_train: Target variable for training.

    Returns:
        np.ndarray: An array of cross-validation scores (RMSE scores).
    """

    kfold = KFold(n_splits=10, random_state=config.MODEL_CONFIG['SEED'], shuffle=True)
    scores = cross_val_score(model_, X_train, y_train, cv=kfold, scoring="neg_mean_squared_error")
    logger.warning(f"Cross-Validation Scores: {scores}")
    logger.warning(f"Cross-Validation Mean Scores: {np.mean(scores)} (RMSE: {np.sqrt(np.abs(np.mean(scores)))})")

    return scores


def hyper_tuning(
    model_: object, best_key: str, X_train: pd.DataFrame, y_train: pd.Series
) -> object:
    """
    Perform hyperparameter tuning for the selected machine learning model.

    Args:
        model_: The machine learning model to be tuned.
        best_key: The key indicating the best model selected for tuning.
        X_train: Feature dataset for training.
        y_train: Target variable for training.

    Returns:
        object: The tuned machine learning model with optimized hyperparameters.
    """

    search = RandomizedSearchCV(
        estimator=model_,
        param_distributions=config.GRIDS[best_key],
        n_iter=10,
        cv=3,
        random_state=config.MODEL_CONFIG['SEED'],
        n_jobs=-1,
        scoring="neg_mean_squared_error",
        verbose=2
    )

    logger.warning("beggining model tunning stage")
    search.fit(X_train, y_train)

    model_tunned = search.best_estimator_
    logger.warning(f"best estimator score: {np.sqrt(np.abs(search.best_score_))}")

    return model_tunned


def _train(
    X_train: pd.DataFrame, y_train: pd.Series, typer: str, cross=False, tuning=False
) -> object:
    """
    Train and evaluate a machine learning model with optional cross-validation and
    hyperparameter tuning. Also, logs the model and its evaluated metrics into MLflow.

    Args:
        X_train: Feature dataset for training.
        y_train: Target variable for training.
        typer: Type of model training (e.g., "baseline" or "tuned").
        cross (optional): Perform cross-validation if True. Defaults to False.
        tuning (optional): Perform hyperparameter tuning if True. Defaults to False.

    Returns:
        object: The trained machine learning model.
    """

    # train validation split
    X_train, X_val, y_train, y_val = train_test_split(
        X_train,
        y_train,
        test_size=config.MODEL_CONFIG["TEST_SIZE"],
        random_state=config.MODEL_CONFIG["SEED"],
    )

    for df in [X_train, X_val, y_train, y_val]:
        df = df.sort_index()

    typer = typer.replace("/", "_")  # for register purposes

    run_name = f"Model_{typer}"

    mlflow.set_experiment(experiment_name=f"/Shared/LP_EXPERIMENTS/{run_name}")

    with mlflow.start_run(run_name=run_name, nested=True) as run:
        logger.warning("#############################################")
        logger.warning(f"{typer} process:")

        # model selection
        model_, best_key = model_selection(X_train, X_val, y_train, y_val)

        # cross validation
        if cross:
            scores = cross_val(model_, X_train, y_train)
            mlflow.log_metric(f"scores_{typer}", np.mean(scores))
        else:
            pass

        if tuning:
            model_ = hyper_tuning(model_, best_key, X_train, y_train)
        else:
            pass
        

        logger.warning("calculating validation metrics for best tunned model")
        y_pred = model_.predict(X_val)
        y_pred = pd.Series(y_pred, index=y_val.index)

        signature = infer_signature(X_val, y_pred)

        # mlflow.sklearn.log_model(
        #     model_,
        #     artifact_path=f"{config.MODEL_CONFIG['MODEL_NAME']}-{typer}",
        #     signature=signature
        # )

        # calculating and persisting metrics
        mae = round(mean_absolute_error(y_val, y_pred), 2)
        mape = round(mean_absolute_percentage_error(y_val, y_pred), 2)
        rmse = round(np.sqrt(mean_squared_error(y_val, y_pred)), 2)

        # logging validation metrics to mlflow
        mlflow.log_metric(f"mae_{typer}", mae)
        mlflow.log_metric(f"mape_{typer}", mape)
        mlflow.log_metric(f"rmse_{typer}", rmse)

        # print validation metrics to console
        logger.warning(f"mae_{typer}: {mae}")
        logger.warning(f"mape_{typer}: {mape}")
        logger.warning(f"rmse_{typer}: {rmse}")
        logger.warning("#############################################")

        client = MlflowClient()
        run_id = run.info.run_id
        model_uri = f"runs:/{run_id}/{config.MODEL_CONFIG['MODEL_NAME']}-{typer}"

        if not client.search_model_versions(f"name='{config.MODEL_CONFIG['MODEL_NAME']}-{typer}'"):
            logging.warning("registering new model")
            mlflow.register_model(model_uri=model_uri, name=f"{config.MODEL_CONFIG['MODEL_NAME']}-{typer}")

        else:
            logging.warning("logging new version to already registered")
            mlflow.sklearn.log_model(
                sk_model=model_,
                artifact_path=f"{config.MODEL_CONFIG['MODEL_NAME']}-{typer}",
                registered_model_name=f"{config.MODEL_CONFIG['MODEL_NAME']}-{typer}",
                signature=signature
            )

    mlflow.end_run()


def all_models_train(scope: str, cross=False, tuning=False) -> None:
    """
    Train multiple machine learning models for consumer occasion, channel, and subchannels.

    Args:
        data: Preprocessed model input data.
        scope: CDA or revendas.

    Returns:
        dict: A dictionary containing trained machine learning models for consumer occasion, channel, and subchannels.
    """

    prices_df = spark.sql(f"select * from global_temp.lp_processed_modeling{scope}").toPandas()
    string_columns = prices_df.select_dtypes(include='object').columns

    for column in string_columns:
        prices_df[column] = prices_df[column].astype("category")

    X, y = prices_df.drop(config.MODEL_CONFIG["TARGET"], axis=1), prices_df[config.MODEL_CONFIG["TARGET"]]

    # model 1 -> consumer occasion model
    _train(X, y, f"{scope}", cross=cross, tuning=tuning)

