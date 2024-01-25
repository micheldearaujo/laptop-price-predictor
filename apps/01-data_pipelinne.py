# Databricks notebook source
# MAGIC %md
# MAGIC # Laptop Prices Predictor - Data Pipeline
# MAGIC **Objective**: The purpose of this notebook this to run the data processing pipeline (extraction, preprocessing, feature engineering) for the project.

# COMMAND ----------

from src.data.download_dataset import data_extraction_orchestration
from src.preprocessing.preprocess import preprocessing_orchestration
from src.features.make_features import feature_engineering_orchestration

# COMMAND ----------

raw_prices_pd = data_extraction_orchestration()
clean_prices_pd = preprocessing_orchestration(raw_prices_pd)


# COMMAND ----------

clean_prices_pd

# COMMAND ----------


