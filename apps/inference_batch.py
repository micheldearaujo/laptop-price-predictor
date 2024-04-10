# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC # Inference Pipeline

# COMMAND ----------

import sys
import os

sys.path.append(os.path.abspath(".."))

from src.models.predict import all_models_predict
from src import config
import pandas as pd

# COMMAND ----------

src = "_dummy" # can be "" or "_dummy"
inference_df = all_models_predict(scope=src)

# COMMAND ----------


