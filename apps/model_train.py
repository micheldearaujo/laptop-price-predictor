# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC # Model Training Pipeline

# COMMAND ----------

import sys
import os

sys.path.append(os.path.abspath(".."))

from src.models.train import all_models_train
from src import config

# COMMAND ----------

src = "_dummy" # can be "" or "_dummy"
tuning = True
cross = False
all_models_train(scope=src, cross=cross, tuning=tuning)

# COMMAND ----------


