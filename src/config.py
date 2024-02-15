import sys
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    ExtraTreesRegressor,
)

from xgboost import XGBRegressor

columns_to_dummy = ["Company", "TypeName", "CPU_BRAND", "GPU_BRAND", "OpSys", "Memory_Type"]
companies_to_agg = ["Razer", "Mediacom", "Microsoft", "Xiaomi", "Vero", "Chuwi", "Google", 'Fujitsu', 'LG','Huawei']

MODEL_CONFIG = {
    "SEED": 1,
    "TEST_SIZE": 0.2,
    "TARGET": "Price_euros",
    "MODEL_NAME": "laptop_prices"
}

GRIDS = {
    "RF": {
        "n_estimators": [
            100,
            200,
            300,
            400,
            500,
            700,
            800,
            1000,
            1300,
            1500,
            1700,
            1900,
            2000,
        ],
        "max_features": ["auto", "sqrt"],
        "max_depth": [None, 10, 20, 30, 50, 70, 90, 100, 110],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "bootstrap": [True, False],
    },
    "GBOOST": {
        "n_estimators": [100, 500, 1000, 1500],
        "learning_rate": [0.01, 0.02, 0.03, 0.04],
        "subsample": [0.9, 0.5, 0.2, 0.1],
        "max_depth": [None, 4, 6, 8, 10],
    },
    "XGBOOST": {
        "n_estimators": [10, 50, 100, 200, 400, 600, 800, 1000],
        "max_depth": [2, 8, 16, 32, 50],
        "min_samples_split": [2, 4, 6],
        "min_samples_leaf": [1, 2, 3],
        "max_features": ["auto", "sqrt", "log2"],
    },
    "XTREES": {
        "n_estimators": [10, 50, 100, 200, 400, 600, 800, 1000],
        "max_depth": [2, 8, 16, 32, 50],
        "min_samples_split": [2, 4, 6],
        "min_samples_leaf": [1, 2],
        "max_features": ["auto", "sqrt", "log2"],
        "bootstrap": [True, False],
    },
}

MODELS = {
    "RF": RandomForestRegressor(),
    "GBOOST": GradientBoostingRegressor(),
    "XGBOOST": XGBRegressor(),
    "XTREES": ExtraTreesRegressor(),
}