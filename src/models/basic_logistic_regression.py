
"""Module holding the functions to read the data from CDP
At the moment, alternative implementation is to read the train.csv file from disk as
the permissions for the table are not set up correctly
"""
import argparse
import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def fit_model(X, y, classifier_settings={}, fit_settings={}):
    strategy = classifier_settings.get("numerical_impute_strategy")
    if strategy is None:
        raise Exception("Missing impute strategy for numerical features")

    numeric_transformer = Pipeline(
        steps=[("num_impute", SimpleImputer(strategy=strategy))]
    )

    strategy = classifier_settings.get("categorical_impute_strategy")
    if strategy is None:
        raise Exception("Missing impute strategy for categorical features")

    categorical_transformer = Pipeline(
        steps=[
            ("cat_impute", SimpleImputer(strategy=strategy)),
            ("ohe", OneHotEncoder(drop="if_binary", handle_unknown="error")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, make_column_selector(dtype_include=np.number)),
            (
                "cat",
                categorical_transformer,
                make_column_selector(dtype_exclude=np.number),
            ),
        ]
    )

    clf = Pipeline(
        steps=[("preprocessor", preprocessor), ("classifier", LogisticRegression())]
    )

    clf.fit(X, y)

    return clf


def get_performance(clf, df_X, y):
    y_pred = clf.predict(df_X)

    accuracy = accuracy_score(y, y_pred)
    roc_auc = roc_auc_score(y, y_pred)
    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)

    metrics = {
        "accuracy": accuracy,
        "roc_auc": roc_auc,
        "precision": precision,
        "recall": recall,
    }

    return metrics
