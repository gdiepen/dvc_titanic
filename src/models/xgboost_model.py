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

import xgboost as xgb 
from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

class IdentityTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    
    def fit(self, input_array, y=None):
        return self
    
    def transform(self, input_array, y=None):
        return input_array*1


def fit_model(X, y, classifier_settings={}, fit_settings={}):
    numeric_transformer = Pipeline(
        steps=[("identity", IdentityTransformer())]
    )

    categorical_transformer = Pipeline(
        steps=[
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


    xgb_clf = xgb.sklearn.XGBClassifier(**classifier_settings)

    clf = Pipeline(
        steps=[("preprocessor", preprocessor), ("classifier", xgb_clf)]
    )

    clf.fit(X, y, **fit_settings)

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
