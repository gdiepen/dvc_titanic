"""Module for the train/test split stage functions"""
import argparse
import os
import pandas as pd
import sys
from sklearn.model_selection import train_test_split

def split_input_into_train_test(df, random_state, test_size):
    """Splitting the raw data in the data/raw/raw_input.pkl file into a separate
    train and test set. The splitting will be done stratified using the Survived
    column. The two dataframes will be written to two pickle files train.pkl and
    test.pkl in the data/train_test folder
    """
    df_train, df_test = train_test_split(df, test_size=test_size, random_state=random_state, stratify=df.Survived)

    print(f"Size train set: {len(df_train)}")
    print(f"Size test set: {len(df_test)}")

    return df_train, df_test
