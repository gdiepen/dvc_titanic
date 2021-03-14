
import os
import pandas as pd
import sys


def get_raw_pax_data(start_id, end_id):
    """Read in the data from our 'raw data source' that is in the CSV file

    The start_id and end_id will allow us to only read in a subset of the data
    """
    input_file = os.path.dirname(__file__) + "/../../data/raw_train_file/train.csv"

    df = pd.read_csv(input_file)

    df = df.loc[(df.PassengerId >= start_id) & (df.PassengerId <= end_id), :].copy()

    return df
