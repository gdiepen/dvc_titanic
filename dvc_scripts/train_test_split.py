"""Module for the train/test split stage functions"""
import argparse
import os
import pandas as pd
import sys
import yaml

# Since we are working in subfolder, make sure we add the main folder to the path
sys.path.insert(0, "")

from src.train_test_split.train_test_split import split_input_into_train_test


def main():
    random_state = None
    test_size = None

    # try reading in the params file if it exists
    try:
        with open("params.yaml", "r") as f:
            params = yaml.safe_load(f)["train_test_split"]

        random_state = int(params["random_state"])
        test_size = float(params["test_size"])

    except Exception as e:
        print(e)

    # As fallback, we can always manually provide start/end id here
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw-input-file",
        metavar="file",
        type=str,
        required=True,
        help="Name of the train output file",
    )
    parser.add_argument(
        "--train-output-file",
        metavar="file",
        type=str,
        required=True,
        help="Name of the train output file",
    )
    parser.add_argument(
        "--test-output-file",
        metavar="file",
        type=str,
        required=True,
        help="Name of the test output file",
    )

    args = parser.parse_args()

    df = pd.read_pickle(args.raw_input_file)
    df_train, df_test = split_input_into_train_test(df, random_state, test_size)

    df_train.to_pickle(args.train_output_file)
    df_test.to_pickle(args.test_output_file)


if __name__ == "__main__":
    main()
