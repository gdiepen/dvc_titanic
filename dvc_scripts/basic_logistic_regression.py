import os
import argparse
import json

import pandas as pd
import pprint
import yaml
import sys
import joblib


sys.path.insert(0, "")

from src.models.basic_logistic_regression import fit_model, get_performance


def main():
    features = None
    classifier_settings = None

    # try reading in the params file if it exists
    try:
        with open("params.yaml", "r") as f:
            params = yaml.safe_load(f)["model"]["basic_logistic_regression"]

        features = params["features"]
        classifier_settings = params["classifier_settings"]

    except Exception as e:
        print(e)

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--evaluate-test-set",
        metavar=("MODELFILE", "DATAFILE"),
        type=str,
        nargs=2,
        action="store",
        help="Apply the model saved in pkl file MODELFILE on data stored in pickle file DATAFILE",
    )

    parser.add_argument(
        "--model-output-file",
        metavar="FILE",
        type=str,
        action="store",
        help="Fit model and store fitted model in pickle file FILE",
    )
    parser.add_argument(
        "--train-input-file",
        metavar="FILE",
        type=str,
        action="store",
        help="Pickle file holding input data to fit model with",
    )
    parser.add_argument(
        "--metrics-output-file",
        metavar="FILE",
        type=str,
        action="store",
        required=True,
        help="Name of file where the metrics will be stored (training or test)",
    )
    parser.add_argument(
        "--show-dvc-add-command",
        action="store_true",
        help="After the script is done, it will show the command you will need to use for adding stage",
    )




    args = parser.parse_args()

    if (features is None) or (len(features) == 0):
        print("ERROR: Feature list is empty...")
        sys.exit(1)

    if (args.model_output_file is None) and (args.evaluate_test_set is None):
        print("ERROR: Please instruct either a evaluation or a training run")
        sys.exit(1)

    if (args.evaluate_test_set is not None) and (args.model_output_file is not None):
        print("ERROR: Cannot provide both --model-output-file and --evaluate-test-set")
        sys.exit(1)
    if (args.evaluate_test_set is not None) and (args.train_input_file is not None):
        print("ERROR: Cannot provide both --train-input-file and --evaluate-test-set")
        sys.exit(1)

    if args.evaluate_test_set is None:
        if (args.model_output_file is None) and (args.train_input_file is None):
            print("ERROR: Please provide both model-output-file and train-input-file")
            sys.exit(1)


    # Now all checking is done, we can continue with the actual calling of the
    # underlying functions
    if args.evaluate_test_set is not None:
        print("Evaluating fitted model")

        model_file, data_file = args.evaluate_test_set

        df = pd.read_pickle(data_file)
        df_X = df.loc[:, features]
        df_y = df.loc[:, "Survived"]

        clf = joblib.load(model_file)

        # Now get the performance on this data set
        metrics = get_performance(clf, df_X, df_y)

        # Ensure the folder where we store the metrics file in exists
        metrics_location = os.path.dirname(args.metrics_output_file)
        os.makedirs(metrics_location, exist_ok=True)

        with open(args.metrics_output_file, "w") as f:
            f.write(json.dumps(metrics, indent=4))

        print("Metrics on provided dataset:")
        pprint.pprint(metrics)

        if args.show_dvc_add_command:
            print()
            print()
            print("Please copy paste the items below AFTER your")
            print("    dvc run -n STAGE_NAME \\")
            print("command")
            print()
            cmdline = f"  -d {sys.argv[0]} \\\n"
            cmdline += f"  -d {args.evaluate_test_set[0]} \\\n"
            cmdline += f"  -d {args.evaluate_test_set[1]} \\\n"
            cmdline += f"  -d {os.path.relpath(fit_model.__globals__['__file__'], '.')} \\\n"
            cmdline += f"  \\\n"
            if args.metrics_output_file:
                cmdline += f"  -M {args.metrics_output_file} \\\n"

            all_args = [x for x in sys.argv if x != "--show-dvc-add-command"]
            all_args = map(lambda x: f"\\\n    {x}" if x[0] == "-" else x, all_args)


            cmdline += "  python " + " ".join(all_args) 
            

            print(cmdline)
            print()
            print()


    elif args.model_output_file is not None:
        print("Fitting model")

        df = pd.read_pickle(args.train_input_file)
        df_X = df.loc[:, features]
        df_y = df.loc[:, "Survived"]

        # Fit the model and also get the performance of the model on the training data
        clf = fit_model(df_X, df_y, classifier_settings)
        metrics = get_performance(clf, df_X, df_y)

        # Ensure the folder where we store the trained model in exists
        model_location = os.path.dirname(args.model_output_file)
        os.makedirs(model_location, exist_ok=True)

        # Ensure the folder where we store the metrics file in exists
        metrics_location = os.path.dirname(args.metrics_output_file)
        os.makedirs(metrics_location, exist_ok=True)

        with open(args.metrics_output_file, "w") as f:
            f.write(json.dumps(metrics, indent=4))

        joblib.dump(clf, args.model_output_file)

        print("Metrics on training dataset:")
        pprint.pprint(metrics)


        if args.show_dvc_add_command:
            print()
            print()
            print("Please copy paste the items below AFTER your")
            print("    dvc run -n STAGE_NAME \\")
            print("command")
            print()
            cmdline = f"  -d {sys.argv[0]} \\\n"
            cmdline += f"  -d {args.train_input_file} \\\n"
            cmdline += f"  -d {os.path.relpath(fit_model.__globals__['__file__'], '.')} \\\n"
            cmdline += f"  \\\n"
            cmdline += f"  -o {args.model_output_file} \\\n"
            cmdline += f"  \\\n"
            if args.metrics_output_file:
                cmdline += f"  -M {args.metrics_output_file} \\\n"
            
            all_args = [x for x in sys.argv if x != "--show-dvc-add-command"]
            all_args = map(lambda x: f"\\\n    {x}" if x[0] == "-" else x, all_args)


            cmdline += "  python " + " ".join(all_args) 
            

            print(cmdline)
            print()
            print()




    else:
        print("Not supposed to happen....")
        sys.exit(1)


if __name__ == "__main__":
    main()
