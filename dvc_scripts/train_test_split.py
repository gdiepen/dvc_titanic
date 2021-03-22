"""Module for the train/test split stage functions"""
import argparse
import os
import pandas as pd
import sys
import yaml
import importlib

# Since we are working in subfolder, make sure we add the main folder to the path
sys.path.insert(0, "")

from src.train_test_split.train_test_split import split_input_into_train_test

def get_dvc_stage_info(deps, outputs, metrics, params, all_args):
    """Build up the commandline that should be added after dvc run -n STAGE
    for the provided dependencies, outputs, metrics, parameters, and arguments
    provided to the script.

    This function will add all python script source files to the list of dependencies
    that are included during the runtime

    Args:
        deps: list of manual dependencies (i.e. not code) that should be added with a
            -d argument to the dvc run commandline. Typically these will be the input
            files that this script depends on, as the python scripts will be
            automatically determined
        outputs: list of all the output files that should be added with a -o argument
            to the dvc run commandline
        metrics: list of the metrics files that should be added with a -M argument
            to the dvc run commandline
        params: list of the parameters that should be added with a -p argument to the
            dvc run commandline.
        all_args: List of all of the commandline arguments that were given to this
            dvc_script script. All of these arguments are used to build up the final
            cmd provided as argument to the dvc run commandline

    Returns:
        String holding the complete text that should be added directly after a
        dvc run -n STAGE_NAME
        run
    """

    python_deps = []
    _modules = sorted(list(set(sys.modules)))

    # For each unique module, we try to import it and try to get the
    # file in which this is defined. For some of the builtins, this will 
    # fail, hence the except 
    for i in _modules:
        imported_lib = importlib.import_module(i)
        try:
            if not os.path.relpath(imported_lib.__file__, "").startswith(".."):
                python_deps.append(os.path.relpath(imported_lib.__file__, ""))
        except AttributeError:
            pass
        except ValueError:
            pass

    # Now create unique sorted list and add concatenate the user based deps
    # and the python module deps
    python_deps = sorted(list(set(python_deps)))
    all_deps = deps + python_deps

    # Start building the dvc run commandline
    # 1. Add all dependencies
    ret = ""
    for i in all_deps:
        ret += f"  -d {i} \\\n"
    ret += f"  \\\n"

    # 2. Add all outputs
    for i in outputs:
        ret += f"  -o {i} \\\n"
    ret += f"  \\\n"

    # 3. Add all parameters
    ret += f"  -p {','.join(params)} \\\n  \\\n"

    # 4. Add all metrics
    for i in metrics:
        ret += f"  -M {i} \\\n"
    ret += f"  \\\n"

    # We want to create newlines at every new argument that was provided to the script
    all_args = map(lambda x: f"\\\n    {x}" if x[0] == "-" else x, all_args)

    # Now build up the final string
    ret += "  python " + " ".join(all_args) 
    return ret

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
    parser.add_argument(
        "--show-dvc-add-command",
        action="store_true",
        help="After the script is done, it will show the command you will need to use for adding stage",
    )



    args = parser.parse_args()

    df = pd.read_pickle(args.raw_input_file)
    df_train, df_test = split_input_into_train_test(df, random_state, test_size)

    df_train.to_pickle(args.train_output_file)
    df_test.to_pickle(args.test_output_file)

    if args.show_dvc_add_command:
        deps = [args.raw_input_file]
        outputs = [args.train_output_file, args.test_output_file]
        metrics = []
        params = ["train_test_split"]

        all_args = [x for x in sys.argv if x != "--show-dvc-add-command"]
        dvc_cmdline = get_dvc_stage_info(deps, outputs, metrics, params, all_args)

        print()
        print()
        print("Please copy paste the items below AFTER command to create stage:")
        print("    dvc run -n STAGE_NAME \\")
        print()
        print(dvc_cmdline)


if __name__ == "__main__":
    main()
