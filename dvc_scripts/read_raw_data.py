import os
import argparse

import yaml
import sys

# Since we are working in subfolder, make sure we add the main folder to the path
#sys.path.insert(0, os.path.dirname(__file__) + "/../../")
sys.path.insert(0, "")

from src.data.read_raw_data import get_raw_pax_data



def main():
    start_id = None
    end_id = None

    # try reading in the params file if it exists
    try:
        with open("params.yaml", "r") as f:
            params = yaml.safe_load(f)["read_raw_data"]

        start_id = int(params["start_id"])
        end_id = int(params["end_id"])

    except Exception as e:
        print(e)

    # As fallback, we can always manually provide start/end id here
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_id", metavar="ID", type=int, help="Minimum pax-id to be read")
    parser.add_argument("--end_id", metavar="ID", type=int, help="Maximum pax-id to be read")
    parser.add_argument("--output-file", metavar="file", type=str, required=True, help="Maximum pax-id to be read")

    args = parser.parse_args()

    if args.start_id is not None:
        print(f"Overruling start_id with {args.start_id}")
        start_id = args.start_id

    if args.end_id is not None:
        print(f"Overruling end_id with {args.end_id}")
        end_id = args.end_id

    if (start_id is None) or (end_id is None):
        print("ERROR: No start and end id information!!")
        sys.exit(1)

    print(f"Reading in pax info with {start_id} <= paxid <= {end_id}")
    df = get_raw_pax_data(start_id, end_id)

    # Now we write it to the place where we expect it with dvc
    output_location = os.path.dirname(__file__) + "/../../data/raw"
    os.makedirs(output_location, exist_ok=True)

    df.to_pickle(os.path.dirname(__file__) + "/../../data/raw/raw_data.pkl")


if __name__ == "__main__":
    main()



