import argparse
import numpy as np
import pandas as pd
import pickle


def main(args):
    filetype = args.data_dir.split(".")[1]
    if filetype == "npz":
        data = np.load(args.data_dir, allow_pickle=True)
        print(list(data.keys())[0])
        print(list(data[list(data.keys())[0]]))
    elif filetype == "csv":
        data = pd.read_csv(args.data_dir)
        print(data.values[0])
    elif filetype == "pkl":
        with open(args.data_dir, mode="rb") as f:
            data = pickle.load(f)
        print(data[0])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        default="datasets/mp_all_20181018_mesh_scale1.npz",
        type=str,
        help="Data directory path (default: ./data)",
    )

    args = parser.parse_args()
    main(args)
