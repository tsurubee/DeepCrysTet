import argparse
from dataset import Dataset
from structure import Structure
from feature import Features
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import numpy as np
import pandas as pd
import os
import pickle


def main(args):
    subdir_path = os.path.join(args.data_dir, args.dataset_name)
    if not os.path.exists(subdir_path):
        os.makedirs(subdir_path)

    if args.dataset_name == "test":
        dataset = Dataset(
            dataset_name="mp_all_20181018",
            datapath=subdir_path,
            filetype=args.filetype,
        )
        df_data = dataset.get_alldata()
        df_data = df_data.iloc[:10, :]
    else:
        dataset = Dataset(
            dataset_name=args.dataset_name,
            datapath=subdir_path,
            filetype=args.filetype,
        )
        df_data = dataset.get_alldata()

    # Case of mp_all_20181018
    if "mpid" in df_data.columns.values:
        # Add crystal structure information
        cs_info = []
        n_workers = None
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = [
                executor.submit(dataset.get_cs_sg, structure)
                for idx, structure in tqdm(df_data[["mpid", "structure"]].values)
            ]
            for f in tqdm(futures):
                cs_info.append(f.result())
        df_data = pd.concat(
            [
                df_data.reset_index(drop=True),
                pd.DataFrame(cs_info).reset_index(drop=True),
            ],
            axis=1,
        )

        print("Creating id_prop.csv")
        targets = [
            "gap pbe",
            "bulk modulus",
            "shear modulus",
            "e_form",
            "cs",
            "sg",
        ]
        for target in tqdm(targets):
            idprop_path = os.path.join(subdir_path, "id_prop_" + target + ".csv")
            if not os.path.exists(idprop_path):
                df_idprop = df_data[["mpid", target]].dropna()
                df_idprop.to_csv(idprop_path, header=False, index=False)

        # Set index
        df_data = df_data.set_index("mpid")

    print("Creating ID list")
    idxlist = df_data.index.values
    idxlist_path = os.path.join(subdir_path, "idlist.pkl")
    if not os.path.exists(idxlist_path):
        with open(idxlist_path, mode="wb") as f:
            pickle.dump(idxlist, f)

    print("Prepareing Delaunay data")
    IS_PRIMITIVE = False
    SCALE = 1
    n_workers = None
    structureFunc = Structure()
    delaunay_path = os.path.join(
        subdir_path, "delaunay_face_scale" + str(SCALE) + ".npz"
    )
    if not os.path.exists(delaunay_path):
        print("Creating " + delaunay_path)
        delaunay_alldata = {}
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = [
                executor.submit(
                    structureFunc.get_delaunay_multipro,
                    idx,
                    df_data.loc[idx]["structure"],
                    IS_PRIMITIVE,
                    SCALE,
                )
                for idx in tqdm(idxlist)
            ]
            for f in tqdm(futures):
                delaunay_alldata.update(f.result())
        np.savez_compressed(delaunay_path, **delaunay_alldata)
    else:
        print("Loading " + delaunay_path)
        delaunay_alldata = np.load(delaunay_path, allow_pickle=True)

    print("Prepareing feature vectors")
    featureFunc = Features()
    mesh_path = os.path.join(subdir_path, "mesh_scale" + str(SCALE) + ".npz")
    if not os.path.exists(mesh_path):
        mesh_alldata = {}
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = [
                executor.submit(featureFunc.create_datasets, idx, delaunay_alldata[idx])
                for idx in tqdm(list(delaunay_alldata.keys()))
            ]
            for f in tqdm(futures):
                mesh_alldata.update(f.result())
        print("Creating " + mesh_path)
        np.savez_compressed(mesh_path, **mesh_alldata)
    else:
        print("Loading " + mesh_path)
        mesh_alldata = np.load(mesh_path, allow_pickle=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        default="./data",
        type=str,
        help="Data directory path (default: ./data)",
    )
    parser.add_argument(
        "--dataset-name", default="mp_all_20181018", type=str, help="Dataset Name"
    )
    parser.add_argument(
        "--filetype", default="pkl", type=str, help="File type of data (csv or pkl)"
    )

    args = parser.parse_args()
    main(args)
