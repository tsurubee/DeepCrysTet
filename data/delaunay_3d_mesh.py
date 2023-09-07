import argparse
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import numpy as np
import pandas as pd
import os
from pymatgen.io.cif import CifParser
from structure import Structure
from feature import Features
from dataset import Dataset


def process_cif_file(data_dir, file):
    parser = CifParser(os.path.join(data_dir, file))
    structure = parser.get_structures()[0]
    return {"id": file.split(".")[0], "structure": structure}


def prepare_delaunay_data(idxlist, df_data, n_workers=None):
    IS_PRIMITIVE = False
    SCALE = 1
    structureFunc = Structure()
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
    return delaunay_alldata


def prepare_feature_vectors(delaunay_alldata, subdir_path, n_workers=None):
    featureFunc = Features()
    mesh_path = os.path.join(subdir_path, "delaunay_3d_mesh.npz")
    mesh_alldata = {}
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = [
            executor.submit(featureFunc.create_datasets, idx, delaunay_alldata[idx])
            for idx in tqdm(list(delaunay_alldata.keys()))
        ]
        for f in tqdm(futures):
            mesh_alldata.update(f.result())
    np.savez_compressed(mesh_path, **mesh_alldata)


def create_id_prop_csv(subdir_path, df_data):
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


def get_cs_info(dataset, df_data, n_workers=None):
    cs_info = []
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = [
            executor.submit(dataset.get_cs_sg, structure)
            for _, structure in tqdm(df_data[["mpid", "structure"]].values)
        ]
        for f in tqdm(futures):
            cs_info.append(f.result())
    return cs_info


def main(args):
    if args.data_dir != "":
        # Convert original cif files to structure objects
        subdir_path = args.save_dir
        if not os.path.exists(subdir_path):
            os.makedirs(subdir_path)
        cif_files = [f for f in os.listdir(args.data_dir) if f.endswith(".cif")]
        data_tmp = []
        print("Converting cif files to Structure objects")
        for file in tqdm(cif_files):
            data_tmp.append(process_cif_file(args.data_dir, file))

        df_data = pd.DataFrame(data_tmp).set_index("id")

    else:
        dataset = Dataset(dataset_name="mp_all_20181018", save_dir=args.save_dir)
        df_data = dataset.get_alldata()
        if args.dataset_name == "test":
            df_data = df_data.iloc[:10, :]

        subdir_path = os.path.join(args.save_dir, "mp_all_20181018")
        if not os.path.exists(subdir_path):
            os.makedirs(subdir_path)

        # Add crystal system and space group to the dataset
        cs_info = get_cs_info(dataset, df_data, n_workers=None)
        df_data = pd.concat(
            [
                df_data.reset_index(drop=True),
                pd.DataFrame(cs_info).reset_index(drop=True),
            ],
            axis=1,
        )

        print("Creating id_prop.csv")
        create_id_prop_csv(subdir_path, df_data)

        df_data = df_data.set_index("mpid")

    print("Creating ID list")
    idxlist = df_data.index.values

    print("Prepareing Delaunay data")
    delaunay_alldata = prepare_delaunay_data(idxlist, df_data, n_workers=None)

    print("Prepareing feature vectors")
    prepare_feature_vectors(delaunay_alldata, subdir_path, n_workers=None)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save-dir",
        default="./datasets",
        type=str,
        help="Directory path where data is stored (default: ./datasets)",
    )
    parser.add_argument(
        "--data-dir",
        default="",
        type=str,
        help="Directory path where the original cif files are located",
    )
    parser.add_argument(
        "--dataset-name", default="mp_all_20181018", type=str, help="Dataset Name"
    )

    args = parser.parse_args()
    main(args)
