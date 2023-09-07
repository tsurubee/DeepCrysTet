import os
import pandas as pd
from matminer.datasets import load_dataset
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer


class Dataset:
    CRYSTAL_SYSTEMS = [
        "triclinic",
        "monoclinic",
        "orthorhombic",
        "tetragonal",
        "trigonal",
        "hexagonal",
        "cubic",
    ]

    def __init__(self, dataset_name=None, save_dir=None):
        self.dataset_name = dataset_name
        self.save_dir = save_dir
        self.csdict = dict(zip(self.CRYSTAL_SYSTEMS, range(len(self.CRYSTAL_SYSTEMS))))

    def get_alldata(self):
        data_path = os.path.join(self.save_dir, self.dataset_name + ".pkl")
        if os.path.exists(data_path):
            print("Loading data")
            return pd.read_pickle(data_path)

        os.makedirs(self.save_dir, exist_ok=True)
        print("Downloading data")
        df_data = load_dataset(self.dataset_name)
        print("Download completed")
        df_data.to_pickle(data_path)
        return df_data

    def get_cs_sg(self, structure):
        sa_structure = SpacegroupAnalyzer(structure)
        cs_num = self.csdict[sa_structure.get_crystal_system()]
        sg_num = sa_structure.get_space_group_number() - 1

        return {"cs": cs_num, "sg": sg_num}
