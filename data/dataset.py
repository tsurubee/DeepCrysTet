import os
import pandas as pd
from matminer.datasets import load_dataset
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer


class Dataset:
    def __init__(self, dataset_name=None, filetype=None, datapath=None):
        self.dataset_name = dataset_name
        self.filetype = filetype
        self.datapath = datapath

        cslist = [
            "triclinic",
            "monoclinic",
            "orthorhombic",
            "tetragonal",
            "trigonal",
            "hexagonal",
            "cubic",
        ]
        self.csdict = dict(zip(cslist, range(len(cslist))))

    def info(self):
        print("Database is '" + self.dataset_name + "'")

    def get_alldata(self):
        if self.filetype == "csv":
            data_path = os.path.join(
                self.datapath, self.dataset_name + "." + self.filetype
            )
            if not os.path.exists(data_path):
                if not os.path.exists(self.datapath):
                    os.makedirs(self.datapath)
                print("Downloading data")
                df_data = load_dataset(self.dataset_name)
                print("...")
                df_data.to_csv(data_path, index=False)
                print("Download completed")
            else:
                print("Loading data")
                df_data = pd.read_csv(data_path)
        elif self.filetype == "pkl":
            data_path = os.path.join(
                self.datapath, self.dataset_name + "." + self.filetype
            )
            if not os.path.exists(data_path):
                if not os.path.exists(self.datapath):
                    os.makedirs(self.datapath)
                print("Downloading data")
                df_data = load_dataset(self.dataset_name)
                print("...")
                df_data.to_pickle(data_path)
                print("Download completed")
            else:
                print("Loading data")
                df_data = pd.read_pickle(data_path)
        ## Other datasets using matminer will be added later
        return df_data

    def get_cs_sg(self, structure):
        sa_structure = SpacegroupAnalyzer(structure)
        sg_num = sa_structure.get_space_group_number() - 1
        cs_num = self.csdict[sa_structure.get_crystal_system()]

        return {"cs": cs_num, "sg": sg_num}
