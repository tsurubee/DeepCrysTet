import pymatgen.core as mg
import numpy as np
from atom_init import Atom_init
from bokeh.sampledata.periodic_table import elements


class Features:
    def get_comp_dict(self, composition):
        try:
            return dict(
                mg.Composition(composition).fractional_composition.get_el_amt_dict()
            )
        except:
            return {}

    def get_ave_atom_init(self, composition):
        comp_dict = self.get_comp_dict(composition)
        atom_init = Atom_init.cgcnn_atom_init()
        el_dict = dict(elements[["symbol", "atomic number"]].values)
        if len(comp_dict) > 1:
            tmp = np.array([0.0] * len(atom_init[1]))
            for el, frac in comp_dict.items():
                tmp += frac * np.array(atom_init[el_dict[el]])
            return tmp
        elif len(comp_dict) == 1:
            return np.array(atom_init[el_dict[list(comp_dict.keys())[0]]])
        else:
            # Elements with atomic number more than 100
            return np.array([np.nan] * len(atom_init[1]))

    def get_delaunay_feature(self, pts, ijks, atom_species):
        mesh_data = []
        for ijk in ijks:
            mesh_tmp = np.concatenate(
                [
                    pts[ijk[0]].astype(np.float32),
                    pts[ijk[1]].astype(np.float32),
                    pts[ijk[2]].astype(np.float32),
                    ijk.astype(np.float32),
                    self.get_ave_atom_init(
                        atom_species[ijk[0]]
                        + atom_species[ijk[1]]
                        + atom_species[ijk[2]]
                    ).astype(np.float32),
                ]
            )
            mesh_data.append(mesh_tmp)
        return np.array(mesh_data)

    def create_datasets(self, idx, delaunay_alldata):
        pts, ijks, atom_species = delaunay_alldata
        mesh_data = self.get_delaunay_feature(pts, ijks, atom_species)
        mesh_dict = {}
        mesh_dict[idx] = mesh_data
        if np.isnan(mesh_data.flatten()).sum() == 0:
            return mesh_dict
