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
        compdict = self.get_comp_dict(composition)

        atom_init = Atom_init.cgcnn_atom_init()
        el_dict = dict(elements[["symbol", "atomic number"]].values)

        try:
            if len(compdict) > 1:
                tmp = 0
                for el, frac in compdict.items():
                    tmp += frac * np.array(atom_init[el_dict[el]])  # /len(compdict)
                return np.array(tmp)
            elif len(compdict) == 1:
                tmp = atom_init[el_dict[list(compdict.keys())[0]]]
                return np.array(tmp)
        except:
            # Elements with atomic number more than 100
            return np.array([np.nan] * len(atom_init[1]))

    def get_delaunay_feature(self, pts, ijks, atom_species):
        mesh_tmp = pts[ijks[0][0]].astype(np.float32)
        mesh_tmp = np.append(mesh_tmp, pts[ijks[0][1]].astype(np.float32), axis=0)
        mesh_tmp = np.append(mesh_tmp, pts[ijks[0][2]].astype(np.float32), axis=0)

        # Index of points creating a triangular surface
        mesh_tmp = np.append(mesh_tmp, ijks[0].astype(np.float32), axis=0)

        comp = (
            atom_species[ijks[0][0]]
            + atom_species[ijks[0][1]]
            + atom_species[ijks[0][2]]
        )

        feature = self.get_ave_atom_init(comp).astype(np.float32)
        mesh_tmp = np.append(mesh_tmp, feature, axis=0)

        mesh_data = mesh_tmp.reshape(1, len(mesh_tmp))

        for idx, ijk in enumerate(ijks[1:]):
            mesh_tmp = pts[ijk[0]].astype(np.float32)
            mesh_tmp = np.append(mesh_tmp, pts[ijk[1]].astype(np.float32), axis=0)
            mesh_tmp = np.append(mesh_tmp, pts[ijk[2]].astype(np.float32), axis=0)

            mesh_tmp = np.append(mesh_tmp, ijk.astype(np.float32), axis=0)

            comp = atom_species[ijk[0]] + atom_species[ijk[1]] + atom_species[ijk[2]]

            feature = self.get_ave_atom_init(comp).astype(np.float32)
            mesh_tmp = np.append(mesh_tmp, feature, axis=0)

            mesh_data = np.append(mesh_data, mesh_tmp.reshape(1, len(mesh_tmp)), axis=0)

        return mesh_data

    def create_datasets(self, idx, delaunay_alldata):
        pts, ijks, _, atom_species, _, _, _ = delaunay_alldata
        mesh_data = self.get_delaunay_feature(pts, ijks, atom_species)
        mesh_dict = {}
        mesh_dict[idx] = mesh_data
        if np.isnan(mesh_data.flatten()).sum() == 0:
            return mesh_dict
