import numpy as np
from scipy.spatial import Delaunay
import pymatgen.core as mg
from pymatgen.ext.matproj import MPRester
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
import itertools
from bokeh.sampledata.periodic_table import elements

from element_color_schemes import ElementColorSchemes

pmg = MPRester("UP0x1rTAXR52g7pi")
elcolor = dict(zip(elements["atomic number"].values, elements["CPK"].values))


class Structure:
    def __init__(self, structure_dirpath="structures/"):
        self.structure_dirpath = structure_dirpath
        self.element_colors = ElementColorSchemes.get_element_color_schemes()

    def get_structure(self, structure, is_primitive=False, scale=1):
        structure_tmp = structure.copy()

        sa_structure = SpacegroupAnalyzer(structure_tmp)
        if is_primitive:
            structure = sa_structure.get_primitive_standard_structure()
            structure.make_supercell([scale, scale, scale])
        else:
            structure = sa_structure.get_refined_structure()
            structure.make_supercell([scale, scale, scale])
        return structure

    def get_delaunay_multipro(self, idx, structure, is_primitive, scale):
        delaunay_dict = {}
        delaunay_data = list(
            self.get_delaunay(
                structure=structure, is_primitive=is_primitive, scale=scale
            ).values()
        )
        delaunay_dict[idx] = delaunay_data
        return delaunay_dict

    def my_round(self, val, digit=2):
        p = 10 ** digit
        return (val * p * 2 + 1) // 2 / p

    def get_round(self, arr, digit=2):
        res = np.array([self.my_round(val, digit) for val in arr])
        return res

    def get_delaunay(self, structure, scale=1, is_primitive=False):
        structure_tmp = self.get_structure(structure, is_primitive, scale)

        structure_tmp.make_supercell([5, 5, 5])
        xyz_list = [
            site["xyz"] for site in structure_tmp.as_dict()["sites"]
        ]  # Information on each site in the crystal structure
        label_list = [site["label"] for site in structure_tmp.as_dict()["sites"]]
        matrix = structure_tmp.lattice.matrix
        a, b, c = self.get_round(structure_tmp.lattice.abc)

        tri = Delaunay(xyz_list)

        simplices_all = tri.simplices
        points_all = tri.points

        tol = 0.05  # Error in atomic coordinates[angstrom]
        include_idxs = []
        for i, point in enumerate(points_all):
            abc_mat = self.get_round(
                structure_tmp.lattice.get_vector_along_lattice_directions(point)
            )
            if (
                (abc_mat[0] >= (a * 2 / 5) - tol)
                and (abc_mat[1] >= (b * 2 / 5) - tol)
                and (abc_mat[2] >= (c * 2 / 5) - tol)
                and (abc_mat[0] <= (a * 3 / 5) + tol)
                and (abc_mat[1] <= (b * 3 / 5) + tol)
                and (abc_mat[2] <= (c * 3 / 5) + tol)
            ):
                include_idxs.append(i)

        ijklist = []
        pidxs = []
        for tet in simplices_all:
            if len(set(tet) & set(include_idxs)) > 0:
                for comb in itertools.combinations(tet, 3):
                    comb = np.sort(comb)
                    i = comb[0]
                    j = comb[1]
                    k = comb[2]

                    ijklist.append((i, j, k))
                    pidxs.extend((i, j, k))
                    pidxs = list(set(pidxs))

        atom_idx_dict = dict(
            zip(set(np.array(label_list)), range(len(set(np.array(label_list)))))
        )
        viz_points = []
        atoms_radius = []
        atoms_color = []
        atom_idxs = []
        atom_species = []
        pidx_dict = {}
        for i, pidx in enumerate(np.sort(pidxs)):
            viz_points.append(points_all[pidx])
            if mg.Element(label_list[pidx]).atomic_radius != None:
                atoms_radius.append(
                    mg.Element(label_list[pidx]).atomic_radius * (10 / scale)
                )
            else:
                atoms_radius.append(10 / scale)
            atoms_color.append(self.element_colors["VESTA"][label_list[pidx]])
            atom_idxs.append(atom_idx_dict[label_list[pidx]])
            atom_species.append(label_list[pidx])
            pidx_dict[pidx] = i

        viz_ijk = []
        for ijk in ijklist:
            ijk_tmp = []
            for tmp in ijk:
                ijk_tmp.append(pidx_dict[tmp])
            viz_ijk.append(tuple(ijk_tmp))

        pts = np.array(viz_points)
        ijk = np.array(list(set(viz_ijk)))

        return {
            "pts": pts,
            "ijk": ijk,
            "matrix": matrix,
            "atom_species": atom_species,
            "atoms_radius": atoms_radius,
            "atoms_color": atoms_color,
            "atom_idxs": atom_idxs,
        }
