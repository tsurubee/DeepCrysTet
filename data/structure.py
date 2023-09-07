import numpy as np
from scipy.spatial import Delaunay
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
import itertools
from bokeh.sampledata.periodic_table import elements

elcolor = dict(zip(elements["atomic number"].values, elements["CPK"].values))


class Structure:
    def get_structure(self, structure, is_primitive=False, scale=1):
        structure_tmp = structure.copy()
        sa = SpacegroupAnalyzer(structure_tmp)
        structure = (
            sa.get_primitive_standard_structure()
            if is_primitive
            else sa.get_refined_structure()
        )
        structure.make_supercell([scale, scale, scale])
        return structure

    def get_delaunay_multipro(self, idx, structure, is_primitive, scale):
        delaunay = self.get_delaunay(structure, is_primitive, scale)
        return {idx: list(delaunay.values())}

    def my_round(self, val, digit=2):
        p = 10 ** digit
        return (val * p * 2 + 1) // 2 / p

    def get_round(self, arr, digit=2):
        return np.array([self.my_round(val, digit) for val in arr])

    def get_delaunay(self, structure, is_primitive=False, scale=1):
        structure_tmp = self.get_structure(structure, is_primitive, scale)
        structure_tmp.make_supercell([5, 5, 5])
        xyz_list = [
            site["xyz"] for site in structure_tmp.as_dict()["sites"]
        ]  # Information on each site in the crystal structure
        label_list = [
            site["species"][0]["element"] for site in structure_tmp.as_dict()["sites"]
        ]
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

        viz_points = []
        atom_species = []
        pidx_dict = {}
        for i, pidx in enumerate(np.sort(pidxs)):
            viz_points.append(points_all[pidx])
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
            "atom_species": atom_species,
        }
