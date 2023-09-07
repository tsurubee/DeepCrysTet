import os
import argparse
import numpy as np
from tqdm import tqdm


def main(args):
    raw_data = np.load(args.data_path)
    mpid_list = raw_data.files
    print("Number of mpid before preprocessing: {}".format(len(mpid_list)))
    data_dict = {}

    def vector_length(vec):
        return np.sqrt((vec[:, 0]) ** 2 + (vec[:, 1]) ** 2 + (vec[:, 2]) ** 2)

    for mpid in tqdm(mpid_list):
        data = raw_data[mpid]
        vertices = data[:, :9]
        n_faces = vertices.shape[0]
        # center
        centers = np.zeros((n_faces, 3), dtype=np.float16)
        centers[:, 0] = (vertices[:, 0] + vertices[:, 3] + vertices[:, 6]) / 3
        centers[:, 1] = (vertices[:, 1] + vertices[:, 4] + vertices[:, 7]) / 3
        centers[:, 2] = (vertices[:, 2] + vertices[:, 5] + vertices[:, 8]) / 3
        # corners
        corners = np.zeros((n_faces, 9), dtype=np.float16)
        corners[:, :3] = vertices[:, :3] - centers
        corners[:, 3:6] = vertices[:, 3:6] - centers
        corners[:, 6:] = vertices[:, 6:] - centers
        # Lengths of the edges
        edges = np.zeros((n_faces, 3), dtype=np.float16)
        edges[:, 0] = vector_length(corners[:, :3] - corners[:, 3:6])
        edges[:, 1] = vector_length(corners[:, :3] - corners[:, 6:])
        edges[:, 2] = vector_length(corners[:, 3:6] - corners[:, 6:])
        edges = np.sort(edges, axis=1)
        # move centroid of centers to (0, 0, 0)
        centroid = np.sum(centers / n_faces, axis=0)
        centers = centers - centroid
        data = np.concatenate([centers, edges, corners, data[:, 12:]], axis=1)
        if np.sum(np.isnan(data)) > 0:
            raise ValueError("Nan exist in data!")
        elif np.sum(np.isinf(data)) > 0:
            raise ValueError("Inf exist in data!")
        data_dict[mpid] = data.astype(np.float16)

    print("Number of mpid after preprocessing: {}".format(len(data_dict)))
    np.savez_compressed(os.path.join(args.save_dir, "mp-3dmesh.npz"), **data_dict)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-path",
        type=str,
        required=True,
        help="Path of the data to be preprocessed",
    )
    parser.add_argument(
        "--save-dir",
        default=".",
        type=str,
        help="Directory to save the preprocessed file (default: .)",
    )
    args = parser.parse_args()
    main(args)
