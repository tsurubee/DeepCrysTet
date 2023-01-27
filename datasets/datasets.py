import numpy as np
import torch
from torch.utils.data import Dataset


class Meshdataset(Dataset):
    def __init__(self, mesh_data, target_data, is_regression):
        self.data = mesh_data
        self.data_list = target_data["mpid"].values
        self.feature_dim = 92
        if is_regression:
            self.y = torch.tensor(target_data["target"].values).to(torch.float32)
        else:
            self.y = torch.tensor(target_data["target"].values).to(torch.long)

    def __len__(self):
        return self.y.__len__()

    def __getitem__(self, index):
        data = self.data[self.data_list[index]]
        centers = torch.tensor(data[:, :3]).to(torch.float32)
        edges = torch.tensor(data[:, 3:6]).to(torch.float32)
        corners = torch.tensor(data[:, 6:15]).to(torch.float32)
        features = torch.tensor(data[:, 15:]).to(torch.float32)
        return (centers, features, corners, edges), self.y[index]


def collate_batch(batch):
    batch_centers, batch_features, batch_corners, batch_edges = [], [], [], []
    crystal_idx, batch_target = [], []
    base_idx = 0
    for i, ((centers, features, corners, edges), targets) in enumerate(batch):
        n_i = centers.shape[0]
        batch_centers.append(centers)
        batch_features.append(features)
        batch_corners.append(corners)
        batch_edges.append(edges)
        new_idx = torch.LongTensor(np.arange(n_i) + base_idx)
        crystal_idx.append(new_idx)
        batch_target.append(targets)
        base_idx += n_i
    return (
        torch.cat(batch_centers, dim=0),
        torch.cat(batch_features, dim=0),
        torch.cat(batch_corners, dim=0),
        torch.cat(batch_edges, dim=0),
        crystal_idx,
    ), torch.stack(batch_target, dim=0)
