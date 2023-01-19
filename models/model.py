import torch
import torch.nn as nn
import torch.nn.functional as F


class DeepCrysTet(nn.Module):
    def __init__(self, feature_dim, is_regression):
        super(DeepCrysTet, self).__init__()
        self.crystalfacenet = CrystalFaceNet(feature_dim)
        self.is_regression = is_regression
        if is_regression:
            self.predict_mlp = nn.Sequential(
                nn.Linear(1024, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Dropout(p=0.2),
                nn.Linear(512, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Dropout(p=0.2),
                nn.Linear(256, 1),
            )
        else:
            self.predict_mlp = nn.Sequential(
                nn.Linear(1024, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Dropout(p=0.2),
                nn.Linear(512, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Dropout(p=0.2),
                nn.Linear(256, 230),
            )
            self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, X):
        X = self.crystalfacenet(X)
        X = self.predict_mlp(X)
        if self.is_regression:
            return X
        else:
            return self.logsoftmax(X)


class CrystalFaceNet(nn.Module):
    def __init__(self, feature_dim):
        super(CrystalFaceNet, self).__init__()
        self.conv1 = nn.Conv1d(256, 512, 1)
        self.conv2 = nn.Conv1d(512, 1024, 1)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(1024)
        self.centers_mlp = nn.Sequential(
            nn.Conv1d(3, 16, 1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Conv1d(16, 32, 1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
        )
        self.edges_mlp = nn.Sequential(
            nn.Conv1d(3, 16, 1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Conv1d(16, 32, 1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
        )
        self.features_mlp = nn.Sequential(
            nn.Conv1d(feature_dim, feature_dim, 1),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(),
            nn.Conv1d(feature_dim, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )
        self.FRC = FaceRotateConvolution()

    def forward(self, X):
        centers, features, corners, edges, crystal_idx = X
        centers = self.centers_mlp(torch.unsqueeze(centers.transpose(1, 0), 0))
        features = self.features_mlp(torch.unsqueeze(features.transpose(1, 0), 0))
        corners = self.FRC(torch.unsqueeze(corners.transpose(1, 0), 0))
        edges = self.edges_mlp(torch.unsqueeze(edges.transpose(1, 0), 0))
        X = torch.cat([centers, edges, corners, features], 1)
        X = F.relu(self.bn1(self.conv1(X)))
        X = self.bn2(self.conv2(X))
        X = torch.squeeze(X).transpose(1, 0)
        X = self.max_pooling(X, crystal_idx)
        return X

    def max_pooling(self, X, crystal_idx):
        return torch.cat(
            [torch.max(X[idx_map], dim=0, keepdim=True)[0] for idx_map in crystal_idx], dim=0
        )


class FaceRotateConvolution(nn.Module):
    def __init__(self):
        super(FaceRotateConvolution, self).__init__()
        self.rotate_mlp = nn.Sequential(
            nn.Conv1d(6, 32, 1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 32, 1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
        )
        self.fusion_mlp = nn.Sequential(
            nn.Conv1d(32, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )

    def forward(self, corners):
        fea = (
            self.rotate_mlp(corners[:, :6])
            + self.rotate_mlp(corners[:, 3:9])
            + self.rotate_mlp(torch.cat([corners[:, 6:], corners[:, :3]], 1))
        ) / 3
        return self.fusion_mlp(fea)
