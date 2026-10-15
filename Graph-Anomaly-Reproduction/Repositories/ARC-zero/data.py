import os
import random

import numpy as np
import scipy.io as sio
import scipy.sparse as sp
import torch
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.random_projection import GaussianRandomProjection
from torch_geometric.data import Data

from utils import normalize_adj, sparse_mx_to_torch_sparse_tensor


ROW_NORM_DATASETS = {"Amazon", "YelpChi", "tolokers", "tfinance", "elliptic"}
NO_SELF_LOOP_NORM_DATASETS = {"YelpChi", "Facebook", "elliptic"}


def preprocess_features(features):
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.0
    r_mat_inv = sp.diags(r_inv)
    return r_mat_inv.dot(features).todense()


def feat_alignment(x, edges, dims):
    edge_src, edge_dst = edges
    num_edges = len(edge_src)

    if x.shape[1] < dims:
        projector = GaussianRandomProjection(n_components=256, random_state=0)
        x = projector.fit_transform(x.cpu().numpy())

    pca = PCA(n_components=dims, random_state=0)
    x_transformed = pca.fit_transform(x)
    x_transformed = torch.FloatTensor(x_transformed)

    x_min, _ = torch.min(x_transformed, dim=0)
    x_max, _ = torch.max(x_transformed, dim=0)
    x_scaled = (x_transformed - x_min) / (x_max - x_min + 1e-12)

    smooth_coefficients = torch.zeros(x_transformed.shape[1])
    for k in range(x_transformed.shape[1]):
        diff = x_scaled[edge_src, k] - x_scaled[edge_dst, k]
        smooth_coefficients[k] = torch.sum(diff ** 2) / max(num_edges, 1)

    _, sorted_indices = torch.sort(smooth_coefficients)
    return x_transformed[:, sorted_indices]


class GraphDataset:
    def __init__(self, dims, name, prefix="./dataset/"):
        self.name = name
        self.shot_mask = None
        self.shot_idx = None

        preprocess_filename = f"{prefix}{name}_{dims}.npz"
        if os.path.exists(preprocess_filename):
            with np.load(preprocess_filename, allow_pickle=True) as f:
                data = f["data"].item()
                feat = f["feat"]
        else:
            data = sio.loadmat(f"{prefix}{name}.mat")
            adj = data["Network"]
            feat = data["Attributes"]

            adj_sp = sp.csr_matrix(adj)
            row, col = adj_sp.nonzero()
            edge_index = torch.tensor([row, col], dtype=torch.long)

            if name in ROW_NORM_DATASETS:
                feat = preprocess_features(sp.lil_matrix(feat))
            else:
                feat = sp.lil_matrix(feat).toarray()

            feat = torch.FloatTensor(feat)
            feat = feat_alignment(feat, edge_index, dims)
            np.savez(preprocess_filename, data=data, feat=feat)

        adj = data["Network"] if "Network" in data else data["A"]
        if name in NO_SELF_LOOP_NORM_DATASETS:
            adj_norm = normalize_adj(adj)
        else:
            adj_norm = normalize_adj(adj + sp.eye(adj.shape[0]))

        adj_norm = sparse_mx_to_torch_sparse_tensor(adj_norm)
        label = data["Label"] if "Label" in data else data["gnd"]

        self.feat = feat
        self.adj = adj
        self.adj_norm = adj_norm

        ano_labels = torch.tensor(np.squeeze(np.array(label)), dtype=torch.float)
        self.graph = Data(
            x=torch.tensor(self.feat, dtype=torch.float),
            x_list=None,
            adj=self.adj_norm,
            ano_labels=ano_labels,
            shot_idx=None,
            shot_mask=None,
        )

    def kmeans_shot(self, k, seed=0):
        feat = self.feat
        if isinstance(feat, torch.Tensor):
            feat = feat.detach().cpu().numpy()

        kmeans = KMeans(n_clusters=k, random_state=seed).fit(feat)
        centers = kmeans.cluster_centers_
        distances = np.sqrt(((feat - centers[:, np.newaxis]) ** 2).sum(axis=2))

        closest_idxs = []
        for i in range(k):
            closest_idxs.append(int(np.argmin(distances[:, i])))

        shot_idx = torch.tensor(closest_idxs, dtype=torch.long)
        shot_mask = torch.zeros(self.feat.shape[0], dtype=torch.bool)
        shot_mask[shot_idx] = True

        self.graph.shot_idx = shot_idx
        self.graph.shot_mask = shot_mask

    def few_shot(self, random_shot=10, seed=101):
        num_nodes = self.graph.ano_labels.shape[0]
        all_idx = list(range(num_nodes))
        random.seed(seed)
        random.shuffle(all_idx)

        shot_idx = torch.tensor(all_idx[:random_shot], dtype=torch.long)
        shot_mask = torch.zeros(num_nodes, dtype=torch.bool)
        shot_mask[shot_idx] = True

        self.graph.shot_idx = shot_idx
        self.graph.shot_mask = shot_mask

    def propagated(self, k, device):
        x = torch.FloatTensor(self.feat).to(device)
        h_list = [x]
        adj = self.adj_norm.to(device)
        for _ in range(k):
            h_list.append(torch.spmm(adj, h_list[-1]))
        self.graph.x_list = h_list
