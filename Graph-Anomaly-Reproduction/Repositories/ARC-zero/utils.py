import json
import os

import numpy as np
import torch
import scipy.sparse as sp
from sklearn.metrics import average_precision_score, roc_auc_score


def test_eval(labels, scores):
    if torch.is_tensor(labels):
        labels = labels.detach().cpu().numpy()
    if torch.is_tensor(scores):
        scores = scores.detach().cpu().numpy()

    return {
        "AUROC": roc_auc_score(labels, scores),
        "AUPRC": average_precision_score(labels, scores),
    }


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def normalize_adj(adj):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def read_json_config(model_name, shot, json_dir):
    filename = os.path.join(json_dir, f"{model_name}_{shot}.json")
    if not os.path.exists(filename):
        return None

    with open(filename, "r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return None
