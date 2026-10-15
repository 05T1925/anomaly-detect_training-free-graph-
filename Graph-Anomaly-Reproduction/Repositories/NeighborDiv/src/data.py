"""Local graph readers; no dataset downloading or label-driven preprocessing."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import shutil
import tempfile
from typing import Optional
import zipfile

import numpy as np
import scipy.io
import scipy.sparse as sp


@dataclass
class GraphData:
    adjacency: sp.csr_matrix
    features: sp.csr_matrix
    labels: Optional[np.ndarray]
    keys: dict[str, str]
    input_info: dict = field(default_factory=dict)


def _pick_key(contents, requested, aliases, role):
    if requested is not None:
        if requested not in contents:
            raise KeyError("Requested {} key {!r} was not found.".format(role, requested))
        return requested, contents[requested]
    for key in aliases:
        if key in contents:
            return key, contents[key]
    raise KeyError("Cannot find {} key. Tried: {}".format(role, ", ".join(aliases)))


def binary_adjacency(matrix, edge_policy="preserve"):
    """Canonical binary CSR with distinct neighbors and no diagonal entries.

    Preserve uses outgoing row neighbors, incoming transposes the stored
    adjacency, and undirected takes the union. Select explicitly for each
    experiment: the Yandex loader and NeighborDiv Table 16 differ.
    """
    if edge_policy not in ("preserve", "incoming", "undirected"):
        raise ValueError("edge_policy must be preserve, incoming or undirected.")
    matrix = sp.csr_matrix(matrix, copy=True)
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Adjacency must be square, got {}.".format(matrix.shape))
    if not np.isfinite(matrix.data).all():
        raise ValueError("Adjacency contains non-finite values.")
    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    matrix.data = np.ones(matrix.nnz, dtype=np.bool_)
    matrix.setdiag(False)
    matrix.eliminate_zeros()
    if edge_policy == "undirected":
        matrix = matrix.maximum(matrix.T).tocsr()
    elif edge_policy == "incoming":
        matrix = matrix.T.tocsr()
    matrix.sort_indices()
    return matrix


def binary_labels(raw, n_nodes, allow_one_hot=False):
    """Validate labels before conversion: 0 normal, 1 anomalous.

    BWGNN T-Finance publishes an (N, 2) one-hot label matrix. Decode that
    documented representation only in the DGL reader; do not guess which
    column/class of an arbitrary dataset is anomalous.
    """
    values = raw.toarray() if sp.issparse(raw) else np.asarray(raw)
    if allow_one_hot and values.shape == (n_nodes, 2):
        if not np.isin(values, (0, 1)).all() or not (values.sum(axis=1) == 1).all():
            raise ValueError("Two-column labels must be binary one-hot rows.")
        values = values.argmax(axis=1)
    elif values.shape not in ((n_nodes,), (n_nodes, 1), (1, n_nodes)):
        raise ValueError("Expected one binary label per node; got {}.".format(values.shape))
    values = values.reshape(-1)
    if not np.isin(values, (0, 1)).all():
        raise ValueError("Anomaly labels must be 0/1; class labels and fractional values are not accepted.")
    return values.astype(np.int64, copy=False)


def _make_graph(raw_a, raw_x, raw_y, keys, edge_policy, info, one_hot=False):
    raw_a = sp.csr_matrix(raw_a)
    raw_a.sum_duplicates()
    raw_a.eliminate_zeros()
    features = sp.csr_matrix(raw_x, dtype=np.float64)
    if features.shape[0] == 0 or features.shape[1] == 0:
        raise ValueError("Features must have at least one node and one column.")
    if not np.isfinite(features.data).all():
        raise ValueError("Features contain non-finite values.")
    adjacency = binary_adjacency(raw_a, edge_policy)
    if adjacency.shape[0] != features.shape[0]:
        raise ValueError("Feature rows and adjacency size do not match.")
    labels = None if raw_y is None else binary_labels(raw_y, features.shape[0], one_hot)
    info = dict(info)
    info.update({
        "edge_policy": edge_policy,
        "raw_adjacency_entries": int(raw_a.nnz),
        "raw_self_loop_nodes": int(np.count_nonzero(raw_a.diagonal())),
        "processed_adjacency_entries": int(adjacency.nnz),
        "binarized": True,
        "self_loops_removed": True,
        "node_order": "original input row order",
        "label_convention": "0=normal, 1=anomalous" if labels is not None else "not loaded",
    })
    return GraphData(adjacency, features, labels, keys, info)


def load_mat_graph(path, adjacency_key=None, feature_key=None, label_key=None,
                   edge_policy="preserve", load_labels=True):
    contents = scipy.io.loadmat(str(path))
    a_key, raw_a = _pick_key(contents, adjacency_key, ("Network", "A", "adj", "Adj"), "adjacency")
    x_key, raw_x = _pick_key(contents, feature_key, ("Attributes", "X", "features", "Features"), "feature")
    raw_y, y_key = None, ""
    if load_labels and (label_key is not None or any(k in contents for k in ("Label", "gnd", "labels", "y"))):
        y_key, raw_y = _pick_key(contents, label_key, ("Label", "gnd", "labels", "y"), "label")
    return _make_graph(raw_a, raw_x, raw_y,
                       {"adjacency": a_key, "features": x_key, "labels": y_key},
                       edge_policy, {"format": "mat"})


def load_npz_graph(path, feature_key=None, label_key=None, edge_policy="undirected", load_labels=True):
    """Read the Yandex heterophilous-graphs release schema, not arbitrary NPZ."""
    with np.load(str(path), allow_pickle=False) as contents:
        x_key, raw_x = _pick_key(contents, feature_key, ("node_features",), "feature")
        _, edges = _pick_key(contents, "edges", (), "edge list")
        edges = np.asarray(edges)
        if edges.ndim != 2 or edges.shape[1] != 2 or not np.issubdtype(edges.dtype, np.integer):
            raise ValueError("NPZ edges must be an integer array with shape (M, 2).")
        n_nodes = raw_x.shape[0]
        if edges.size and (edges.min() < 0 or edges.max() >= n_nodes):
            raise ValueError("NPZ edge endpoints are outside the feature row range.")
        adjacency = sp.csr_matrix((np.ones(edges.shape[0], dtype=np.bool_),
                                   (edges[:, 0], edges[:, 1])), shape=(n_nodes, n_nodes))
        raw_y, y_key = None, ""
        if load_labels:
            y_key, raw_y = _pick_key(contents, label_key, ("node_labels",), "label")
        info = {"format": "yandex_npz", "stored_edge_rows": int(edges.shape[0]),
                "classification_split_masks": "ignored; score/evaluate on the full graph"}
        return _make_graph(adjacency, raw_x, raw_y,
                           {"adjacency": "edges", "features": x_key, "labels": y_key}, edge_policy, info)


def load_dgl_graph(path, feature_key=None, label_key=None, edge_policy="preserve", load_labels=True):
    """Optional DGL reader. Import torch/DGL only when this format is used."""
    try:
        import dgl
    except ImportError as exc:
        raise ImportError("DGL input requires a compatible torch/DGL environment; MAT/NPZ do not.") from exc
    graphs, _ = dgl.load_graphs(str(path))
    if len(graphs) != 1 or not graphs[0].is_homogeneous:
        raise ValueError("Expected one homogeneous DGL graph; conversion must be specified for other inputs.")
    graph = graphs[0]
    x_key, raw_x = _pick_key(graph.ndata, feature_key, ("feature", "feat", "features"), "feature")
    raw_y, y_key = None, ""
    label_encoding = "not loaded"
    if load_labels:
        y_key, label_tensor = _pick_key(graph.ndata, label_key, ("label", "labels", "y"), "label")
        raw_y = label_tensor.detach().cpu().numpy()
        label_encoding = "two-column one-hot -> argmax" if raw_y.shape == (graph.num_nodes(), 2) else "binary vector"
    source, target = graph.edges(order="eid")
    adjacency = sp.csr_matrix((np.ones(graph.num_edges(), dtype=np.bool_),
                               (source.cpu().numpy(), target.cpu().numpy())),
                              shape=(graph.num_nodes(), graph.num_nodes()))
    return _make_graph(adjacency, raw_x.detach().cpu().numpy(), raw_y,
                       {"adjacency": "DGL edges", "features": x_key, "labels": y_key}, edge_policy,
                       {"format": "dgl", "stored_edge_rows": int(graph.num_edges()),
                        "label_encoding": label_encoding}, one_hot=True)


def load_graph(path, data_format="auto", adjacency_key=None, feature_key=None,
               label_key=None, edge_policy="auto", load_labels=True):
    path = Path(path)
    if data_format == "auto":
        data_format = {".mat": "mat", ".npz": "npz", ".bin": "dgl", ".zip": "dgl-zip"}.get(path.suffix.lower())
        if data_format is None:
            raise ValueError("Unknown file suffix. Use --format dgl for extensionless DGL files.")
    if edge_policy == "auto":
        edge_policy = "undirected" if data_format == "npz" else "preserve"
    if adjacency_key is not None and data_format != "mat":
        raise ValueError("--adjacency-key is only valid for MAT files.")
    if data_format == "mat":
        return load_mat_graph(path, adjacency_key, feature_key, label_key, edge_policy, load_labels)
    if data_format == "npz":
        return load_npz_graph(path, feature_key, label_key, edge_policy, load_labels)
    if data_format == "dgl":
        return load_dgl_graph(path, feature_key, label_key, edge_policy, load_labels)
    if data_format == "dgl-zip":
        # Stream only the expected member to a temporary file. Never extract
        # arbitrary archive paths into the shared data directory.
        with zipfile.ZipFile(path) as archive:
            members = [m for m in archive.infolist() if not m.is_dir() and Path(m.filename).name == "tfinance"]
            if len(members) != 1:
                raise ValueError("DGL ZIP input must contain exactly one file named tfinance.")
            with tempfile.TemporaryDirectory(prefix="neighbordiv-tfinance-") as tmp:
                extracted = Path(tmp) / "tfinance"
                with archive.open(members[0]) as src, extracted.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
                graph = load_dgl_graph(extracted, feature_key, label_key, edge_policy, load_labels)
                graph.input_info.update({"format": "dgl-zip", "archive_member": members[0].filename,
                                         "extracted_bytes": members[0].file_size})
                return graph
    raise ValueError("Unsupported graph format: {}".format(data_format))
