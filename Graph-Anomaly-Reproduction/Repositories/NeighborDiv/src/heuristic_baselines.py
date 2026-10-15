"""Appendix D heuristic baselines and shared median calibration."""

from __future__ import annotations

import ctypes
import hashlib
import numpy as np
import os
from pathlib import Path
import scipy.sparse as sp
import subprocess
import tempfile
from sklearn.decomposition import TruncatedSVD

from .data import binary_adjacency


def project_l1(features, rank=8, random_state=0, n_iter=5):
    """Paper-style U Sigma projection followed by row-wise L1 normalization."""
    features = sp.csr_matrix(features, dtype=np.float64)
    if features.shape[1] == 1 and rank == 1:
        projected = features.toarray()
    else:
        projector = TruncatedSVD(n_components=rank, algorithm="randomized",
                                 n_iter=n_iter, random_state=int(random_state))
        projected = projector.fit_transform(features)
    if not np.isfinite(projected).all():
        raise ValueError("Projected features contain non-finite values.")
    scale = np.max(np.abs(projected), axis=1, keepdims=True)
    scaled = np.divide(projected, scale, out=np.zeros_like(projected), where=scale > 0)
    l1 = np.abs(scaled).sum(axis=1, keepdims=True)
    return np.divide(scaled, l1, out=np.zeros_like(scaled), where=l1 > 0)


def calibrate_raw(raw, valid):
    """Appendix D median reference, absolute deviation, z-score and zero fallback."""
    raw = np.asarray(raw, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    if raw.ndim != 1 or valid.shape != raw.shape or not np.isfinite(raw).all():
        raise ValueError("Expected finite one-dimensional raw scores and matching valid mask.")
    scores = np.zeros(raw.size, dtype=np.float64)
    if not valid.any():
        return scores
    values = raw[valid]
    deviation = np.abs(values - np.median(values))
    std = float(deviation.std(ddof=0))
    if std > 0:
        scores[valid] = (deviation - deviation.mean()) / std
    return scores


def normalized_undirected(adjacency):
    adjacency = binary_adjacency(adjacency, edge_policy="undirected").astype(np.float64)
    degree = np.asarray(adjacency.sum(axis=1)).reshape(-1)
    inv = np.zeros_like(degree)
    positive = degree > 0
    inv[positive] = np.power(degree[positive], -0.5)
    normalized = sp.diags(inv).dot(adjacency).dot(sp.diags(inv)).tocsr()
    return adjacency, degree, normalized


def _pack_adjacency(adjacency):
    n = adjacency.shape[0]
    packed = np.zeros((n, (n + 7) // 8), dtype=np.uint8)
    rows, cols = adjacency.nonzero()
    shifts = np.left_shift(np.uint8(1), (cols % 8).astype(np.uint8))
    np.bitwise_or.at(packed, (rows, cols // 8), shifts)
    return packed


def _score_amen_ego_native(adjacency, attributes, degree, two_m):
    """Use a native OpenMP bitset kernel for dense graphs with huge egonets."""
    source = Path(__file__).with_name("amen_ego_native.cpp")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()[:16]
    build_dir = Path(tempfile.gettempdir()) / "neighbordiv-native"
    build_dir.mkdir(parents=True, exist_ok=True)
    library_path = build_dir / ("amen_ego_{}.so".format(digest))
    if not library_path.exists():
        subprocess.check_call([
            "g++", "-O3", "-fopenmp", "-shared", "-fPIC", str(source), "-o", str(library_path)
        ])

    n, dim = attributes.shape
    words = (n + 63) // 64
    packed = np.zeros((n, words), dtype=np.uint64)
    rows, cols = adjacency.nonzero()
    bits = np.left_shift(np.uint64(1), (cols % 64).astype(np.uint64))
    np.bitwise_or.at(packed, (rows, cols // 64), bits)

    indptr64 = np.ascontiguousarray(adjacency.indptr, dtype=np.int64)
    indices64 = np.ascontiguousarray(adjacency.indices, dtype=np.int64)
    degree64 = np.ascontiguousarray(degree, dtype=np.float64)
    attributes = np.ascontiguousarray(attributes, dtype=np.float64)
    raw = np.zeros(n, dtype=np.float64)

    pointer_i64 = ctypes.POINTER(ctypes.c_int64)
    pointer_f64 = ctypes.POINTER(ctypes.c_double)
    pointer_u64 = ctypes.POINTER(ctypes.c_uint64)
    native = ctypes.CDLL(str(library_path)).score_amen_ego
    native.argtypes = [ctypes.c_int64, ctypes.c_int64, pointer_i64, pointer_i64,
                       pointer_f64, pointer_f64, pointer_u64, pointer_f64, ctypes.c_int]
    native.restype = None
    native(n, dim,
           indptr64.ctypes.data_as(pointer_i64), indices64.ctypes.data_as(pointer_i64),
           degree64.ctypes.data_as(pointer_f64), attributes.ctypes.data_as(pointer_f64),
           packed.ctypes.data_as(pointer_u64), raw.ctypes.data_as(pointer_f64),
           min(16, max(1, os.cpu_count() or 1)))
    return raw


def score_lcc(adjacency):
    """LCC over an undirected binary graph, using packed bitset intersections."""
    adjacency = binary_adjacency(adjacency, edge_policy="undirected")
    degree = np.diff(adjacency.indptr).astype(np.int64)
    n = adjacency.shape[0]
    packed = _pack_adjacency(adjacency)
    popcount = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)
    raw = np.zeros(n, dtype=np.float64)
    valid = degree >= 2
    for node in np.flatnonzero(valid):
        neighbors = adjacency.indices[adjacency.indptr[node]:adjacency.indptr[node + 1]]
        mask = np.zeros(packed.shape[1], dtype=np.uint8)
        indices = neighbors // 8
        bits = np.left_shift(np.uint8(1), (neighbors % 8).astype(np.uint8))
        np.bitwise_or.at(mask, indices, bits)
        directed_neighbor_edges = int(popcount[np.bitwise_and(packed[neighbors], mask)].sum(dtype=np.uint64))
        raw[node] = directed_neighbor_edges / float(int(degree[node]) * (int(degree[node]) - 1))
    return calibrate_raw(raw, valid), raw, valid


def score_nrs(adjacency, projected_l1):
    adjacency, degree, normalized = normalized_undirected(adjacency)
    aggregate = normalized.dot(projected_l1)
    raw = np.linalg.norm(projected_l1 - aggregate, axis=1)
    valid = degree >= 1
    return calibrate_raw(raw, valid), raw, valid


def score_pcd(adjacency, projected_l1, hop_weights=(1.0, 0.7, 0.5)):
    adjacency, degree, normalized = normalized_undirected(adjacency)
    current = np.asarray(projected_l1, dtype=np.float64)
    raw = np.zeros(adjacency.shape[0], dtype=np.float64)
    for weight in hop_weights:
        following = normalized.dot(current)
        left_norm = np.linalg.norm(current, axis=1)
        right_norm = np.linalg.norm(following, axis=1)
        denominator = left_norm * right_norm
        cosine = np.divide(np.einsum("ij,ij->i", current, following), denominator,
                           out=np.zeros_like(denominator), where=denominator > 0)
        raw += float(weight) * (1.0 - np.clip(cosine, -1.0, 1.0))
        current = following
    valid = degree >= 1
    return calibrate_raw(raw, valid), raw, valid


def score_amen_ego(adjacency, projected_l1, max_nodes=None):
    """Appendix D AMEN-Ego score with the AMEN size-invariant rescaling.

    `max_nodes` supports timing/profiling only. A result with a limit is partial
    and must not be used as a dataset metric.
    """
    adjacency, degree, _ = normalized_undirected(adjacency)
    n, dim = projected_l1.shape
    lo = projected_l1.min(axis=0)
    span = projected_l1.max(axis=0) - lo
    attributes = np.divide(projected_l1 - lo, span, out=np.zeros_like(projected_l1), where=span > 0)
    two_m = float(degree.sum())
    if two_m == 0:
        return np.zeros(n), np.zeros(n), np.zeros(n, dtype=bool)

    raw = np.zeros(n, dtype=np.float64)
    valid = degree >= 1
    # The direct ego-by-ego Python path has O(sum_i sum_{p in C_i} d_p)
    # work. On T-Finance this exceeds 190 billion edge visits, so use the
    # equivalent native bitset intersection kernel when a local compiler is
    # available. Small and medium graphs remain on the easy-to-audit Python path.
    work_estimate = int(np.dot(degree, degree + 1.0))
    if max_nodes is None and work_estimate >= 1000000000:
        raw = _score_amen_ego_native(adjacency, attributes, degree, two_m)
        return calibrate_raw(raw, valid), raw, valid

    eligible = np.flatnonzero(valid)
    if max_nodes is not None:
        if isinstance(max_nodes, bool) or not isinstance(max_nodes, (int, np.integer)) or max_nodes < 1:
            raise ValueError("max_nodes must be a positive integer.")
        eligible = eligible[:int(max_nodes)]
        valid[:] = False
        valid[eligible] = True
    member = np.zeros(n, dtype=bool)
    for ego in eligible:
        nbrs = adjacency.indices[adjacency.indptr[ego]:adjacency.indptr[ego + 1]]
        community = np.concatenate((np.asarray([ego], dtype=np.int64), nbrs))
        member[community] = True
        local_degree = degree[community]
        weighted_degree_features = np.sum(local_degree[:, None] * attributes[community], axis=0)
        null_term = np.square(weighted_degree_features) / two_m
        internal_observed = np.zeros(dim, dtype=np.float64)
        internal_weighted = np.zeros(dim, dtype=np.float64)
        external_weighted = np.zeros(dim, dtype=np.float64)
        for p in community:
            q = adjacency.indices[adjacency.indptr[p]:adjacency.indptr[p + 1]]
            inside = member[q]
            q_inside = q[inside]
            if q_inside.size:
                expected = np.minimum(1.0, degree[p] * degree[q_inside] / two_m)
                product = attributes[p][None, :] * attributes[q_inside]
                internal_observed += product.sum(axis=0)
                internal_weighted += ((1.0 - expected)[:, None] * product).sum(axis=0)
            q_outside = q[~inside]
            if q_outside.size:
                expected = np.minimum(1.0, degree[p] * degree[q_outside] / two_m)
                product = attributes[p][None, :] * attributes[q_outside]
                external_weighted += ((1.0 - expected)[:, None] * product).sum(axis=0)
        member[community] = False

        # AMEN Sec. 4.2 uses the scalar worst-case Imin (similarity=1),
        # whereas null_term is the actual attribute-dependent null mass.
        null_bound = float(local_degree.sum() ** 2 / two_m)
        internal_scale = float(community.size ** 2) + null_bound
        x_i = np.divide(internal_observed - null_term + null_bound, internal_scale,
                        out=np.zeros(dim, dtype=np.float64), where=internal_scale > 0)
        x_e = np.divide(-external_weighted, internal_weighted + external_weighted,
                        out=np.zeros(dim, dtype=np.float64), where=(internal_weighted + external_weighted) > 0)
        x = x_i + x_e
        positive = np.maximum(x, 0.0)
        normality = float(np.linalg.norm(positive)) if np.any(positive > 0) else float(x.max())
        raw[ego] = -normality
    return calibrate_raw(raw, valid), raw, valid


def score_baseline(name, adjacency, projected_l1, max_nodes=None):
    if name == "LCC":
        return score_lcc(adjacency)
    if name == "NRS":
        return score_nrs(adjacency, projected_l1)
    if name == "PCD":
        return score_pcd(adjacency, projected_l1)
    if name == "AMEN-Ego":
        return score_amen_ego(adjacency, projected_l1, max_nodes=max_nodes)
    raise ValueError("Unknown heuristic baseline: {}".format(name))
