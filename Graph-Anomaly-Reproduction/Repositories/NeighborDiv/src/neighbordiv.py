"""From-paper NeighborDiv implementation, arXiv:2605.20879v1, Eqs. (1)-(15).

Scoring accepts only adjacency and features. Labels are isolated in evaluation.
Full pairs are streamed in bounded chunks; no pair subsampling occurs in Full.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import scipy.sparse as sp
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import average_precision_score, roc_auc_score

# Keep the original imports/API available for existing notebooks.
from .data import GraphData, binary_adjacency, binary_labels, load_mat_graph


@dataclass
class ScoreResult:
    scores: np.ndarray
    diversity: np.ndarray
    deviation: np.ndarray
    degree: np.ndarray
    valid: np.ndarray
    metadata: dict


def _positive_integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError("{} must be a positive integer.".format(name))


def _normalize_projected(projected):
    """L1 then L2 row normalization; only actual zero rows stay zero.

    Rescale first to avoid norm overflow/underflow. Do not treat a small but
    nonzero vector as zero solely because it is below machine epsilon.
    """
    if not np.isfinite(projected).all():
        raise ValueError("SVD produced non-finite projected features.")
    scale = np.max(np.abs(projected), axis=1, keepdims=True)
    rows = np.divide(projected, scale, out=np.zeros_like(projected), where=scale > 0)
    l1_norm = np.abs(rows).sum(axis=1, keepdims=True)
    rows = np.divide(rows, l1_norm, out=np.zeros_like(rows), where=l1_norm > 0)
    l2_norm = np.sqrt(np.square(rows).sum(axis=1, keepdims=True))
    rows = np.divide(rows, l2_norm, out=np.zeros_like(rows), where=l2_norm > 0)
    return rows, int(np.count_nonzero(scale == 0))


def _unrank_pairs(degree, ranks):
    """Unrank lexicographic p<q pairs using O(len(ranks)) working memory.

    Row p begins at p*(2*d-p-1)//2. Integer binary search avoids allocating
    a degree-sized row-start table or rounding a floating-point inverse.
    """
    ranks = np.asarray(ranks, dtype=np.int64)
    lower = np.zeros(ranks.size, dtype=np.int64)
    upper = np.full(ranks.size, degree - 1, dtype=np.int64)
    while np.any(upper - lower > 1):
        middle = (lower + upper) // 2
        starts = middle * (2 * degree - middle - 1) // 2
        keep_lower = starts <= ranks
        lower = np.where(keep_lower, middle, lower)
        upper = np.where(keep_lower, upper, middle)
    starts = lower * (2 * degree - lower - 1) // 2
    return lower, lower + 1 + ranks - starts


def _sample_pair_indices(degree, count, rng):
    """Uniform subset without replacement via Floyd's O(count)-memory sampler.

    Pair IDs are sampled directly; constructing the population of all pairs
    (or a degree-sized table) would defeat the sampling efficiency variant.
    """
    total = degree * (degree - 1) // 2
    if count < 1 or count > total:
        raise ValueError("Pair sample count must be between 1 and choose(degree, 2).")
    chosen = set()
    for last in range(total - count, total):
        candidate = int(rng.integers(0, last + 1))
        chosen.add(last if candidate in chosen else candidate)
    return _unrank_pairs(degree, np.asarray(sorted(chosen), dtype=np.int64))


def _similarity_batches(neighbors, projected, max_pairs, pair_batch_size, rng):
    degree = len(neighbors)
    total = degree * (degree - 1) // 2
    if max_pairs is not None and total > max_pairs:
        rows, cols = _sample_pair_indices(degree, max_pairs, rng)
        for start in range(0, max_pairs, pair_batch_size):
            end = min(start + pair_batch_size, max_pairs)
            yield np.einsum("ij,ij->i", projected[neighbors[rows[start:end]]],
                            projected[neighbors[cols[start:end]]])
    else:
        for start in range(0, total, pair_batch_size):
            ranks = np.arange(start, min(start + pair_batch_size, total), dtype=np.int64)
            rows, cols = _unrank_pairs(degree, ranks)
            yield np.einsum("ij,ij->i", projected[neighbors[rows]], projected[neighbors[cols]])


def _pair_mean_variance(batches):
    """Merge block central moments; population variance, without cancellation.

    A degree-two node has one pair and variance zero. The paper does not
    specify ddof; an unbiased variance would be undefined for this case.
    """
    count, mean, m2 = 0, 0.0, 0.0
    for values in batches:
        block_count = int(values.size)
        block_mean = float(values.mean())
        block_m2 = float(np.square(values - block_mean).sum())
        total_count = count + block_count
        delta = block_mean - mean
        m2 += block_m2 + delta * delta * count * block_count / total_count
        mean += delta * block_count / total_count
        count = total_count
    return mean, max(0.0, m2 / count)


def _full_pair_moments(neighbor_features):
    """All-pair moments via a centered Gram identity, without sampling.

    O(d*r^2) rather than O(d^2*r) work. This optional backend can perturb
    almost-tied scores by roundoff; the streamed backend remains default.
    """
    d = len(neighbor_features)
    if d < 2:
        raise ValueError("Pair moments require at least two neighbors.")
    if d == 2:
        return float(np.dot(neighbor_features[0], neighbor_features[1])), 0.0
    if np.all(neighbor_features == neighbor_features[0]):
        return float(np.dot(neighbor_features[0], neighbor_features[0])), 0.0
    mu = neighbor_features.mean(axis=0)
    centered = neighbor_features - mu
    linear = centered.dot(mu)
    squared_norms = np.einsum("ij,ij->i", centered, centered)
    gram = centered.T.dot(centered)
    pairs = d * (d - 1) / 2.0
    mean_quadratic = -squared_norms.sum() / (2.0 * pairs)
    central_second = ((d - 2) * np.dot(linear, linear)
                      + (np.square(gram).sum() - np.dot(squared_norms, squared_norms)) / 2.0
                      - 2.0 * np.dot(linear, squared_norms)) / pairs
    return (float(np.dot(mu, mu) + mean_quadratic),
            max(0.0, float(central_second - mean_quadratic ** 2)))


def score_graph(adjacency, features, rank=8, max_pairs=None, random_state=0,
                svd_random_state=None, pair_random_state=None,
                svd_algorithm="randomized", svd_n_iter=5, pair_batch_size=65536,
                diversity_statistic="variance", reference="median", low_degree_fallback="zero",
                full_backend="stream"):
    """Implement Eqs. (1)-(13) and the explicitly selected ablations.

    Main setting: r=8, all unordered pairs, variance, median, zero fallback.
    SVD solver, seeds and population moment conventions are local choices:
    the paper does not publish enough detail to infer the author's defaults.
    """
    adjacency = binary_adjacency(adjacency)
    features = sp.csr_matrix(features, dtype=np.float64)
    n_nodes, n_features = features.shape
    if n_nodes < 1 or n_features < 1 or adjacency.shape != (n_nodes, n_nodes):
        raise ValueError("Expected a non-empty (N,D) feature matrix and (N,N) adjacency.")
    if not np.isfinite(features.data).all():
        raise ValueError("Features contain non-finite values.")
    _positive_integer(rank, "rank")
    _positive_integer(pair_batch_size, "pair_batch_size")
    _positive_integer(svd_n_iter, "svd_n_iter")
    if rank > min(n_nodes, n_features):
        raise ValueError("rank must not exceed min(N,D)={}.".format(min(n_nodes, n_features)))
    if svd_algorithm not in ("randomized", "arpack"):
        raise ValueError("svd_algorithm must be randomized or arpack.")
    if svd_algorithm == "arpack" and rank >= min(n_nodes, n_features):
        raise ValueError("ARPACK requires rank < min(N,D); use randomized for a full-dimension projection.")
    if max_pairs is not None:
        _positive_integer(max_pairs, "max_pairs")
    if diversity_statistic not in ("variance", "std", "mean"):
        raise ValueError("diversity_statistic must be variance, std or mean.")
    if full_backend not in ("stream", "moments"):
        raise ValueError("full_backend must be stream or moments.")
    if reference not in ("median", "mean") or low_degree_fallback not in ("zero", "median"):
        raise ValueError("Unsupported reference or low-degree fallback.")
    svd_seed = random_state if svd_random_state is None else svd_random_state
    pair_seed = random_state if pair_random_state is None else pair_random_state
    for seed in (svd_seed, pair_seed):
        if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or not 0 <= seed < 2**32:
            raise ValueError("Seeds must be integers in [0, 2**32).")

    # Eq. (1): no centering or label-dependent feature transformations.
    if n_features == 1:
        # TruncatedSVD requires at least two input columns. A one-column
        # rank-one representation is X itself, up to an irrelevant sign.
        projected = features.toarray()
        singular_values = [float(np.linalg.norm(projected))]
        projection_method = "exact one-column projection"
    elif rank == n_features:
        # sklearn's TruncatedSVD requires n_components < n_features even
        # though keeping all feature dimensions is a valid paper rank choice.
        # Use the exact full-column SVD; the resulting U Sigma differs from X
        # only by a rotation for cosine comparisons, but records the literal
        # projected representation and its singular values.
        left, singular_values_array, _ = np.linalg.svd(features.toarray(), full_matrices=False)
        projected = left[:, :rank] * singular_values_array[:rank]
        singular_values = singular_values_array[:rank].tolist()
        projection_method = "exact full-column NumPy SVD"
    else:
        projector = TruncatedSVD(n_components=rank, algorithm=svd_algorithm,
                                 n_iter=svd_n_iter, random_state=int(svd_seed))
        projected = projector.fit_transform(features)
        singular_values = projector.singular_values_.tolist()
        projection_method = "sklearn TruncatedSVD"
    # Eq. (4): normalized dot products; zero projected vectors stay zero.
    projected, zero_feature_rows = _normalize_projected(projected)

    rng = np.random.default_rng(int(pair_seed))
    degree = np.diff(adjacency.indptr)
    valid = degree >= 2  # Eq. (10).
    diversity = np.full(n_nodes, np.nan, dtype=np.float64)
    total_pairs, evaluated_pairs, enumerated_pairs, moment_nodes = 0, 0, 0, 0
    for node in np.flatnonzero(valid):
        neighbors = adjacency.indices[adjacency.indptr[node]:adjacency.indptr[node + 1]]
        possible = int(degree[node]) * (int(degree[node]) - 1) // 2
        total_pairs += possible
        evaluated_pairs += possible if max_pairs is None else min(possible, max_pairs)
        if full_backend == "moments" and (max_pairs is None or possible <= max_pairs):
            mean, variance = _full_pair_moments(projected[neighbors])
            moment_nodes += 1
        else:
            mean, variance = _pair_mean_variance(
                _similarity_batches(neighbors, projected, max_pairs, pair_batch_size, rng))
            enumerated_pairs += possible if max_pairs is None else min(possible, max_pairs)
        diversity[node] = variance if diversity_statistic == "variance" else (
            np.sqrt(variance) if diversity_statistic == "std" else mean)

    values = diversity[valid]
    median = float(np.median(values)) if values.size else 0.0
    graph_reference = (median if reference == "median" else float(values.mean())) if values.size else 0.0
    deviation = np.full(n_nodes, np.nan, dtype=np.float64)
    deviation[valid] = np.abs(values - graph_reference)  # Eq. (12).
    mean_deviation = float(deviation[valid].mean()) if values.size else 0.0
    std_deviation = float(deviation[valid].std(ddof=0)) if values.size else 0.0
    scores = np.zeros(n_nodes, dtype=np.float64)
    # Only a truly zero denominator is degenerate. Tiny positive dispersions
    # still contain a ranking signal; machine-epsilon gating erased it.
    if std_deviation > 0.0:
        scores[valid] = (deviation[valid] - mean_deviation) / std_deviation
    fallback = float(np.median(scores[valid])) if low_degree_fallback == "median" and values.size else 0.0
    scores[~valid] = fallback
    if not np.isfinite(scores).all():
        raise ValueError("Calibration produced non-finite scores.")

    metadata = {
        "n_nodes": int(n_nodes), "n_edges_directed": int(adjacency.nnz),
        "n_features": int(n_features), "rank": int(rank),
        "pair_mode": "full" if max_pairs is None else "sampled", "max_pairs": max_pairs,
        "pair_batch_size": int(pair_batch_size), "pair_sampler": "uniform-without-replacement/Floyd",
        "total_possible_neighbor_pairs": total_pairs, "evaluated_neighbor_pairs": evaluated_pairs,
        "full_backend": full_backend, "enumerated_neighbor_pairs": enumerated_pairs,
        "nodes_using_full_moment_identity": moment_nodes,
        "pair_count_scope": "evaluated pairs counts logical all-pair moments; enumerated pairs counts individual dot products",
        "n_valid_nodes_degree_ge_2": int(valid.sum()), "n_low_degree_nodes": int((~valid).sum()),
        "n_isolated_nodes": int((degree == 0).sum()), "n_degree_one_nodes": int((degree == 1).sum()),
        "degree_max": int(degree.max()), "degree_mean": float(degree.mean()),
        "diversity_statistic": diversity_statistic, "reference": reference,
        "diversity_median": median, "graph_reference": graph_reference,
        "deviation_mean": mean_deviation, "deviation_std_population": std_deviation,
        "degenerate_calibration": "no valid nodes" if not values.size else (
            "zero deviation standard deviation" if std_deviation == 0.0 else None),
        "low_degree_fallback": low_degree_fallback, "low_degree_fallback_value": fallback,
        "svd_algorithm": svd_algorithm, "svd_n_iter": int(svd_n_iter),
        "projection_method": projection_method, "svd_singular_values": singular_values,
        "zero_projected_feature_rows": zero_feature_rows,
        "zero_vector_cosine_convention": "zero normalized vector contributes dot-product similarity 0",
        "svd_random_state": int(svd_seed), "pair_sampling_random_state": int(pair_seed),
        "variance_ddof": 0, "calibration_std_ddof": 0,
        "median_convention": "NumPy: arithmetic average of central values for even length",
        "scoring_dtype": "float64",
    }
    return ScoreResult(scores, diversity, deviation, degree, valid, metadata)


def fit_score(adjacency, features, rank=8, max_pairs=None, random_state=0, **kwargs):
    """Compatibility wrapper returning the original (scores, metadata) tuple."""
    result = score_graph(adjacency, features, rank, max_pairs, random_state, **kwargs)
    return result.scores, result.metadata


def predict_anomalies(scores, threshold_lambda=1.0):
    """Eqs. (14)-(15): compute threshold over ALL nodes, strict score > tau.

    Predictions do not enter AUROC/AP. No label-based threshold tuning.
    """
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 1 or not scores.size or not np.isfinite(scores).all():
        raise ValueError("Scores must be a finite non-empty vector.")
    if not np.isfinite(threshold_lambda) or threshold_lambda < 0:
        raise ValueError("threshold_lambda must be finite and nonnegative.")
    mean, std = float(scores.mean()), float(scores.std(ddof=0))
    threshold = mean + threshold_lambda * std
    if not np.isfinite(threshold):
        raise ValueError("Threshold calculation overflowed.")
    return (scores > threshold).astype(np.int64), {
        "lambda": float(threshold_lambda), "threshold": threshold,
        "score_mean_all_nodes": mean, "score_std_all_nodes_population": std,
        "comparison": "score > threshold", "std_ddof": 0,
    }


def evaluate_scores(labels: Optional[np.ndarray], scores, mask=None):
    """Ranking evaluation after scoring; optional mask supports Table 17."""
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 1 or not scores.size or not np.isfinite(scores).all():
        raise ValueError("Scores must be a finite non-empty vector.")
    if labels is None:
        return {"AUROC": None, "AP": None, "evaluation": "skipped (no labels provided)"}
    labels = binary_labels(labels, scores.size)
    if mask is not None:
        mask = np.asarray(mask)
        if mask.shape != scores.shape or mask.dtype != np.bool_:
            raise ValueError("Evaluation mask must be boolean with one entry per node.")
        labels, scores = labels[mask], scores[mask]
    metrics = {"evaluated_nodes": int(scores.size), "positive_count": int((labels == 1).sum()),
               "positive_rate": float((labels == 1).mean()) if labels.size else None}
    if np.unique(labels).size < 2:
        metrics.update({"AUROC": None, "AP": None, "evaluation": "undefined (subset needs both label classes)"})
    else:
        metrics.update({"AUROC": float(roc_auc_score(labels, scores)),
                        "AP": float(average_precision_score(labels, scores)),
                        "evaluation": "continuous anomaly scores; positive_label=1"})
    return metrics
