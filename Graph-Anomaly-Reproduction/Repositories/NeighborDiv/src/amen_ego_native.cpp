#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>
#include <omp.h>

extern "C" void score_amen_ego(
    int64_t n,
    int64_t dim,
    const int64_t* indptr,
    const int64_t* indices,
    const double* degree,
    const double* attributes,
    const uint64_t* packed_adjacency,
    double* raw,
    int threads) {
    const int64_t words = (n + 63) / 64;
    double two_m = 0.0;
    for (int64_t i = 0; i < n; ++i) two_m += degree[i];
    if (two_m == 0.0) return;

    std::vector<double> row_volume(static_cast<size_t>(n * dim), 0.0);
#pragma omp parallel for schedule(static) num_threads(threads)
    for (int64_t p = 0; p < n; ++p) {
        const double* xp = attributes + p * dim;
        double* volume = row_volume.data() + p * dim;
        for (int64_t edge = indptr[p]; edge < indptr[p + 1]; ++edge) {
            const int64_t q = indices[edge];
            const double expected = std::min(1.0, degree[p] * degree[q] / two_m);
            const double surprise = 1.0 - expected;
            const double* xq = attributes + q * dim;
            for (int64_t f = 0; f < dim; ++f) {
                volume[f] += surprise * xp[f] * xq[f];
            }
        }
    }

    std::atomic<int64_t> completed(0);
#pragma omp parallel num_threads(threads)
    {
        std::vector<uint64_t> mask(static_cast<size_t>(words), 0);
        std::vector<double> internal_observed(static_cast<size_t>(dim));
        std::vector<double> internal_weighted(static_cast<size_t>(dim));
        std::vector<double> weighted_volume(static_cast<size_t>(dim));
        std::vector<double> weighted_degree_features(static_cast<size_t>(dim));
        std::vector<double> null_term(static_cast<size_t>(dim));
#pragma omp for schedule(dynamic, 4)
        for (int64_t ego = 0; ego < n; ++ego) {
            const int64_t begin = indptr[ego];
            const int64_t end = indptr[ego + 1];
            const int64_t ego_degree = end - begin;
            if (ego_degree == 0) {
                raw[ego] = 0.0;
                continue;
            }

            std::fill(mask.begin(), mask.end(), 0);
            mask[ego / 64] |= uint64_t(1) << (ego % 64);
            for (int64_t edge = begin; edge < end; ++edge) {
                const int64_t p = indices[edge];
                mask[p / 64] |= uint64_t(1) << (p % 64);
            }
            std::fill(internal_observed.begin(), internal_observed.end(), 0.0);
            std::fill(internal_weighted.begin(), internal_weighted.end(), 0.0);
            std::fill(weighted_volume.begin(), weighted_volume.end(), 0.0);
            std::fill(weighted_degree_features.begin(), weighted_degree_features.end(), 0.0);
            std::fill(null_term.begin(), null_term.end(), 0.0);

            const int64_t community_size = ego_degree + 1;
            double community_degree_sum = 0.0;
            for (int64_t member_index = 0; member_index < community_size; ++member_index) {
                const int64_t p = member_index == 0 ? ego : indices[begin + member_index - 1];
                community_degree_sum += degree[p];
                const double* xp = attributes + p * dim;
                const double* p_volume = row_volume.data() + p * dim;
                for (int64_t f = 0; f < dim; ++f) {
                    weighted_volume[f] += p_volume[f];
                    weighted_degree_features[f] += degree[p] * xp[f];
                }

                const uint64_t* row_bits = packed_adjacency + p * words;
                for (int64_t word_index = 0; word_index < words; ++word_index) {
                    uint64_t intersection = row_bits[word_index] & mask[word_index];
                    while (intersection != 0) {
                        const int bit = __builtin_ctzll(intersection);
                        const int64_t q = word_index * 64 + bit;
                        intersection &= intersection - 1;
                        if (q >= n) continue;
                        const double expected = std::min(1.0, degree[p] * degree[q] / two_m);
                        const double surprise = 1.0 - expected;
                        const double* xq = attributes + q * dim;
                        for (int64_t f = 0; f < dim; ++f) {
                            const double product = xp[f] * xq[f];
                            internal_observed[f] += product;
                            internal_weighted[f] += surprise * product;
                        }
                    }
                }
            }

            for (int64_t f = 0; f < dim; ++f) {
                null_term[f] = weighted_degree_features[f] * weighted_degree_features[f] / two_m;
            }
            const double internal_scale = static_cast<double>(community_size) * community_size;
            const double null_bound = community_degree_sum * community_degree_sum / two_m;
            double positive_norm_sq = 0.0;
            double maximum = -INFINITY;
            bool has_positive = false;
            for (int64_t f = 0; f < dim; ++f) {
                double external = weighted_volume[f] - internal_weighted[f];
                if (external < 0.0 && external > -1e-12) external = 0.0;
                if (external < 0.0) external = 0.0;
                const double denominator = internal_scale + null_bound;
                const double x_i = denominator > 0.0 ? (internal_observed[f] - null_term[f] + null_bound) / denominator : 0.0;
                const double x_e = weighted_volume[f] > 0.0 ? -external / weighted_volume[f] : 0.0;
                const double value = x_i + x_e;
                if (value > 0.0) {
                    has_positive = true;
                    positive_norm_sq += value * value;
                }
                maximum = std::max(maximum, value);
            }
            const double normality = has_positive ? std::sqrt(positive_norm_sq) : maximum;
            raw[ego] = -normality;

            const int64_t done = completed.fetch_add(1, std::memory_order_relaxed) + 1;
            if (done % 5000 == 0) {
#pragma omp critical
                std::fprintf(stderr, "AMEN-Ego native: %lld/%lld nodes\n",
                             static_cast<long long>(done), static_cast<long long>(n));
            }
        }
    }
}
