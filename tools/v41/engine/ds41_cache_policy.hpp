// Backend-independent cache history. Placement only: never changes routes or weight bytes.
#pragma once
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

namespace strata::v41 {
class Ds41CacheHistory {
    struct Entry { double mass = 0; uint64_t at = 0; };
    int64_t layers_ = 0, experts_ = 0;
    double half_life_ = 64;
    std::vector<Entry> entries_;
    std::vector<uint64_t> clocks_;
public:
    bool reset(int64_t layers, int64_t experts, double half_life) {
        if (layers <= 0 || experts <= 0 || !std::isfinite(half_life) || half_life <= 0 ||
            layers > std::numeric_limits<int64_t>::max() / experts) return false;
        layers_ = layers; experts_ = experts; half_life_ = half_life;
        entries_.assign((size_t) layers * (size_t) experts, {});
        clocks_.assign((size_t) layers, 0);
        return true;
    }
    // One token = top_k route entries. Advance per token even for a multi-token pass,
    // so batching the same route stream does not change the history.
    void observe(int64_t layer, const int32_t * ids, int64_t n, int64_t top_k) {
        if (layer < 0 || layer >= layers_ || !ids || top_k <= 0 || n % top_k) return;
        auto & now = clocks_[(size_t) layer];
        for (int64_t i = 0; i < n; ++i) {
            if (i % top_k == 0) ++now;
            const int32_t expert = ids[i];
            if (expert < 0 || expert >= experts_) continue;
            auto & e = entries_[(size_t) (layer * experts_ + expert)];
            e.mass = e.mass * std::exp2(-(double) (now - e.at) / half_life_) + 1;
            e.at = now;
        }
    }
    double score(int64_t layer, int64_t expert) const {
        const auto & e = entries_[(size_t) (layer * experts_ + expert)];
        return e.mass * std::exp2(-(double) (clocks_[(size_t) layer] - e.at) / half_life_);
    }
    bool worth_replacing(int64_t layer, int64_t incoming, int64_t resident) const {
        return score(layer, incoming) > score(layer, resident);
    }
};
} // namespace strata::v41
