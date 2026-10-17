// Synthetic delivery integration: out-of-order split reads, RAM/VRAM replacement and CUDA stream reuse.
// No model files. Compare every route-indexed output with the existing all-file GPU path, bit for bit.
#include "ds41_moe.hpp"
#include "ggml.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <thread>

using namespace strata::v41;
namespace {
constexpr int H = 5120, FF = 2304, NL = 2, NX = 16, K = 6;
void cuda_check(cudaError_t e) {
    if (e != cudaSuccess) { std::fprintf(stderr, "CUDA: %s\n", cudaGetErrorString(e)); std::exit(2); }
}
class SplitSource final : public Ds41BlobSource {
    size_t bytes_;
    std::vector<std::vector<uint8_t>> data_;
public:
    explicit SplitSource(const strata::kernels::cpu::NativeFmt & f) : bytes_(f.bytes), data_(NL * NX) {
        std::mt19937 rng(1017);
        std::normal_distribution<float> normal(0.f, 0.02f);
        for (auto & blob : data_) {
            blob.resize(bytes_);
            auto role = [&](int type, int rows, int cols, size_t offset) {
                std::vector<float> values((size_t) rows * cols);
                for (auto & x : values) x = normal(rng);
                ggml_quantize_chunk((ggml_type) type, values.data(), blob.data() + offset, 0, rows, cols, nullptr);
            };
            role(10, FF, H, 0); role(10, FF, H, f.up_off); role(11, H, FF, f.down_off);
        }
    }
    const char * kind() const override { return "synthetic delayed split reads"; }
    bool open(std::string &) override { return true; }
    size_t blob_bytes() const override { return bytes_; }
    void read(int64_t l, int64_t e, uint8_t * dst) const override {
        std::memcpy(dst, data_[(size_t) (l * NX + e)].data(), bytes_);
    }
    void read_part(int64_t l, int64_t e, int part, int parts, uint8_t * dst) const override {
        const size_t lo = (bytes_ * (size_t) part / (size_t) parts) & ~(size_t) 4095;
        const size_t hi = part + 1 == parts ? bytes_ :
            (bytes_ * (size_t) (part + 1) / (size_t) parts) & ~(size_t) 4095;
        std::this_thread::sleep_for(std::chrono::microseconds(300 * ((e + part) % 7)));
        std::memcpy(dst + lo, data_[(size_t) (l * NX + e)].data() + lo, hi - lo);
    }
};
struct Call {
    int layer, nt;
    bool device;
    std::vector<int32_t> ids;
    std::vector<float> w, x;
};
}
int main() {
    Ds41MoeGeom g;
    g.n_embd = H; g.n_ff = FF; g.n_layers = NL; g.n_experts = NX; g.top_k = K;
    g.gu_type = 10; g.d_type = 11; g.eclass.assign(NL * NX, 0); g.cls_gu = {10}; g.cls_d = {11};
    strata::kernels::cpu::NativeFmt layout;
    std::string error;
    if (!ds41_moe_blob_layout(g, layout, error)) { std::fprintf(stderr, "%s\n", error.c_str()); return 1; }
    SplitSource source(layout);
    std::vector<Call> calls;
    for (int step = 0; step < 12; ++step) for (int l = 0; l < NL; ++l)
        for (int nt : {1, 2}) for (bool device : {false, true}) {
            Call c{l, nt, device, {}, {}, {}};
            c.ids.resize(nt * K); c.w.resize(nt * K); c.x.resize(nt * H);
            for (int t = 0; t < nt; ++t) for (int k = 0; k < K; ++k) {
                c.ids[t * K + k] = (step * 3 + t + k) % NX;
                c.w[t * K + k] = (step + k) % 5 ? 0.125f : 0.f;
            }
            for (int i = 0; i < nt * H; ++i) c.x[i] = 0.03f * std::sin((float) (step + i));
            calls.push_back(std::move(c));
        }
    float * output = nullptr;
    cuda_check(cudaMalloc((void **) &output, 2 * H * sizeof(float)));
    std::vector<std::vector<float>> reference;
    int failures = 0;
    for (int mode = 0; mode < 4; ++mode) {
        // mode 0 = canonical all-file; 1 = existing cache; 2 = pipeline; 3 = pipeline + policy.
        Ds41MoeConfig cfg;
        cfg.threads = 2; cfg.pcie_frac = 1; cfg.pf_b = 2;
        cfg.no_cache = mode == 0; cfg.slots = mode == 0 ? 0 : 4;
        cfg.arena_gib = cfg.max_arena_gib = mode == 0 ? 0 : (4.0 * layout.bytes + 1) / 1073741824.0;
        cfg.vram_lru = mode != 0; cfg.file_pipeline = mode >= 2;
        cfg.arena_adapt = cfg.cache_freq_recency = mode == 3; cfg.cache_half_life = 4;
        Ds41MoeTier tier;
        if (!tier.init(g, &source, cfg, error)) { std::fprintf(stderr, "init: %s\n", error.c_str()); return 1; }
        std::vector<std::pair<int32_t, int32_t>> rank;
        for (int e = 0; e < NX; ++e) for (int l = 0; l < NL; ++l) rank.emplace_back(l, e);
        if (!tier.seed_from_ranked(rank, error)) { std::fprintf(stderr, "seed: %s\n", error.c_str()); return 1; }
        for (size_t i = 0; i < calls.size(); ++i) {
            const auto & c = calls[i];
            std::vector<float> got(c.nt * H);
            tier.prefetch(c.layer, c.ids.data(), K);
            bool ok;
            if (c.device) {
                ok = c.nt == 1 ? tier.run_to_dev(c.layer, c.ids.data(), c.w.data(), c.x.data(), output) :
                    tier.run_multi_to_dev(c.layer, c.nt, c.ids.data(), c.w.data(), c.x.data(), output);
                cuda_check(cudaMemcpy(got.data(), output, got.size() * sizeof(float), cudaMemcpyDeviceToHost));
            } else {
                ok = c.nt == 1 ? tier.run(c.layer, c.ids.data(), c.w.data(), c.x.data(), got.data()) :
                    tier.run_multi(c.layer, c.nt, c.ids.data(), c.w.data(), c.x.data(), got.data());
            }
            if (!ok) { std::fprintf(stderr, "mode %d call %zu failed\n", mode, i); return 1; }
            if (!std::all_of(got.begin(), got.end(), [](float v) { return std::isfinite(v); }) ||
                std::all_of(got.begin(), got.end(), [](float v) { return v == 0.f; })) {
                std::fprintf(stderr, "mode %d call %zu has nonfinite or all-zero output\n", mode, i); return 1;
            }
            if (mode == 0) reference.push_back(got);
            else if (std::memcmp(got.data(), reference[i].data(), got.size() * sizeof(float))) {
                std::fprintf(stderr, "mode %d call %zu nt %d device %d differs\n", mode, i, c.nt, c.device);
                ++failures;
            }
        }
        const auto st = tier.stats();
        if (cfg.file_pipeline && !st.file_ready_transfers) ++failures;
        if (cfg.cache_freq_recency && !st.arena_swaps) ++failures;
        std::printf("mode %d: %zu calls, %lld completion transfers, %lld RAM swaps, %d cumulative failures\n",
            mode, calls.size(), (long long) st.file_ready_transfers, (long long) st.arena_swaps, failures);
    }
    cuda_check(cudaFree(output));
    return failures ? 1 : 0;
}
