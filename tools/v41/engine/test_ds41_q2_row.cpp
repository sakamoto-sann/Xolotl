// Source-only gate until builds are authorized: compare the Engram row decoder with ggml's reference.
#include "ds41_q2_row.hpp"
#include "ggml.h"
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>

int main() {
    std::mt19937 rng(1017);
    for (int trial = 0; trial < 1024; ++trial) {
        std::array<uint8_t, 84> block;
        for (auto & b : block) b = (uint8_t) rng();
        // Finite half scales including subnormals; arbitrary nibble scales and every 2-bit lane.
        uint16_t d = (uint16_t) (rng() % 0x7c00), m = (uint16_t) (rng() % 0x7c00);
        std::memcpy(block.data() + 80, &d, 2); std::memcpy(block.data() + 82, &m, 2);
        float actual[256], expected[256];
        strata::v41::ds41_q2_row(block.data(), actual);
        ggml_get_type_traits(GGML_TYPE_Q2_K)->to_float(block.data(), expected, 256);
        for (int i = 0; i < 256; ++i)
            if (actual[i] != expected[i]) {
                std::fprintf(stderr, "trial %d lane %d: %g != %g\n", trial, i, actual[i], expected[i]); return 1;
            }
    }
    std::puts("Q2_K Engram rows match ggml");
    return 0;
}
