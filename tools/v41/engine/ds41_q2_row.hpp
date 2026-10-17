// One GGML Q2_K block (scales[16], qs[64], fp16 d, fp16 dmin) -> 256 floats.
// Used for SSD Engram rows; no ggml/backend dependency, so the disk bench stays standalone.
#pragma once
#include "strata/artifact/dequant.hpp"

namespace strata::v41 {
inline void ds41_q2_row(const uint8_t * block, float * out) {
    const float d = fp16_to_fp32(read_u16(block + 80));
    const float m = fp16_to_fp32(read_u16(block + 82));
    for (int half = 0; half < 2; ++half) {
        const uint8_t * q = block + 16 + half * 32;
        for (int group = 0; group < 8; ++group) {
            const uint8_t sc = block[half * 8 + group];
            const float dl = d * (sc & 15), ml = m * (sc >> 4);
            for (int j = 0; j < 16; ++j)
                out[half * 128 + group * 16 + j] = dl * ((q[(group % 2) * 16 + j] >> (2 * (group / 2))) & 3) - ml;
        }
    }
}
} // namespace strata::v41
