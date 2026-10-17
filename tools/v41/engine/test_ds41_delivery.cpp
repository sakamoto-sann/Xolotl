// Backend-independent tests, prepared for a later authorized build. No model or CUDA initialization.
#include "ds41_cache_policy.hpp"
#include "ds41_read_completions.hpp"
#include <array>
#include <cmath>
#include <cstdio>
#include <limits>
#include <thread>
#include <vector>

int main() {
    int failures = 0;
    auto check = [&](bool ok, const char * name) {
        std::printf("%s %s\n", ok ? "ok" : "FAIL", name); failures += !ok;
    };
    using namespace strata::v41;
    Ds41CacheHistory one, batch;
    check(one.reset(2, 4, 4) && batch.reset(2, 4, 4), "history geometry");
    const int32_t trace[] = {0, 1, 0, 1, 0, 1, 2, 3};
    batch.observe(0, trace, 8, 2);
    for (int i = 0; i < 8; i += 2) one.observe(0, trace + i, 2, 2);
    for (int e = 0; e < 4; ++e)
        check(one.score(0, e) == batch.score(0, e), "batching preserves token history");
    check(one.score(0, 0) > one.score(0, 2), "frequent recent expert beats a one-off");
    check(!one.worth_replacing(0, 2, 0), "one-off cannot evict a hotter resident");
    const double before = one.score(0, 0);
    const int32_t quiet[] = {3, 3};
    for (int i = 0; i < 16; ++i) one.observe(1, quiet, 2, 2);
    check(one.score(0, 0) == before, "layer-local aging");
    for (int i = 0; i < 32; ++i) one.observe(0, quiet, 2, 2);
    check(std::fabs(one.score(0, 0) - before / 256) < 1e-12, "old popularity decays by configured half-life");
    check(one.score(0, 3) > one.score(0, 0), "new workload displaces old popularity");
    check(one.worth_replacing(0, 3, 0), "admit the new hot expert after old popularity ages out");
    check(!one.reset(1, 4, 0) && !one.reset(1, 4, std::numeric_limits<double>::infinity()), "invalid decay rejected");
    check(!one.reset(0, 4, 1) && !one.reset(2, 0, 1), "invalid dimensions rejected");

    Ds41ReadCompletions<8> ready;
    std::array<int, 8> counts{};
    counts[0] = 2; counts[1] = 1;
    check(ready.reset(3, counts), "bounded read batch");
    check(ready.wait_next() == 2, "stashed expert ready without disk read");
    check(ready.part_done(0), "first partial read");
    check(ready.part_done(1) && ready.wait_next() == 1, "later expert passes an unfinished earlier expert");
    check(ready.part_done(0) && ready.wait_next() == 0, "notify only after all parts");
    check(!ready.part_done(0) && !ready.part_done(9), "duplicate/out-of-range completion rejected");
    check(ready.wait_next() == -1, "drained queue does not hang");
    check(!ready.reset(9, counts), "capacity limit");

    counts.fill(4);
    check(ready.reset(8, counts), "full bounded batch");
    std::array<int, 32> bytes{};
    std::vector<std::thread> workers;
    for (int i = 31; i >= 0; --i)
        workers.emplace_back([&, i] { bytes[(size_t) i] = i + 1; ready.part_done((size_t) i / 4); });
    std::array<bool, 8> seen{};
    for (int i = 0; i < 8; ++i) {
        const int b = ready.wait_next();
        check(b >= 0 && b < 8 && !seen[(size_t) b], "exactly one completion per expert");
        if (b < 0 || b >= 8) continue;
        seen[(size_t) b] = true;
        for (int p = 0; p < 4; ++p) {
            const int at = b * 4 + p;
            check(bytes[(size_t) at] == at + 1, "all reader writes visible before H2D submission");
        }
    }
    for (auto & worker : workers) worker.join();
    return failures ? 1 : 0;
}
