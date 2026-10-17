# 1017 Q2_K runtime qualification

Status at 2026-10-17 21:50 JST: CUDA engine build and helper tests passed; full-model generation and performance
measurements are waiting for the complete, verified model download. The user has lifted the earlier build/inference
hold and made Xolotl the priority. This document does not yet establish working generation or a speed improvement.

## Inputs and hardware

- Source tree: `13a627af3c40099f05d7b1fe80715ce72d8a6534`. The remote build tree's tracked files match this commit.
- `ds41_generate` SHA-256: `a08e374712924735bf803cbacac5ad562b257cd0e1c054c18745be0159a3961e`.
- ggml: llama.cpp `3cf03257f219afbe7334045ff7c6a06ac68c627d` plus `tools/xolotl/ggml/xolotl-ggml.patch`.
- Model: the pinned ten-file artifact in [the Q2_K input contract](DEEPSEEK_1017_Q2_K.md), 264,515,416,800 bytes.
- RTX 3090, 24,576 MiB VRAM; NVIDIA driver 610.57.04; Ryzen 9 3900X; about 31 GiB system RAM; Samsung 970 EVO SSD.
- Isolated CUDA 13.0.2 components and GCC 14.4.0. The missing Profiler API headers were added from the same NVIDIA
  redistribution manifest after SHA-256 verification. No system compiler or Comfy installation was modified.

## Completed checks

| Check | Result |
| --- | --- |
| CMake configuration with tests enabled | Passed after avoiding a duplicate `ds4_geometry_test` registration |
| `ds41_generate` CUDA executable and `ds41_moe_cpu` library | Built successfully, sm_86, Release |
| Geometry, row-decoder, delivery, ops, dense and CPU/GPU MoE test executables | Built; model/fixture-dependent tests are not yet run |
| `test_ds41_delivery` | Passed on Linux/GCC 14 and macOS/Clang; macOS ThreadSanitizer also passed |
| `test_ds41_q2_row` | Passed on Linux against ggml: 1,024 Q2_K blocks, all 256 lanes equal |
| Read-pool lifetime probe | Extracted the actual pool into a temporary probe: 1,000 alternating zero/tiny/large batches, 32 workers, macOS ThreadSanitizer; passed |
| Python input tests | 15 passed |
| Python checkpoint suite | 16 discovered: 10 passed, 6 skipped because optional dependencies/fixtures were unavailable |
| Diff whitespace check and pre-push secret scan | Passed |

Configuration uses `STRATA_ENABLE_CUDA=ON`, `STRATA_GGML_CUDA=ON`, `STRATA_BUILD_TESTS=ON`,
`STRATA_GGML_STATIC_DEPS=OFF`, `CMAKE_CUDA_ARCHITECTURES=86` and an explicit patched ggml checkout. The new
static-dependency option defaults ON to retain existing configuration behavior; OFF allows the available shared
cuBLAS installation. Build concurrency was bounded at four jobs, MemoryMax 12 GiB and no swap.

Source inspection also found that 1017 BF16 expansion incremented the optional requantization counters and could
ask for the name of unset `GGML_TYPE_COUNT`. BF16 expansion now has separate counters and a separate message.
The rebuilt dense object is newer than that source edit. The actual model load remains a pending check.

No independent reviewer was available with delegation disabled. A separate read-only self-review covered the
changed configuration and conversion branches. HIP/SYCL toolchains and execution have not been validated.

## Measurement procedure prepared

The controller waits for all ten size/SHA-256-verified files and a successful build, then runs Python header preflight
and the C++ geometry gate. An exact-routing arithmetic smoke must answer 42 before performance trials can start.

The trials use a non-thinking Python code prompt, temperature 0, seed 42, four CPU threads, context 4,096,
`--prompt-batch 0 --prefill-chunk 1 --no-chunk-prefetch --no-enc-skip --route-bias 0 --route-bias-file 0`.
The RAM arena is compared at 8 and 12 GiB, each with default placement versus
`--file-pipeline --cache-policy freq-recency`. The VRAM margin is 3 GiB. The Engram cache is bounded with
`STRATA_ENGRAM_CACHE_ROWS=131072`; its tables stay on SSD and learned n-gram/hash metadata is unchanged.
DSpark and n-gram speculative decoding are not part of this initial comparison.

Each inference runs under the shared GPU lock and memguard with MemoryMax 18 GiB, MemorySwapMax 0, at least
22 GiB available before starting and a 4 GiB system-available floor. Existing Comfy services remain running and
must have an empty queue before a trial starts. A 30-minute per-trial timeout is enforced.

Before each run, file-specific `POSIX_FADV_DONTNEED` is requested for this new model only. This is an advisory cold
cache request, not proof that every page is evicted. No global cache drop or Comfy model eviction is performed.

Collected measurements distinguish:

- Decode tokens/s from streamed token timestamps; first-token time includes model loading and cache seeding.
- Process RSS/HWM, kernel cgroup `memory.peak`, and sampled cgroup anonymous/file-cache components.
- Sampled process physical read bytes and total GPU memory, including the existing Comfy allocation.
- Engine logs with logical expert payload counters, which are not physical SSD traffic measurements.

Opt-in generated IDs must match the same-arena baseline before the controller marks the comparison complete.
Results remain a short single-prompt experiment; they do not establish broad coding quality or steady-state
production performance. Final measurements will be added after the jobs finish.
