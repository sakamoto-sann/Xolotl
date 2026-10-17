# 1017 Q2_K runtime qualification

Measured on 2026-10-17, with the complete 1017 Q2_K model. CUDA build, real-file input gates and short generation
passed. The combined opt-in path reached 2.00 tokens/s with a 12 GiB RAM arena, versus 0.76 tokens/s for default
placement. This is one short prompt per configuration, not a production or broad model-quality qualification.

## Inputs and hardware

- Performance implementation: `b1e0db0` (before the later Engram counter-only diagnostic addition).
- Benchmark executable SHA-256: `4add175a2749c5a543e2270502f92fc0af74226d78623a8c86c7899d2af15a3c`.
- ggml: llama.cpp `3cf03257f219afbe7334045ff7c6a06ac68c627d` plus `tools/xolotl/ggml/xolotl-ggml.patch`.
- Model: [pinned ten-file Q2_K artifact](DEEPSEEK_1017_Q2_K.md), 264,515,416,800 bytes. All ten downloaded
  files passed size/SHA-256 checks against the pinned manifest; real-file preflight checked 1,046 tensors.
- RTX 3090, 24,576 MiB VRAM; driver 610.57.04; Ryzen 9 3900X; about 31 GiB system RAM; Samsung 970 EVO SSD.
- Isolated CUDA 13.0.2 and GCC 14.4.0. System compiler, CUDA and Comfy installation were not modified.

## Measured generation

The arithmetic smoke answered `42` followed by EOS. All eight 32-token code trials produced identical token IDs,
including across RAM sizes. The code output is a truncated function prefix/comment, not a complete executable
coding solution. A matching-reference logits/routing comparison and long-context quality evaluation remain open.

| Configuration | RAM arena GiB | Decode tokens/s | First token s | Process HWM GiB | Cgroup anon peak GiB | Minimum system available GiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 8 | 0.765 | 42.67 | 11.123 | 8.461 | 14.124 |
| pipeline-policy | 8 | 1.916 | 26.70 | 11.447 | 8.502 | 14.178 |
| baseline | 12 | 0.760 | 44.71 | 15.230 | 12.460 | 10.093 |
| pipeline-only | 12 | 0.847 | 41.86 | 13.710 | 12.445 | 10.254 |
| pipeline-adaptive-lru | 12 | 1.959 | 28.73 | 15.118 | 12.500 | 10.223 |
| pipeline-policy | 12 | 1.999 | 28.30 | 15.164 | 12.500 | 10.105 |
| baseline | 16 | 0.761 | 47.08 | 18.091 | 16.467 | 6.125 |
| pipeline-policy | 16 | 2.011 | 30.90 | 18.091 | 16.510 | 6.174 |

`pipeline-only` adds `--file-pipeline`; `pipeline-adaptive-lru` also enables `--arena-adapt`;
`pipeline-policy` uses `--file-pipeline --cache-policy freq-recency` (which enables arena adaptation).
At 12 GiB, the combined path is 2.63x the baseline. Adaptive LRU already reaches 1.96 tokens/s; the frequency
heuristic's additional difference is only about 2% in this single trial and is not an established repeatable gain.
Most of this observed gain comes with adaptive cache placement, so it must not all be attributed to the pipeline
or frequency heuristic. The 16 GiB arena adds about 4 GiB anonymous memory with less than 1% more decode speed.
The 12 GiB combined configuration is the provisional balance for this machine.

All trials hit the 18 GiB cgroup charged-memory ceiling (including file cache), with swap disabled. RSS/HWM,
anonymous memory and cgroup charged memory are distinct measurements; the arena flag is not total RAM use.
No trial was stopped for the 4 GiB system-memory floor. Total GPU use was about 21 GiB including idle Comfy;
about 16.2 GiB was allocated to 1,358–1,359 expert slots. At 12 GiB the combined run reports 63.9% VRAM hits
versus 9.6% for baseline. Process physical-read counters include startup/prefill and must not be read as decode-only
SSD bandwidth. Raw metric values and token IDs are in [measurement data](DEEPSEEK_1017_MEASUREMENTS.json).

## Measurement procedure

Each case starts a fresh process, using the same non-thinking 24-token Python merge-function prompt, 32 output
tokens, temperature 0, seed 42, four CPU threads and context 4,096. Decode rate is `(tokens - 1) / (last timestamp -
first timestamp)`. First-token latency includes model loading, cache seeding and prefill, excluding tokenizer setup.

Flags shared by all trials: `--llama-q2-k --backend cuda --experts gpu --pcie 1 --vram-margin 3
--prompt-batch 0 --prefill-chunk 1 --no-chunk-prefetch --no-enc-skip --route-bias 0 --route-bias-file 0`.
The learned Engram n-gram order remains four. Tables stay in read-only shard files on SSD, with
`STRATA_ENGRAM_CACHE_ROWS=131072`; no DSpark/MTP or n-gram speculative decoding was enabled.

File-specific `POSIX_FADV_DONTNEED` was requested before each case. This is advisory and does not prove a fully
cold device/cache state. No global cache drop or Comfy model eviction was performed. Each inference uses the
shared GPU lock and memguard: MemoryMax 18 GiB, MemorySwapMax 0, at least 22 GiB available before starting,
4 GiB available-memory floor and 30-minute timeout. Comfy stayed running; its queue was checked idle before each
trial. This does not qualify simultaneous active Comfy generation.

Process RSS/HWM, anonymous/file memory, I/O and GPU figures are sampled. Total cgroup peak comes from kernel
`memory.peak`. Existing swap usage is unrelated to these no-swap inference scopes.

The synthetic expert tests measured relative L1 error versus dequantized-weight F32 of 0.0221 for CPU and 0.0125
for GPU, within existing activation-rounding tolerances. CPU/GPU outputs differ, so `--pcie 1` kept the performance
comparison on GPU arithmetic. Synthetic parity and matching generated prefixes do not prove full model quality.
The pipeline tests use delayed memory-backed aligned reads; the full-model trials exercise actual shard files.

A startup failure exposed uniform VRAM cache sizing ignoring the byte budget. The fix caps uniform slot count
using the aligned stride, including deferred setup and sub-slot budgets. Q2_K default seeding is now interleaved
across layers; explicit ranks and native GSQ seeding remain unchanged. Performance trials above all use the fix.
An earlier instrumentation/startup attempt was excluded from the table.

## Engram file-offload diagnostic

The follow-up counter-only build at `d0e39b0` has SHA-256
`b7237a0a99cd4afadbe7b1a4a57b65c41f11011c5e33ca491244e5133fc3b128`.
It answered the arithmetic prompt with `42` and EOS using the 12 GiB combined configuration. Its completed-pass
Engram counters report 76,608 file-read payload bytes, 912 read groups and 912 row-cache misses (zero hits in
this short unique-row sample). This confirms the file-backed row path was exercised. OS page cache may satisfy
those reads; these counters do not establish physical-device traffic or a measured row-cache speedup.
The counters are cumulative over prefill and decode, and are printed to stderr only for the direct Q2_K mode.
This diagnostic run is separate from the frozen eight-case performance cohort above.

## Completed checks

| Check | Result |
| --- | --- |
| CMake configuration with tests enabled | Passed after avoiding a duplicate `ds4_geometry_test` registration |
| `ds41_generate` CUDA executable and `ds41_moe_cpu` library | Built successfully, sm_86, Release |
| Geometry, row-decoder, delivery, ops, dense and CPU/GPU MoE test executables | Built; real-file geometry passed; other fixture-dependent tests remain unrun |
| `test_ds41_delivery` | Passed on Linux/GCC 14 and macOS/Clang; macOS ThreadSanitizer also passed |
| `test_ds41_q2_row` | Passed on Linux against ggml: 1,024 Q2_K blocks, all 256 lanes equal |
| `test_ds41_q2_expert --synthetic q2_K/q3_K` | Passed on the 3090 at 5,120 / 2,304 dimensions: GPU weight dequantization equals ggml on all 35,389,440 values, and single/multi-token and decode-once/per-entry checks passed |
| Original `native_expert_parity --synthetic q4_K/q8_0` | Passed with the original 2,560 / 640 defaults after parameterizing test dimensions |
| `test_ds41_pipeline` | Passed at the 1017 expert dimensions (5,120 / 2,304), using two synthetic layers and 16 experts each. Six configurations x 96 calls, host/device outputs and single/multi-token calls matched the existing all-file GPU path bit for bit; 540 pipeline transfers, and 432 transfers / 68 RAM swaps with policy enabled |
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
The rebuilt dense object is newer than that source edit. The real model loaded successfully, including 44 BF16 tensors expanded to 300.31 MiB.

No independent reviewer was available with delegation disabled. A separate read-only self-review covered the
changed configuration and conversion branches. HIP/SYCL toolchains and execution have not been validated.
