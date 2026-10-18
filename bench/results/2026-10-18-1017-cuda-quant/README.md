# CUDA quantization kernel qualification, 2026-10-18

Status: experimental, opt-in. CUDA remains enabled. No production default changed.

Host: Omarchy, RTX 3090 24 GiB (sm_86), Ryzen 9 3900X, 32 GiB RAM, CUDA 13 / GCC 14.4. Source base b24d3ddbe7733c19ddbb847f1b9cbf05254958c3. Build options STRATA_DSV41_FQ_OP=ON and STRATA_DSV41_HC_MIX=ON. Synthetic tests run under the existing GPU lock, with MemoryMax=4G, MemorySwapMax=0, CPUQuota=400%.

## Changes

- Add an opt-in float4 CUDA fake-quant kernel, retaining each 16/32-value quantization group and scalar rounding. STRATA_CUDA_FQ_VECTOR=1 selects it only for aligned tensors of at least 65,536 values. Default remains the existing scalar fused kernel.
- Add a synchronized CUDA graph benchmark, comparing composite graph against fused quantization. Input storage is explicitly retained because the allocator may otherwise reuse it in-place across iterations.
- Add long-context CUDA indexer tests and a bounded --rows option to the packed KV regression test.

Apply tools/v41/ggml/dsv41-fq-vector.patch inside third_party/llama.cpp with patch -p1, after the existing fake-quant patches. Configure STRATA_DSV41_FQ_OP=ON, then build bench_ds41_fq, test_ds41_indexer, test_ds41_fq and test_ds41_kvfp4. The patch is tracked because the vendored checkout is ignored.

## Correctness

- CUDA fused vs composite quantization: 2,097,152 synthetic values per mode, three modes, zero bit differences with vector mode enabled.
- Packed NVFP4 KV: 67,108,864 values; packed MXFP4 indexer keys: 16,777,216 values. Both bit-identical to fake-quant/BF16 storage at 131,072 rows.
- Indexer: rows 8,192 / 65,537 / 131,072 crossed with 1 / 3 / 16 queries. All scores bit-identical for packed MXFP4 versus BF16; sampled float64 references, known maximum near the end, and masked final row pass. Tested scalar and vector modes.
- Invalid --rows values and unknown arguments: five cases return 2 before backend initialization.
- These are synthetic indexer/storage checks. They do not prove correctness of all attention kernels, full-model retrieval, or DSH tool behavior.

## Microbenchmark

Same executable, vector flags 0 / 1 / 1 / 0 in fresh processes (ABBA). Each shape/mode uses ten warmups and five alternating-order timing samples; each sample contains 200 iterations, or 30 for the largest tensor. All 60 shape/mode results have zero bit differences.

At n=131,072, mean of two process medians:
- FP8: scalar 6.4634 us, vector 6.1245 us (5.2% lower).
- FP4 E4M3: scalar 6.5060 us, vector 6.1906 us (4.8% lower).
- FP4 Pow2: scalar 6.4321 us, vector 6.1872 us (3.8% lower).

At n=2,097,152, vector is approximately 0.0-0.8% slower depending on mode. Do not enable vector mode globally on this evidence. Small shapes retain the scalar dispatch. Times include synchronized backend graph dispatch, not kernel-only GPU time. The much larger composite-to-fused improvement is a separate comparison.

## Limits

No Nsight profiler installed. HIP and SYCL builds unavailable and not validated. No independent reviewer available with subagents disabled; a separate read-only self-review is recorded alongside the evidence. The earlier 67,597-token DSH retrieval returned one wrong record out of three; this optimization has not resolved that quality failure. DSpark remains nondefault.

## Full-model comparison

Same fused-vector executable (SHA256 f37158799879773a6b8c1b9812fd819a6fb7698b443181fb022843ce96e98891), vector mode OFF in both arms. Only STRATA_FQ_COMPOSITE differs (present=1 versus absent); STRATA_HC_MIX_COMPOSITE=1 in both arms. CUDA, Q2_K, 131,072 context capacity, arena16, prompt-batch12288, prefill-chunk32, enc-skip, route biases0, packed KV and all other settings held fixed. No DSpark. Temperature0, seed42, max192. One 4,060-token coding fixture, three requests per arm, baseline arm first. “Cold” means a fresh engine / prompt cache, not dropped operating-system SSD cache.

| Request | Composite seconds | Fused FQ seconds |
|---|---:|---:|
| Fresh engine | 100.6448 | 99.8378 |
| Warm repeat A | 26.1388 | 26.2944 |
| Warm repeat B | 24.9163 | 24.8492 |

Warm means 25.5275 versus 25.5718 seconds (fused 0.17% slower). Fresh-engine latency is 0.80% lower; one cold sample cannot establish a gain. All six responses are byte-identical (85 generated tokens, SHA256 4eba27540f840395a5ea6e92da11e3ce579886813f11534ecf9dcd23761dc5c9). Each generated lower_bound function passes 2,000 deterministic cases against bisect_left. Scope is one short coding output, not long-context retrieval or general coding quality.

API cgroup peak memory: 22,720,159,744 bytes composite / 22,879,424,512 bytes fused, swap peak zero both. The benchmark controller finished successfully and restored the original CUDA configuration byte-for-byte. GET /v1/models returned the loaded Q2_K model and 131,072 context capacity.

Decision: no production activation on these timings. Keep CUDA enabled, retain the experimental vector path as opt-in, and do not claim this resolves the earlier long-context quality failure. Single-op graph gains do not establish an end-to-end inference gain.
