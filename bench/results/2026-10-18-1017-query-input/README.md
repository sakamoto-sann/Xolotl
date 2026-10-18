# 1017 shared-query precision diagnostic (CUDA, 2026-10-18)

The 35-tensor retrieval sidecar restores indexer and compressed-KV weights, but indexer queries also depend on the shared `attn_q_a` projection. Those eight upstream projections were still Q2_K. The opt-in `deepseek41-retrieval-qinput` profile adds only layers 2, 8, 14, 20, 24, 28, 32 and 36. These projections feed both attention and indexer; this experiment does not isolate the indexer alone.

Unset `STRATA_INDEXER_SIDECAR` and existing 24/35 profiles retain their previous behavior. There is no CUDA kernel or default-setting change. Expert weights and the main Q2_K embedding stay unchanged. This is a diagnostic profile, not a qualified default.

## Preparation

Using pinned checkpoint metadata and a new dedicated cache/output:

```sh
python tools/v41/v41_indexer_sidecar.py \
  --checkpoint-root /path/to/pinned-1017-metadata \
  --cache-dir /path/to/new-query-cache \
  --out /path/to/1017-RETRIEVAL-QINPUT-F32.gguf \
  --include-compressor --include-query-input
```

The source is `deepseek-ai/DeepSeek-V4.1-Flash-1017` at `1bae0e1088993b9b82de08f59de5be5a34e42170`. There are 59 source tensors/scales and 43 F32 output tensors. Downloads use pinned HTTPS byte ranges with checked `206` status, Content-Range, size and cached payload hashes. Complete source shards were not downloaded or independently hashed. Loader model labels and schema validation alone do not authenticate arbitrary sidecars.

Source artifact: 457,192,640 bytes, SHA256 `e9bb4d6e643b2e10e04efb7bd031415f7df36848fea97583870895b35ad6cdeb`. The eight new matrices have original shape `[1280,5120]` and require 200 MiB of F32 payload. They match a separate CUDA FP8 E4M3 reference lookup and E8M0 scaling bitwise for every value. The preceding 35 tensors remain bit-identical.

## Controlled comparison

The control artifact keeps those same 35 source tensors, but uses the native Q2_K q_a values dequantized to F32. It retains the information lost in the original quantization while changing the runtime arithmetic path. The source arm replaces them with original checkpoint values, also F32. Both arms therefore have identical tensor types, shapes and GPU payload size. Native quantized multiplication and F32 multiplication can use different activation and accumulation arithmetic; a control improvement alone would not prove a CUDA kernel defect.

Control: 457,192,768 bytes, SHA256 `630e434cf6ecc8b51d871b7cc1d8ad2207f9aa468705d1a18909c070ee41c2ba`. All output values checked against the native Q2 dequantization. Original Q2 shard identities were not independently authenticated. All eight full-matrix relative RMS errors versus restored source are 29.68–29.72%; this is not by itself evidence of an end-to-end quality cause.

Hardware: Ryzen 3900X, 32 GiB RAM, RTX 3090 24 GiB, Samsung 970 EVO. CUDA/gpu experts, no-enc-skip, 131072 context, prompt batch12288, prefill chunk32, arena16 GiB, margin1.5 GiB, temperature0/seed42, reasoning NONE, STEP_MAX96. DSpark and KEEP off; no other model loaded. Other GPU processes were retained.

## Validation and remaining gates

CUDA Release executable SHA256 `a97dae0f53d76e1b773bdc0a387569f6b25ccfd3903bd9473e1d8a73f7dab36e`.

Tiny CPU validator fixtures accept all three 24/35/43 profiles and reject 38 malformed artifacts, covering names, duplicate tensors, shapes, types, source/revision, architecture, counts, nonfinite values and truncated bounds. Valid fixtures include signed zero and a subnormal. Run with:

```sh
python tools/v41/tests/make_query_input_fixtures.py \
  --test /path/to/test_ds41_indexer_sidecar --out /path/to/new-fixtures
```

The new binary with the unchanged 24-tensor configuration reproduced the existing 85-token 4K code output byte for byte in105.449s. This is a compatibility check, not a speed improvement.

Actual DSH 68K prefill with two rapid steering prompts completed for both arms. Neither produced the required final JSON. These are full-workflow failures, not accuracy scores of zero out of three:

| Profile | DSH seconds | Result | Tool calls | Prefill seconds |
|---|---:|---|---:|---:|
| Prior source35, native Q2 q_a | 817.462 | Final JSON, 2/3 correct | 2 permitted reads | 590.424 |
| Source35 plus Q2-dequant F32 q_a control | 663.237 | Incorrectly asserted the JSON file was binary/unreadable | 0 | 593.250 |
| Source35 plus original-source F32 q_a | 647.821 | Incorrectly treated actual human followups as ledger prompt injection | 0 | 592.510 |

The final two failures were model assertions, not real tool failures or permission denials. Canonical system/user messages and tool schemas were identical in both new arms and the preceding source35 baseline (SHA256 `d72112c12b2fed3f3daa68e2c717a3e2e772f30724aa76cc50ffb83e16109499`). Native input length was68186 tokens. The harness permits bounded re-reads and evaluates the final JSON; one additional permitted read from a raw single-step replay is not a full-workflow failure.

The control's52,428,800 values also match ggml C++ `to_float` bitwise, independently of NumPy dequantization. A328.6 MiB CPU diagnostic checked this during3.2s of the source prefill, so these are not strict speed comparisons. The failed control trajectory spent35.665s of its51.67s native generation waiting for expert-file delivery (80.149 GiB logical demand); this is a bottleneck observation, not a useful-answer speed result.

Both new arms retained all human markers and had prompt-interruption advancement below1s. Peak API memory was23,800,676,352 and24,405,950,464 bytes respectively, swap0, no critical cgroup memory events. Owned-session cleanup and baseline restoration returned0. Long edit, holdout and generation-steering stages did not run after the final-answer gate failed. Neither profile was promoted. Restoring these eight q_a matrices alone did not solve the case; other main-model quantization effects are not ruled out. No stable-default or speed-improvement claim is made.

Separate read-only self-review only; an independent reviewer was unavailable. HIP/SYCL were not tested. Keep draft and do not merge on this evidence.
