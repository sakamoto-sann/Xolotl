# 1017 retrieval precision experiments

RTX3090 24GiB, Ryzen3900X,32GiB RAM,970EVO NVMe. Main model remains the same Q2_K1017 shards. CUDA stays enabled; no other model is loaded.

## Opt-in precision sidecar

STRATA_INDEXER_SIDECAR selects an additional small GGUF, with24 F32 indexer matrices/norms. The experimental --include-compressor export also restores11 compressed-KV projection/norm tensors (35 total). The loader requires pinned1017 source metadata, exact main-model names/shapes, F32 finite data, and bounded storage; it rejects duplicate names. The unset path is unchanged. Original Q2 allocations remain resident in addition to the replacement tensors.

Pinned source: deepseek-ai/DeepSeek-V4.1-Flash-1017@1bae0e1088993b9b82de08f59de5be5a34e42170. HTTPS tensor byte ranges are checked for206 status, exact Content-Range and size. Per-tensor hashes are recorded. Full safetensors shard hashes are NOT verified. This is not a full-shard checkpoint verification claim.

Prepare without GPU initialization or full-shard downloads:

    python tools/v41/v41_indexer_sidecar.py --checkpoint-root PINNED_METADATA_DIR --cache-dir NEW_RANGE_CACHE --out NEW_INDEXER.gguf
    # Experimental compressed-KV restoration:
    python tools/v41/v41_indexer_sidecar.py --checkpoint-root PINNED_METADATA_DIR --cache-dir NEW_RANGE_CACHE --out NEW_RETRIEVAL.gguf --include-compressor

Existing outputs/staged outputs are retained. Cached headers must carry matching source provenance; altered cached tensors fail their manifest hashes.

Indexer file:174067648 bytes, SHA256 dede1dac447c332ba4d3a55a276a41af1fd7a258fc3ef2d32f4e9ee53d972c25.
Combined file:247476928 bytes, SHA256 2e13aab914c7eb30cc119a774cb2c78dfd8cb51b18a5335add1c85bbba97cf16.
All24/35 output tensors are bit-identical to an independent CUDA13 host FP8 conversion/BF16 reference. All254 finite FP8 encodings match; all8 restored norms match the main GGUF. The sampled first8 rows of27 Q2 matrices have29.15–31.73% relative RMS error against the checkpoint. This quantization error alone does not establish end-to-end causation.

Validation:

    cmake --build BUILD --target ds41_generate test_ds41_indexer_sidecar
    python tools/v41/test_indexer_sidecar.py --validator BUILD/test_ds41_indexer_sidecar

For each24/35-tensor architecture, valid input is accepted and10 invalid fixtures rejected (shape, type, missing, unknown, nonfinite, source, revision, architecture, truncation, duplicate). These validator tests initialize no GPU.

## Actual DSH results so far

DSH0.2.1-alpha.5, production provider131072 context,8192 output limit,30min idle timeout. Initial native request67579 tokens. Same model, original persona, temperature0/seed42, CUDA, no DSpark. Read-only catalog and dispatch policy.

| Configuration | Exact requested records | DSH total | Initial prefill |
|---|---:|---:|---:|
| Original Q2_K + enc-skip |0/3(placeholders)|393.817s|295.492s|
| F32 indexer + enc-skip |2/3|472.047s|293.359s|
| F32 indexer + no-enc-skip |3/3|757.949s|581.492s|

All these turns completed with zero tool/attempt errors. The indexer-only + enc-skip case copied R01628 into the R02824 output; it is a failure. Disabling enc-skip changes the result to exact3/3 but nearly doubles prefill time. A same-session holdout at 68142+ tokens also returned three different requested records correctly in221.702s, using two allowed reads and no tool/attempt errors (six distinct records across the two turns). This is limited fixture evidence, not a general long-context quality claim. Combined-sidecar runtime qualification remains pending; do not activate it based only on conversion/kernel tests.

HIP/SYCL builds unavailable. Keep draft; no independent reviewer was available with subagents disabled. A separate read-only self-review covered the loader ownership, metadata/shape/bounds/finite checks, source-range provenance and existing negative fixtures; no independent review is claimed.
