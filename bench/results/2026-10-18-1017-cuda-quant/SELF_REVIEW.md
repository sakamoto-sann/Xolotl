# Read-only self-review: CUDA kernel changes
Scope: tools/v41/engine/{bench_ds41_fq.cpp,test_ds41_indexer.cpp,test_ds41_kvfp4.cpp,v41_engine.cmake} and tools/v41/ggml/dsv41-fq-vector.patch, against b24d3dd. Excludes unrelated untracked tools/v41/serve/dsh.

No independent review service is available for patch review; subagents are disabled. This is a self-review, not an independent approval.

Inspected: dispatch defaults and alignment guards, group/shuffle widths, padded lanes before tail return, group divisibility, float rounding, graph allocation lifetime/input preservation, packed/BF16 layout and masks, sampled reference indexing, CLI parse bounds, CMake CUDA gating, and timing/quality claim boundaries. No actionable defects identified in this scope.

Evidence: CUDA build and scalar/vector quantization, packed KV and indexer checks pass; ABBA benchmark has zero bit mismatches; invalid CLI cases reject input. diff --check passes. Remaining gaps: HIP/SYCL unavailable; no full-model numerical identity guarantee from synthetic tests; no independent review; no full attention test; vector timing does not justify global enablement. Model-level timing/quality evidence must be assessed separately before deployment.
