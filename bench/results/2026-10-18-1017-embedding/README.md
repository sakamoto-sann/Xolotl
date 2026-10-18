# Optional 1017 BF16 embedding restoration (diagnostic)

RTX3090 24GiB, Ryzen3900X,32GiB RAM. The main model remains the existing1017 Q2_K shards and CUDA remains enabled. This is an opt-in precision diagnostic, not a qualified long-context default.

## What changes

`STRATA_EMBEDDING_SIDECAR=/absolute/path/1017-EMBEDDING-BF16.gguf` selects the original BF16 `embed.weight` rows instead of the main file's Q2_K token embeddings. The table stays memory-mapped on the host. Existing `upload_inputs` converts only requested rows to F32 and uploads the same input tensor; there is no whole-table GPU allocation. The mapped artifact remains owned for the dense model's lifetime. With the variable unset, the original pointer/type and graph path remain in use.

The loader requires one BF16 `token_embd.weight` tensor matching the actual trunk dimensions, the pinned model labels and source-shard hash label, valid byte bounds and finite BF16 values. Runtime schema checks do not authenticate arbitrary replacement files; the preparation command performs the actual full source-shard verification and checks every output payload bit.

## Source and preparation

The official source is `deepseek-ai/DeepSeek-V4.1-Flash-1017` revision `1bae0e1088993b9b82de08f59de5be5a34e42170`, shard `model-00002-of-00048.safetensors` (1,323,858,272bytes), SHA256 `43616eebd88dd4f1f1b642772356fc160478d1d9d04edd240431e7d065d2eda7`. This full shard was verified. That statement does not extend to the other source shards used for retrieval sidecars.

With NumPy and the bundled gguf-py available, choose an existing output directory:

```sh
python tools/v41/v41_embedding_sidecar.py \
  --source /weights/model-00002-of-00048.safetensors \
  --out /weights/1017-EMBEDDING-BF16.gguf
```

The command verifies size, full SHA256, source schema and file identity, preserves BF16 bits, checks finite values and compares all661,913,600 output values bitwise with the source. Existing output, staged output and manifest files are retained. No Torch, GPU initialization or additional model is needed. The measured output SHA256 is `94fd4fda4cc44561f007dc4e21602c7f349c26e3a1a5537c51a59e16f665ad6b`.

## Evidence and limits

- 77 sampled main-file embedding rows (special tokens and a seeded random sample) have28.68–33.36% relative RMS error versus the verified BF16 source. Best-fit scale0.9721–0.9979 rules out a gross source mapping/rescaling mismatch. This is weight error, not a measured end-to-end accuracy percentage.
- CUDA Release build passed. CPU validator accepted two valid fixtures and rejected14 malformed fixtures (source labels, architecture, name/type/shape, extra/missing tensor, truncation, NaN and signed infinities). A separate Mac/Clang check using a hand-written GGUF fixture writer accepted two and rejected15, including duplicates and finite subnormal/signed-zero edges.
- Feature-disabled4K code output was byte-identical to the previously validated85-token result (105.514s). With restored embedding and retrieval35 sidecar, the generated binary-search function passed2,000 functional cases (104.727s). These checks use different precision profiles and do not establish a speed improvement.
- Frozen68K replay and actual DSH long-context/steering/edit qualification are still pending. Previous retrieval-only profiles left one of three records wrong. Do not enable this feature based on a short code response.
- Review is a separate read-only self-review; an independent reviewer was unavailable. HIP and SYCL builds were unavailable. Keep the change draft; do not merge on this evidence alone.

CPU validator fixtures can be run without initializing the GPU:

```sh
cmake --build BUILD --target test_ds41_embedding_sidecar
python tools/v41/tests/make_embedding_sidecar_fixtures.py \
  --repo . --test BUILD/test_ds41_embedding_sidecar --out NEW_FIXTURE_DIRECTORY
```
